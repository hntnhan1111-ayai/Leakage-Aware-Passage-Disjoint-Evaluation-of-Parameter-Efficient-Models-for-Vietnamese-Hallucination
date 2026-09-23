from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import LABELS
from .metrics import save_metrics
from .prompts import build_prompt


@dataclass
class EncodedPrompt:
    input_ids: list[int]
    field_token_lengths: dict[str, int]
    field_tokens_used: dict[str, int]


def load_tokenizer_or_processor(model_ref: str):
    from transformers import AutoProcessor, AutoTokenizer

    errors: list[str] = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    except Exception as exc:
        errors.append(f"AutoTokenizer: {type(exc).__name__}: {exc}")
        try:
            processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
            tokenizer = getattr(processor, "tokenizer", processor)
        except Exception as processor_exc:
            errors.append(f"AutoProcessor: {type(processor_exc).__name__}: {processor_exc}")
            raise RuntimeError("Tokenizer/processor load failed: " + " | ".join(errors)) from processor_exc
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_model_class_candidates():
    import transformers

    names = ["AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"]
    return [getattr(transformers, name) for name in names if hasattr(transformers, name)]


def load_base_model(model_ref: str, *, quantized: bool, torch_dtype: str = "bfloat16"):
    import torch
    from transformers import BitsAndBytesConfig

    dtype = getattr(torch, torch_dtype)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    errors = []
    for model_class in _load_model_class_candidates():
        try:
            return model_class.from_pretrained(model_ref, **kwargs)
        except TypeError as exc:
            fallback = dict(kwargs)
            fallback["dtype"] = fallback.pop("torch_dtype")
            try:
                return model_class.from_pretrained(model_ref, **fallback)
            except Exception as retry_exc:
                errors.append(f"{model_class.__name__}: {type(retry_exc).__name__}: {retry_exc}")
        except Exception as exc:
            errors.append(f"{model_class.__name__}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Model load failed: " + " | ".join(errors))


def attach_adapter(model, adapter_dir: str | Path):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)


def _trim_head_tail(tokens: list[int], budget: int) -> list[int]:
    if len(tokens) <= budget:
        return tokens
    if budget <= 0:
        return []
    head = max(1, int(round(budget * 0.60)))
    tail = max(0, budget - head)
    return tokens[:head] + (tokens[-tail:] if tail else [])


def encode_prompt_fields(tokenizer, row: dict[str, Any], *, condition: str, max_prompt_tokens: int) -> EncodedPrompt:
    rendered = build_prompt(row, condition=condition)
    markers = ["CONTEXT:\n", "\n\nUSER_PROMPT:\n", "\n\nMODEL_RESPONSE:\n", "\n\nLabel:"]
    context = str(row.get("context", "")) if condition in {"full", "no_prompt", "shuffled_context"} else "[REMOVED]"
    user_prompt = str(row.get("prompt", "")) if condition in {"full", "shuffled_context"} else "[REMOVED]"
    response = str(row.get("response", ""))
    prefix = rendered.split("CONTEXT:\n", 1)[0]

    def ids(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
    marker_ids = [ids(marker) for marker in markers]
    context_ids, prompt_ids, response_ids = ids(context), ids(user_prompt), ids(response)
    fixed = len(prefix_ids) + sum(len(value) for value in marker_ids)
    available = max(24, int(max_prompt_tokens) - fixed)

    prompt_budget = min(len(prompt_ids), max(16, int(available * 0.15)))
    response_budget = min(len(response_ids), max(32, int(available * 0.25)))
    context_budget = max(0, available - prompt_budget - response_budget)
    if len(context_ids) < context_budget:
        spare = context_budget - len(context_ids)
        context_budget = len(context_ids)
        add_response = min(spare, len(response_ids) - response_budget)
        response_budget += add_response
        spare -= add_response
        prompt_budget += min(spare, len(prompt_ids) - prompt_budget)
    used_context = _trim_head_tail(context_ids, context_budget)
    used_prompt = prompt_ids[:prompt_budget]
    used_response = response_ids[:response_budget]
    input_ids = (
        prefix_ids
        + marker_ids[0]
        + used_context
        + marker_ids[1]
        + used_prompt
        + marker_ids[2]
        + used_response
        + marker_ids[3]
    )
    input_ids = input_ids[: int(max_prompt_tokens)]
    return EncodedPrompt(
        input_ids=input_ids,
        field_token_lengths={"context": len(context_ids), "prompt": len(prompt_ids), "response": len(response_ids)},
        field_tokens_used={"context": len(used_context), "prompt": len(used_prompt), "response": len(used_response)},
    )


def discover_lora_target_modules(model, *, text_only: bool = True) -> list[str]:
    import torch

    allowed = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    blocked_fragments = {"vision", "image", "audio", "projector", "multimodal", "multi_modal", "clip"}
    targets: set[str] = set()
    for name, module in model.named_modules():
        lowered = name.lower()
        if text_only and any(fragment in lowered for fragment in blocked_fragments):
            continue
        if not (isinstance(module, torch.nn.Linear) or module.__class__.__name__ in {"Linear4bit", "Linear8bitLt"}):
            continue
        tail = name.split(".")[-1]
        if tail in allowed:
            targets.add(name if text_only else tail)
    if not targets:
        raise RuntimeError("No supported LoRA target modules found")
    return sorted(targets)


class LabelCompletionDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_seq_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        label_ids = self.tokenizer(" " + str(row["label"]), add_special_tokens=False).input_ids
        encoded = encode_prompt_fields(
            self.tokenizer,
            row,
            condition="full",
            max_prompt_tokens=max(32, self.max_seq_length - len(label_ids)),
        )
        input_ids = (encoded.input_ids + label_ids)[: self.max_seq_length]
        labels = ([-100] * len(encoded.input_ids) + label_ids)[: self.max_seq_length]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def collate_label_completion(features, tokenizer, *, add_zero_mm_token_type_ids: bool = False):
    import torch

    max_len = max(len(item["input_ids"]) for item in features)
    pad_id = tokenizer.pad_token_id
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        padding = max_len - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_id] * padding)
        batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
        batch["labels"].append(item["labels"] + [-100] * padding)
    tensors = {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
    if add_zero_mm_token_type_ids:
        zeros = torch.zeros_like(tensors["input_ids"])
        tensors["token_type_ids"] = zeros
        tensors["mm_token_type_ids"] = zeros
    return tensors


def score_candidate_labels(model, tokenizer, row: dict[str, Any], *, condition: str, max_prompt_tokens: int, add_zero_mm_token_type_ids: bool = False):
    import torch

    device = next(model.parameters()).device
    encoded = encode_prompt_fields(tokenizer, row, condition=condition, max_prompt_tokens=max_prompt_tokens)
    prompt_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
    scores: dict[str, float] = {}
    with torch.inference_mode():
        for label in LABELS:
            label_ids = tokenizer(" " + label, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            input_ids = torch.cat([prompt_ids, label_ids], dim=1)
            attention_mask = torch.ones_like(input_ids)
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if add_zero_mm_token_type_ids:
                zeros = torch.zeros_like(input_ids)
                kwargs["token_type_ids"] = zeros
                kwargs["mm_token_type_ids"] = zeros
            outputs = model(**kwargs)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            start = prompt_ids.shape[1] - 1
            end = start + label_ids.shape[1]
            label_logits = logits[:, start:end, :]
            label_targets = targets[:, start:end]
            loss = torch.nn.functional.cross_entropy(
                label_logits.reshape(-1, label_logits.shape[-1]),
                label_targets.reshape(-1),
                reduction="mean",
            )
            scores[label] = float(loss.detach().cpu())
    return min(scores, key=scores.get), scores, encoded


def evaluate_rows(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    condition: str = "full",
    max_prompt_tokens: int = 1024,
    add_zero_mm_token_type_ids: bool = False,
) -> pd.DataFrame:
    import torch

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    predictions = []
    for row_index, row in enumerate(rows):
        label, scores, encoded = score_candidate_labels(
            model,
            tokenizer,
            row,
            condition=condition,
            max_prompt_tokens=max_prompt_tokens,
            add_zero_mm_token_type_ids=add_zero_mm_token_type_ids,
        )
        predictions.append(
            {
                "row_index": row_index,
                "id": row["id"],
                "label": row["label"],
                "predict_label": label,
                "condition": condition,
                "context": row["context"],
                "prompt": row["prompt"],
                "response": row["response"],
                "candidate_nll": json.dumps(scores, ensure_ascii=False, sort_keys=True),
                "context_tokens": encoded.field_token_lengths["context"],
                "context_tokens_used": encoded.field_tokens_used["context"],
                "prompt_tokens": encoded.field_token_lengths["prompt"],
                "prompt_tokens_used": encoded.field_tokens_used["prompt"],
                "response_tokens": encoded.field_token_lengths["response"],
                "response_tokens_used": encoded.field_tokens_used["response"],
            }
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(predictions)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    save_metrics(frame, output_dir)
    runtime = {
        "condition": condition,
        "rows": len(frame),
        "total_seconds": elapsed,
        "seconds_per_sample": elapsed / len(frame) if len(frame) else None,
        "samples_per_second": len(frame) / elapsed if elapsed > 0 else None,
        "max_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
    }
    (output_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    return frame
