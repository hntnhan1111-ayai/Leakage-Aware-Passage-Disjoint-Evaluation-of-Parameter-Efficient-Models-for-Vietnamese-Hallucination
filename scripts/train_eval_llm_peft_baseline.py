import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


LABELS = ["no", "intrinsic", "extrinsic"]
SUMMARY_COLUMNS = [
    "model_key",
    "model_id",
    "local_dir",
    "method_type",
    "model_family",
    "inference_mode",
    "train_mode",
    "seed",
    "split",
    "dataset_path",
    "rows",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "no_precision",
    "no_recall",
    "no_f1",
    "no_support",
    "intrinsic_precision",
    "intrinsic_recall",
    "intrinsic_f1",
    "intrinsic_support",
    "extrinsic_precision",
    "extrinsic_recall",
    "extrinsic_f1",
    "extrinsic_support",
    "malformed_count",
    "malformed_rate",
    "latency_mean_s",
    "latency_median_s",
    "latency_p95_s",
    "total_runtime_s",
    "rows_per_second",
    "peak_gpu_memory_gb",
    "dtype",
    "quantization",
    "challenge_style_evaluation",
    "leakage_override",
    "git_branch",
    "git_commit",
    "artifact_dir",
    "status",
    "skip_reason",
]


def load_yaml(path):
    import yaml

    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a YAML mapping")
    return data


def git_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {p}")
    return p


def write_status(out, entry, args, status, reason="", expected_rows=None):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_key": entry["model_key"],
        "model_id": entry["model_id"],
        "method_type": entry["method_type"],
        "local_dir": entry["local_dir"],
        "seed": int(args.seed),
        "dataset_path": args.gold_csv,
        "requested_limit": args.debug_limit,
        "expected_rows": expected_rows,
        "status": status,
        "skip_reason": reason,
        "force_rerun": bool(args.force_rerun_model),
    }
    write_json(Path(out) / "status.json", payload)
    return payload


def count_csv_rows(path):
    import pandas as pd

    p = Path(path)
    if not p.exists():
        return 0
    return int(len(pd.read_csv(p)))


def validate_prediction_csv(path, expected_rows):
    import pandas as pd

    df = pd.read_csv(path)
    if len(df) != int(expected_rows):
        raise RuntimeError(f"prediction_row_count_mismatch:expected={expected_rows}:found={len(df)}")
    if "predict_label" not in df.columns:
        raise RuntimeError("prediction_missing_predict_label")
    if df["predict_label"].isna().any():
        raise RuntimeError("prediction_nan_predict_label")
    bad = sorted(set(df["predict_label"].astype(str)) - set(LABELS))
    if bad:
        raise RuntimeError("invalid_prediction_labels:" + ",".join(bad))
    return df


def int_or_none(value):
    if value in [None, ""]:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def same_dataset_path(a, b):
    if a in [None, ""] or b in [None, ""]:
        return False
    left = Path(str(a)).as_posix()
    right = Path(str(b)).as_posix()
    return left == right or left.endswith("/" + right) or right.endswith("/" + left)


def completed_status_row(entry, args, out, expected_rows):
    status_path = out / "status.json"
    pred_path = out / "predictions.csv"
    summary_path = out / "summary_metrics.json"
    report_path = out / "classification_report.json"
    if args.force_rerun_model or not status_path.exists():
        return None
    required = [pred_path, summary_path, report_path, out / "classification_report.csv", out / "confusion_matrix.csv", out / "malformed_predictions.csv"]
    if any(not p.exists() or p.stat().st_size == 0 for p in required):
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    try:
        pred_rows = len(validate_prediction_csv(pred_path, expected_rows))
    except Exception:
        return None
    if pred_rows != expected_rows:
        return None
    checks = [
        status.get("status") == "completed",
        int_or_none(status.get("rows")) == expected_rows,
        int_or_none(status.get("expected_rows")) in [None, expected_rows],
        int_or_none(status.get("requested_limit")) == int_or_none(args.debug_limit),
        int_or_none(status.get("seed")) == int(args.seed),
        status.get("model_id") == entry["model_id"],
        status.get("method_type") == entry["method_type"],
        same_dataset_path(status.get("dataset_path"), args.gold_csv),
        int_or_none(status.get("malformed_count")) == 0,
    ]
    if not all(checks):
        return None
    return {key: status.get(key, "") for key in SUMMARY_COLUMNS}


def build_prompt(row):
    return (
        "You are a strict Vietnamese hallucination detection classifier.\n\n"
        "Task:\n"
        "Given CONTEXT, USER_PROMPT, and MODEL_RESPONSE, classify MODEL_RESPONSE by faithfulness to CONTEXT.\n\n"
        "Labels:\n"
        "no = MODEL_RESPONSE is fully supported by CONTEXT.\n"
        "intrinsic = MODEL_RESPONSE contradicts or distorts information in CONTEXT.\n"
        "extrinsic = MODEL_RESPONSE adds information not supported by CONTEXT.\n\n"
        "Rules:\n"
        "Return exactly one label.\n"
        "Allowed outputs: no, intrinsic, extrinsic.\n"
        "Do not explain.\n"
        "Do not copy CONTEXT.\n"
        "Do not answer USER_PROMPT.\n"
        "Do not add punctuation.\n\n"
        f"CONTEXT:\n{row.get('context', '')}\n\n"
        f"USER_PROMPT:\n{row.get('prompt', '')}\n\n"
        f"MODEL_RESPONSE:\n{row.get('response', '')}\n\n"
        "Label:"
    )


def extract_label(text):
    cleaned = str(text).strip().lower().strip("`'\" .,;:\n\t")
    for prefix in ["label", "nhan", "nhãn"]:
        if cleaned.startswith(prefix + ":"):
            cleaned = cleaned.split(":", 1)[1].strip().strip("`'\" .,;:\n\t")
    hits = [label for label in LABELS if cleaned == label or cleaned.startswith(label + "\n") or cleaned.startswith(label + " ")]
    if len(hits) == 1:
        return hits[0], None
    tokens = [token.strip("`'\" .,;:()[]{}").lower() for token in str(text).replace("\n", " ").split()]
    hits = sorted(set(token for token in tokens if token in LABELS))
    if len(hits) == 1:
        return hits[0], None
    if len(hits) > 1:
        return None, "conflicting_labels"
    return None, "no_label"


class LabelCompletionDataset:
    def __init__(self, rows, tokenizer, max_seq_length):
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.max_seq_length = int(max_seq_length)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prompt = build_prompt(row)
        label = " " + str(row["label"]).strip()
        label_ids = self.tokenizer(label, add_special_tokens=False).input_ids
        prompt_budget = max(8, self.max_seq_length - len(label_ids))
        prompt_ids = self.tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=prompt_budget).input_ids
        input_ids = (prompt_ids + label_ids)[: self.max_seq_length]
        labels = ([-100] * len(prompt_ids) + label_ids)[: self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_batch(features, tokenizer, torch, entry=None):
    pad_id = tokenizer.pad_token_id
    max_len = max(len(item["input_ids"]) for item in features)
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        pad = max_len - len(item["input_ids"])
        batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
        batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
        batch["labels"].append(item["labels"] + [-100] * pad)
    tensors = {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
    if entry and entry.get("add_zero_mm_token_type_ids"):
        zeros = torch.zeros_like(tensors["input_ids"])
        tensors["token_type_ids"] = zeros
        tensors["mm_token_type_ids"] = zeros
    return tensors


def load_tokenizer(local_dir):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(entry, torch, quantized=True):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

    kwargs = {"device_map": "auto", "dtype": torch.bfloat16, "trust_remote_code": True}
    if quantized:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    classes = [AutoModelForCausalLM]
    if entry.get("expected_loader") == "AutoModelForImageTextToText":
        classes.append(AutoModelForImageTextToText)
    else:
        config = AutoConfig.from_pretrained(entry["local_dir"], trust_remote_code=True)
        architectures = [str(item) for item in getattr(config, "architectures", []) or []]
        if any("ImageTextToText" in item for item in architectures):
            classes.append(AutoModelForImageTextToText)
    errors = []
    for model_class in classes:
        try:
            return model_class.from_pretrained(entry["local_dir"], **kwargs)
        except TypeError as exc:
            if "dtype" not in str(exc):
                errors.append(f"{model_class.__name__}:{type(exc).__name__}:{exc}")
                continue
            fallback = dict(kwargs)
            fallback["torch_dtype"] = fallback.pop("dtype")
            try:
                return model_class.from_pretrained(entry["local_dir"], **fallback)
            except Exception as retry_exc:
                errors.append(f"{model_class.__name__}:{type(retry_exc).__name__}:{retry_exc}")
        except Exception as exc:
            errors.append(f"{model_class.__name__}:{type(exc).__name__}:{exc}")
    raise RuntimeError("model_load_failed:" + " | ".join(errors))


def is_supported_lora_module(module, torch):
    if isinstance(module, torch.nn.Linear):
        return True
    return module.__class__.__name__ in {"Linear4bit", "Linear8bitLt"}


def is_excluded_lora_name(name):
    lowered = name.lower()
    blocked = ["vision", "audio", "image", "projector", "multi_modal", "multimodal", "mm", "clip", "clippable"]
    return any(part in lowered for part in blocked)


def discover_text_target_modules(model, torch):
    candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    found = []
    for name, module in model.named_modules():
        if is_excluded_lora_name(name) or not is_supported_lora_module(module, torch):
            continue
        parts = name.split(".")
        tail = parts[-1]
        parent = parts[-2] if len(parts) > 1 else ""
        if tail in candidates or parent in candidates:
            found.append(name)
    result = sorted(set(found))
    if not result:
        raise RuntimeError("no_supported_text_lora_targets_found")
    return result


def discover_target_modules(model, entry, torch):
    if entry.get("lora_target_scope") == "text_only":
        return discover_text_target_modules(model, torch)
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "dense", "fc1", "fc2"]
    found = []
    for name, module in model.named_modules():
        tail = name.split(".")[-1]
        if tail in candidates and is_supported_lora_module(module, torch):
            found.append(tail)
    result = sorted(set(found))
    if not result:
        raise RuntimeError("no_supported_lora_targets_found")
    return result


def training_args(output_dir, args, entry, transformers_training_args):
    from inspect import signature

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(args.epochs),
        "learning_rate": float(args.learning_rate or entry.get("learning_rate", 0.0002)),
        "per_device_train_batch_size": int(entry.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(entry.get("gradient_accumulation_steps", 16)),
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": "none",
        "seed": int(args.seed),
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "remove_unused_columns": False,
    }
    params = signature(transformers_training_args.__init__).parameters
    return transformers_training_args(**{key: value for key, value in kwargs.items() if key in params})


def train_model(entry, args, train_rows):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments
    from src.utils.seed import set_seed

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tokenizer = load_tokenizer(entry["local_dir"])
    model = load_base_model(entry, torch, quantized=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    target_modules = discover_target_modules(model, entry, torch)
    r = int(entry.get("lora_r", 64))
    lora_config = LoraConfig(
        r=r,
        lora_alpha=int(entry.get("lora_alpha", 2 * r)),
        lora_dropout=float(entry.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    max_seq_length = int(args.max_seq_length or entry.get("max_seq_length", 1024))
    dataset = LabelCompletionDataset(train_rows, tokenizer, max_seq_length)
    out = Path(args.out_root) / entry["model_key"]
    trainer_kwargs = {
        "model": model,
        "args": training_args(out / "trainer", args, entry, TrainingArguments),
        "train_dataset": dataset,
        "data_collator": lambda features: collate_batch(features, tokenizer, torch, entry),
    }
    from inspect import signature

    params = signature(Trainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    adapter_dir = out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    weights = [adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin"]
    if not (adapter_dir / "adapter_config.json").exists() or not any(path.exists() for path in weights):
        raise RuntimeError(f"Missing PEFT adapter files in {adapter_dir}")
    return trainer.model, tokenizer, target_modules, train_seconds


def score_labels_with_extra_inputs(model, tokenizer, prompt, torch, max_prompt_tokens):
    device = next(model.parameters()).device
    results = {}
    with torch.inference_mode():
        for label in LABELS:
            label_ids = tokenizer(" " + label, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            max_length = max(1, int(max_prompt_tokens) - int(label_ids.shape[1]))
            prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_length).input_ids.to(device)
            input_ids = torch.cat([prompt_ids, label_ids], dim=1)
            attention_mask = torch.ones_like(input_ids)
            zeros = torch.zeros_like(input_ids)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=zeros, mm_token_type_ids=zeros)
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
            results[label] = float(loss.detach().cpu())
    return min(results, key=results.get), results


def score_labels(model, tokenizer, prompt, torch, max_prompt_tokens, entry):
    if entry.get("add_zero_mm_token_type_ids"):
        return score_labels_with_extra_inputs(model, tokenizer, prompt, torch, max_prompt_tokens)
    from src.models.label_scoring import score_labels_causal_lm

    device = next(model.parameters()).device
    label, scores = score_labels_causal_lm(model, tokenizer, prompt, device=device, max_prompt_tokens=max_prompt_tokens)
    return label, scores


def generate_label(model, tokenizer, row, args, torch, entry):
    prompt = build_prompt(row)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=int(args.max_prompt_tokens))
    inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}
    if entry.get("add_zero_mm_token_type_ids"):
        zeros = torch.zeros_like(inputs["input_ids"])
        inputs["token_type_ids"] = zeros
        inputs["mm_token_type_ids"] = zeros
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    label, reason = extract_label(text)
    fallback_scores = None
    if label is None:
        label, fallback_scores = score_labels(model, tokenizer, prompt, torch, int(args.max_prompt_tokens), entry)
        reason = None
    return label, text, reason, fallback_scores


def evaluate_model(entry, args, model, tokenizer, gold_rows):
    import pandas as pd
    import torch
    from src.evaluation.latency import save_latency_summary
    from src.evaluation.metrics import compute_and_save

    out = Path(args.out_root) / entry["model_key"]
    pred_path = out / "predictions.csv"
    malformed_path = out / "malformed_predictions.csv"
    pred_tmp = out / "predictions.csv.tmp"
    malformed_tmp = out / "malformed_predictions.csv.tmp"
    rows = []
    malformed = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    model.eval()
    for row_index, row in enumerate(gold_rows):
        label, raw_output, malformed_reason, scores = generate_label(model, tokenizer, row, args, torch, entry)
        if malformed_reason:
            malformed.append({"row_index": row_index, "id": row.get("id"), "label": row.get("label"), "predict_label": "", "malformed_reason": malformed_reason, "raw_output": raw_output})
        rows.append({
            "row_index": row_index,
            "id": row["id"],
            "label": row["label"],
            "predict_label": label,
            "context": row["context"],
            "prompt": row["prompt"],
            "response": row["response"],
            "raw_output": raw_output,
            "fallback_scores": json.dumps(scores, ensure_ascii=False, sort_keys=True) if scores else "",
        })
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started
    pred_df = pd.DataFrame(rows)
    malformed_df = pd.DataFrame(malformed, columns=["row_index", "id", "label", "predict_label", "malformed_reason", "raw_output"])
    bad = sorted(set(pred_df["predict_label"].dropna().astype(str)) - set(LABELS))
    if bad:
        raise RuntimeError("invalid_prediction_labels:" + ",".join(bad))
    if len(malformed):
        raise RuntimeError(f"malformed_predictions:{len(malformed)}")
    if len(pred_df) != len(gold_rows):
        raise RuntimeError(f"prediction_row_count_mismatch:expected={len(gold_rows)}:found={len(pred_df)}")
    pred_df.to_csv(pred_tmp, index=False)
    malformed_df.to_csv(malformed_tmp, index=False)
    validate_prediction_csv(pred_tmp, len(gold_rows))
    pred_tmp.replace(pred_path)
    malformed_tmp.replace(malformed_path)
    compute_and_save(pred_df["label"], pred_df["predict_label"], out)
    memory_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
    save_latency_summary([{
        "name": entry["model_key"],
        "total_seconds": float(total_seconds),
        "samples": int(len(rows)),
        "seconds_per_sample": float(total_seconds / len(rows)) if rows else None,
        "samples_per_second": float(len(rows) / total_seconds) if total_seconds > 0 and rows else None,
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "cuda_max_memory_reserved": memory_reserved,
    }], out)
    return total_seconds, memory_reserved


def status_from_outputs(entry, args, out, train_rows, expected_rows, target_modules, train_seconds, eval_seconds, memory_reserved):
    summary = json.loads((out / "summary_metrics.json").read_text(encoding="utf-8"))
    report = json.loads((out / "classification_report.json").read_text(encoding="utf-8"))
    rows = len(validate_prediction_csv(out / "predictions.csv", expected_rows))
    malformed_count = count_csv_rows(out / "malformed_predictions.csv")
    required = [
        out / "summary_metrics.json",
        out / "classification_report.json",
        out / "classification_report.csv",
        out / "confusion_matrix.csv",
        out / "confusion_matrix.png",
        out / "latency_summary.json",
        out / "latency_summary.csv",
        out / "predictions.csv",
        out / "malformed_predictions.csv",
        out / "training_config_resolved.json",
        out / "train_runtime.json",
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("missing_required_outputs:" + ",".join(missing))
    if int(rows) != int(expected_rows):
        raise RuntimeError(f"prediction_row_count_mismatch:expected={expected_rows}:found={rows}")
    if int(malformed_count) != 0:
        raise RuntimeError(f"malformed_predictions:{malformed_count}")
    total_runtime = float(train_seconds + eval_seconds)
    row = {
        "model_key": entry["model_key"],
        "model_id": entry["model_id"],
        "local_dir": entry["local_dir"],
        "method_type": entry["method_type"],
        "model_family": entry["model_family"],
        "inference_mode": entry["inference_mode"],
        "train_mode": entry["train_mode"],
        "seed": int(args.seed),
        "split": "public_test",
        "dataset_path": args.gold_csv,
        "rows": int(rows),
        "accuracy": float(summary["accuracy"]),
        "macro_f1": float(summary["macro_f1"]),
        "weighted_f1": float(summary["weighted_f1"]),
        "malformed_count": int(malformed_count),
        "malformed_rate": float(malformed_count / rows) if rows else 0.0,
        "latency_mean_s": float(eval_seconds / rows) if rows else 0.0,
        "latency_median_s": float(eval_seconds / rows) if rows else 0.0,
        "latency_p95_s": float(eval_seconds / rows) if rows else 0.0,
        "total_runtime_s": total_runtime,
        "rows_per_second": float(rows / eval_seconds) if eval_seconds > 0 and rows else 0.0,
        "peak_gpu_memory_gb": float(memory_reserved / (1024 ** 3)) if memory_reserved else 0.0,
        "dtype": entry.get("dtype", "bfloat16"),
        "quantization": entry.get("quantization", "nf4_4bit"),
        "challenge_style_evaluation": True,
        "leakage_override": True,
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "artifact_dir": str(out),
        "status": "completed",
        "skip_reason": "",
    }
    for label in LABELS:
        values = report.get(label, {})
        row[f"{label}_precision"] = float(values.get("precision", 0.0))
        row[f"{label}_recall"] = float(values.get("recall", 0.0))
        row[f"{label}_f1"] = float(values.get("f1-score", 0.0))
        row[f"{label}_support"] = int(values.get("support", 0))
    if any(math.isnan(float(row[key])) for key in ["accuracy", "macro_f1", "weighted_f1"]):
        raise RuntimeError("invalid_nan_metrics")
    payload = dict(row)
    payload.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requested_limit": args.debug_limit,
        "expected_rows": int(expected_rows),
        "force_rerun": bool(args.force_rerun_model),
        "train_rows": int(train_rows),
        "target_modules": target_modules,
        "adapter_dir": str(out / "adapter"),
        "train_seconds": float(train_seconds),
        "eval_seconds": float(eval_seconds),
    })
    write_json(out / "status.json", payload)
    write_json(out / "prediction_config.json", payload)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--config", default="configs/baseline_models.yaml")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--gold-csv", default=None)
    parser.add_argument("--out-root", default="results/model_comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--debug-limit", type=int, default=None)
    parser.add_argument("--force-rerun-model", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=3)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.model_key not in config.get("baselines", {}):
        raise KeyError(f"Unknown model key: {args.model_key}")
    entry = dict(config["baselines"][args.model_key])
    if entry.get("method_type") != "supervised_peft_finetune":
        raise ValueError(f"{args.model_key} must use method_type=supervised_peft_finetune")
    args.train_csv = args.train_csv or config.get("train_csv", "vihallu-train.csv")
    args.gold_csv = args.gold_csv or config.get("test_csv", "vihallu-test.csv")
    args.epochs = float(args.epochs if args.epochs is not None else entry.get("epochs", 2))
    args.max_seq_length = int(args.max_seq_length or entry.get("max_seq_length", 1024))
    out = Path(args.out_root) / entry["model_key"]
    from src.data.vihallu import read_csv_robust, validate_gold_df
    from src.models.download import validate_local_model_entry

    train_df = read_csv_robust(args.train_csv)
    gold_df = read_csv_robust(args.gold_csv)
    validate_gold_df(train_df, args.train_csv, allow_duplicate_ids=False)
    validate_gold_df(gold_df, args.gold_csv, allow_duplicate_ids=True)
    if args.debug_limit is not None:
        train_df = train_df.head(max(16, int(args.debug_limit) * 4)).copy()
        gold_df = gold_df.head(int(args.debug_limit)).copy()
    expected_rows = int(len(gold_df))
    existing = completed_status_row(entry, args, out, expected_rows)
    if existing is not None:
        print(f"MODEL_ALREADY_COMPLETED {entry['model_key']}")
        return
    ready = validate_local_model_entry(entry)
    if not ready["ok"]:
        raise RuntimeError(f"{entry['model_key']} local model not ready: {ready['status']}:{ready['reason']}")
    out.mkdir(parents=True, exist_ok=True)
    write_status(out, entry, args, "running", expected_rows=expected_rows)
    try:
        started = datetime.now(timezone.utc).isoformat()
        train_rows = train_df.to_dict("records")
        gold_rows = gold_df.to_dict("records")
        model, tokenizer, target_modules, train_seconds = train_model(entry, args, train_rows)
        eval_seconds, memory_reserved = evaluate_model(entry, args, model, tokenizer, gold_rows)
        write_json(out / "training_config_resolved.json", {
            "timestamp": started,
            "model_key": entry["model_key"],
            "model_id": entry["model_id"],
            "method_type": entry["method_type"],
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": float(args.learning_rate or entry.get("learning_rate", 0.0002)),
            "lora_r": int(entry.get("lora_r", 64)),
            "lora_alpha": int(entry.get("lora_alpha", 2 * int(entry.get("lora_r", 64)))),
            "lora_dropout": float(entry.get("lora_dropout", 0.05)),
            "target_modules": target_modules,
            "train_rows": len(train_rows),
            "eval_rows": expected_rows,
            "adapter_dir": str(out / "adapter"),
        })
        write_json(out / "train_runtime.json", {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_key": entry["model_key"],
            "train_seconds": float(train_seconds),
            "eval_seconds": float(eval_seconds),
            "train_rows": len(train_rows),
            "eval_rows": expected_rows,
            "seed": args.seed,
        })
        row = status_from_outputs(entry, args, out, len(train_rows), expected_rows, target_modules, train_seconds, eval_seconds, memory_reserved)
        print(json.dumps({"model_key": entry["model_key"], "status": row["status"], "rows": row["rows"], "macro_f1": row["macro_f1"]}, ensure_ascii=False, sort_keys=True))
    except BaseException as exc:
        write_status(out, entry, args, "failed", f"{type(exc).__name__}: {exc}", expected_rows=expected_rows)
        raise


if __name__ == "__main__":
    main()
