#!/usr/bin/env python3
from __future__ import annotations

import argparse
from inspect import signature
import json
import shutil
from pathlib import Path
import sys
import time

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vihallu_repro.data import sha256_file
from vihallu_repro.environment import environment_metadata
from vihallu_repro.peft_runtime import (
    LabelCompletionDataset,
    collate_label_completion,
    discover_lora_target_modules,
    evaluate_rows,
    load_base_model,
    load_tokenizer_or_processor,
)
from vihallu_repro.prompts import PROMPT_TEMPLATE_VERSION, build_prompt


def model_reference(entry: dict) -> str:
    local = ROOT / entry["local_dir"]
    return str(local) if local.exists() and any(local.iterdir()) else entry["hf_id"]


def build_training_args(output_dir: Path, defaults: dict, seed: int):
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(defaults["epochs"]),
        "per_device_train_batch_size": int(defaults["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(defaults["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(defaults["gradient_accumulation_steps"]),
        "learning_rate": float(defaults["learning_rate"]),
        "weight_decay": float(defaults["weight_decay"]),
        "warmup_ratio": float(defaults["warmup_ratio"]),
        "logging_steps": 10,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": int(defaults.get("save_total_limit", 1)),
        "report_to": "none",
        "seed": int(seed),
        "data_seed": int(seed),
        "bf16": bool(defaults.get("bf16", True)),
        "gradient_checkpointing": True,
        "optim": str(defaults.get("optimizer", "paged_adamw_8bit")),
        "remove_unused_columns": False,
    }
    params = signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    if "gradient_checkpointing_kwargs" in params:
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in params})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=["vistral_peft", "qwen35_peft", "gemma4_peft"], required=True)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    entry = dict(config["models"][args.model_key])
    defaults = dict(config["peft_defaults"])
    defaults.update({key: value for key, value in entry.items() if key in defaults})
    if args.epochs is not None:
        defaults["epochs"] = args.epochs
    output = ROOT / config["project"]["output_root"] / args.model_key / f"seed_{args.seed}"
    complete_marker = output / "COMPLETED.json"
    if complete_marker.exists() and not args.force:
        print(f"SKIP completed: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)

    frames = {}
    for split, key in [("train", "train_csv"), ("dev", "dev_csv"), ("test", "test_csv")]:
        frame = pd.read_csv(ROOT / config["project"][key])
        if args.debug_limit:
            frame = frame.head(args.debug_limit if split != "train" else max(64, args.debug_limit * 4)).copy()
        frames[split] = frame

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_ref = model_reference(entry)
    tokenizer = load_tokenizer_or_processor(model_ref)
    model = load_base_model(model_ref, quantized=True, torch_dtype="bfloat16")
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    target_modules = discover_lora_target_modules(model, text_only=True)
    lora_config = LoraConfig(
        r=int(entry["lora_r"]),
        lora_alpha=int(entry["lora_alpha"]),
        lora_dropout=float(defaults["lora_dropout"]),
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    max_seq_length = int(defaults["max_seq_length"])
    train_dataset = LabelCompletionDataset(frames["train"].to_dict("records"), tokenizer, max_seq_length)
    dev_dataset = LabelCompletionDataset(frames["dev"].to_dict("records"), tokenizer, max_seq_length)
    add_zero_mm = bool(entry.get("add_zero_mm_token_type_ids", False))
    trainer_kwargs = {
        "model": model,
        "args": build_training_args(output / "trainer", defaults, args.seed),
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": lambda features: collate_label_completion(
            features, tokenizer, add_zero_mm_token_type_ids=add_zero_mm
        ),
    }
    params = signature(Trainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    # AUTO_RESUME_LATEST_VALID_CHECKPOINT
    trainer_dir = output / "trainer"
    resume_checkpoint = None

    if trainer_dir.exists() and not args.force:
        candidates = sorted(
            (
                checkpoint
                for checkpoint in trainer_dir.glob("checkpoint-*")
                if checkpoint.is_dir()
            ),
            key=lambda checkpoint: int(checkpoint.name.rsplit("-", 1)[-1]),
            reverse=True,
        )

        for checkpoint in candidates:
            if (checkpoint / "checkpoint-is-incomplete.txt").exists():
                continue

            required_state = (
                checkpoint / "trainer_state.json",
                checkpoint / "optimizer.pt",
                checkpoint / "scheduler.pt",
            )

            has_model = any(
                (checkpoint / filename).is_file()
                for filename in (
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                    "model.safetensors",
                    "pytorch_model.bin",
                )
            )

            if all(item.is_file() for item in required_state) and has_model:
                resume_checkpoint = str(checkpoint)
                break

        if candidates and resume_checkpoint is None:
            raise RuntimeError(
                f"No complete resumable checkpoint found under {trainer_dir}"
            )

    if resume_checkpoint:
        print(f"RESUME_FROM_CHECKPOINT={resume_checkpoint}", flush=True)
    else:
        print("RESUME_FROM_CHECKPOINT=None; starting fresh model run", flush=True)

    started = time.perf_counter()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    train_seconds = time.perf_counter() - started
    adapter_dir = output / entry.get("adapter_subdir", "adapter")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    for split in ["dev", "test"]:
        evaluate_rows(
            trainer.model,
            tokenizer,
            frames[split].to_dict("records"),
            output_dir=output / split,
            condition="full",
            max_prompt_tokens=int(defaults["max_prompt_tokens"]),
            add_zero_mm_token_type_ids=add_zero_mm,
        )

    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in trainer.model.parameters())
    base_config = getattr(getattr(trainer.model, "base_model", trainer.model), "config", None)
    resolved_commit = getattr(base_config, "_commit_hash", None) if base_config is not None else None
    revision_file = (ROOT / entry["local_dir"] / ".resolved_revision")
    if resolved_commit is None and revision_file.exists():
        resolved_commit = revision_file.read_text(encoding="utf-8").strip() or None
    metadata = {
        "model_key": args.model_key,
        "hf_id": entry["hf_id"],
        "model_reference": model_ref,
        "requested_revision": entry.get("revision"),
        "resolved_commit": resolved_commit,
        "seed": args.seed,
        "train_rows": len(frames["train"]),
        "dev_rows": len(frames["dev"]),
        "test_rows": len(frames["test"]),
        "train_seconds": train_seconds,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric_eval_loss": trainer.state.best_metric,
        "checkpoint_selection": "minimum development loss; test evaluated after selection",
        "inference_mode": defaults["inference_mode"],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_template_example": build_prompt(frames["train"].iloc[0].to_dict(), condition="full"),
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quantization": True,
        },
        "lora": {
            "r": int(entry["lora_r"]),
            "alpha": int(entry["lora_alpha"]),
            "dropout": float(defaults["lora_dropout"]),
            "bias": "none",
            "target_modules": target_modules,
        },
        "training": {
            "epochs": float(defaults["epochs"]),
            "learning_rate": float(defaults["learning_rate"]),
            "weight_decay": float(defaults["weight_decay"]),
            "warmup_ratio": float(defaults["warmup_ratio"]),
            "per_device_train_batch_size": int(defaults["per_device_train_batch_size"]),
            "gradient_accumulation_steps": int(defaults["gradient_accumulation_steps"]),
            "effective_batch_size": int(defaults["per_device_train_batch_size"]) * int(defaults["gradient_accumulation_steps"]),
            "optimizer": defaults["optimizer"],
            "max_seq_length": max_seq_length,
            "field_aware_truncation": "context head+tail, prompt head, response head with fixed token budgets",
        },
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "data_sha256": {split: sha256_file(ROOT / config["project"][f"{split}_csv"]) for split in ["train", "dev", "test"]},
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        "environment": environment_metadata(),
    }
    complete_marker.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # CLEAN_TRAINER_CHECKPOINTS
    shutil.rmtree(output / "trainer", ignore_errors=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
