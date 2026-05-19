import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_environment


LABELS = ["no", "intrinsic", "extrinsic"]
ADAPTER_WEIGHT_FILES = ["adapter_model.safetensors", "adapter_model.bin"]


def git_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_json(path, data):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {out}")
    return out


def write_log_jsonl(path, rows):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows or []:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if not out.exists():
        raise RuntimeError(f"Missing output: {out}")
    return out


def validate_adapter_dir(adapter_dir):
    p = Path(adapter_dir)
    config = p / "adapter_config.json"
    weights = [p / name for name in ADAPTER_WEIGHT_FILES]
    if not config.exists():
        raise FileNotFoundError(f"Missing adapter config: {config}")
    if not any(item.exists() and item.stat().st_size > 0 for item in weights):
        raise FileNotFoundError(f"Missing adapter weights in {p}. Expected one of: {weights}")
    return p


def resolve_training_config(args, manifest):
    training = dict(manifest["training"])
    model_key = args.model_key or manifest.get("model_key") or manifest["runtime"]["inference_model_key"]
    model_entry = manifest["models"][model_key]
    config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "manifest": args.manifest,
        "seed": int(args.seed if args.seed is not None else manifest["seed"]),
        "labels": manifest["labels"],
        "model_key": model_key,
        "model_id": model_entry["hf_id"],
        "model_dir": args.model_dir or model_entry["local_dir"],
        "train_csv": args.train_csv or manifest["datasets"]["train"]["canonical"],
        "output_dir": args.output_dir or training["output_dir"],
        "adapter_dir": args.adapter_dir or training["adapter_dir"],
        "prompt_template_id": 1,
        "max_length": int(training["max_length"]),
        "max_seq_length": int(training["max_seq_length"]),
        "quantization": manifest["quantization"],
        "lora": {
            "r": int(manifest["lora"]["r"]),
            "lora_alpha": int(manifest["lora"]["lora_alpha"]),
            "lora_dropout": float(manifest["lora"]["lora_dropout"]),
            "target_modules": list(manifest["lora"]["target_modules"]),
            "bias": manifest["lora"]["bias"],
            "task_type": manifest["lora"]["task_type"],
        },
        "training": training,
        "token_key_present": verify_environment.find_token_source() is not None,
    }
    return config


def validate_training_dataset(path):
    from src.data.vihallu import infer_split_from_path, read_csv_robust, validate_gold_df

    split = infer_split_from_path(path)
    if split == "private_test":
        raise RuntimeError("vihallu-private-test.csv must never be used for supervised training or gold evaluation.")
    df = read_csv_robust(path)
    validate_gold_df(df, path, allow_duplicate_ids=False)
    return df


def compute_jaccard_similarity(text1, text2, tokenizer):
    tokens1 = set(tokenizer.tokenize(str(text1)))
    tokens2 = set(tokenizer.tokenize(str(text2)))
    union = tokens1.union(tokens2)
    if not union:
        return 0.0
    return len(tokens1.intersection(tokens2)) / len(union)


def create_training_text(row, tokenizer):
    context = row.get("context", "")
    response = row.get("response", "")
    score = compute_jaccard_similarity(context, response, tokenizer)
    return (
        "Phan tich hallucination bang cach so sanh truc tiep CONTEXT va RESPONSE.\n\n"
        f"Jaccard score: {score:.4f} (1.0 = giong het)\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"RESPONSE:\n{response}\n\n"
        "PHAN TICH:\n"
        "1. RESPONSE co thong tin nao duoc them vao khong co trong CONTEXT khong?\n"
        "2. RESPONSE co mau thuan voi CONTEXT khong?\n\n"
        f"KET LUAN (no/intrinsic/extrinsic): {row['label']}"
    )


def make_bnb_config(quantization, torch, BitsAndBytesConfig):
    dtype_name = quantization.get("bnb_4bit_compute_dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=bool(quantization.get("load_in_4bit", True)),
        bnb_4bit_quant_type=quantization.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=bool(quantization.get("bnb_4bit_use_double_quant", True)),
    )


def build_sft_args(SFTConfig, resolved):
    training = resolved["training"]
    base = {
        "output_dir": resolved["output_dir"],
        "learning_rate": float(training["learning_rate"]),
        "num_train_epochs": float(training["num_train_epochs"]),
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "warmup_steps": int(training["warmup_steps"]),
        "weight_decay": float(training["weight_decay"]),
        "max_grad_norm": float(training["max_grad_norm"]),
        "lr_scheduler_type": training["lr_scheduler_type"],
        "optim": training["optim"],
        "label_smoothing_factor": float(training["label_smoothing_factor"]),
        "bf16": bool(training["bf16"]),
        "logging_steps": int(training["logging_steps"]),
        "report_to": training["report_to"],
        "save_strategy": training["save_strategy"],
        "gradient_checkpointing": bool(training["gradient_checkpointing"]),
        "packing": bool(training["packing"]),
        "dataset_text_field": "text",
    }
    variants = [
        dict(base, max_length=int(training["max_length"])),
        dict(base, max_seq_length=int(training["max_seq_length"])),
        base,
    ]
    last_error = None
    for item in variants:
        try:
            return SFTConfig(**item)
        except TypeError as exc:
            last_error = exc
    raise last_error


def make_trainer(model, tokenizer, train_dataset, peft_config, resolved):
    from trl import SFTTrainer

    try:
        from trl import SFTConfig

        args = build_sft_args(SFTConfig, resolved)
        attempts = [
            {"model": model, "args": args, "train_dataset": train_dataset, "peft_config": peft_config, "processing_class": tokenizer},
            {"model": model, "args": args, "train_dataset": train_dataset, "peft_config": peft_config, "tokenizer": tokenizer},
            {"model": model, "args": args, "train_dataset": train_dataset, "peft_config": peft_config},
        ]
        last_error = None
        for kwargs in attempts:
            try:
                return SFTTrainer(**kwargs), "trl_sfttrainer"
            except TypeError as exc:
                last_error = exc
        raise last_error
    except Exception as sft_exc:
        from datasets import Dataset
        from peft import get_peft_model
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        tokenized = train_dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=resolved["max_seq_length"]),
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        model = get_peft_model(model, peft_config)
        training = resolved["training"]
        args = TrainingArguments(
            output_dir=resolved["output_dir"],
            learning_rate=float(training["learning_rate"]),
            num_train_epochs=float(training["num_train_epochs"]),
            per_device_train_batch_size=int(training["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            warmup_steps=int(training["warmup_steps"]),
            weight_decay=float(training["weight_decay"]),
            max_grad_norm=float(training["max_grad_norm"]),
            lr_scheduler_type=training["lr_scheduler_type"],
            optim=training["optim"],
            label_smoothing_factor=float(training["label_smoothing_factor"]),
            bf16=bool(training["bf16"]),
            logging_steps=int(training["logging_steps"]),
            report_to=training["report_to"],
            save_strategy=training["save_strategy"],
            gradient_checkpointing=bool(training["gradient_checkpointing"]),
            remove_unused_columns=False,
        )
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
        trainer._hallu_fallback_reason = str(sft_exc)
        return trainer, "transformers_trainer_fallback"


def dry_run(args):
    manifest = verify_environment.load_manifest(args.manifest)
    resolved = resolve_training_config(args, manifest)
    train_path = Path(resolved["train_csv"])
    if not train_path.exists():
        train_path = Path(verify_environment.resolve_dataset(manifest, "train")["train"])
        resolved["train_csv"] = str(train_path)
    df = validate_training_dataset(train_path)
    Path(resolved["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(resolved["adapter_dir"]).parent.mkdir(parents=True, exist_ok=True)
    if not resolved["token_key_present"]:
        raise RuntimeError("HF_TOKEN is required in .env or the process environment for target training.")
    print(f"DRY_RUN_OK train_rows={len(df)} adapter_dir={resolved['adapter_dir']} output_dir={resolved['output_dir']} model_key={resolved['model_key']}")


def train(args):
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    try:
        from peft import prepare_model_for_kbit_training
    except Exception:
        prepare_model_for_kbit_training = None
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from src.utils.env import load_hf_token
    from src.utils.seed import set_seed

    manifest = verify_environment.load_manifest(args.manifest)
    resolved = resolve_training_config(args, manifest)
    train_path = Path(resolved["train_csv"])
    if not train_path.exists():
        train_path = Path(verify_environment.resolve_dataset(manifest, "train")["train"])
        resolved["train_csv"] = str(train_path)
    df = validate_training_dataset(train_path)
    token = load_hf_token(required=True)
    set_seed(resolved["seed"])
    Path(resolved["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(resolved["adapter_dir"]).mkdir(parents=True, exist_ok=True)
    write_json(Path(resolved["output_dir"]) / "training_config_resolved.json", resolved)
    tokenizer = AutoTokenizer.from_pretrained(resolved["model_dir"], trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_df = df.copy()
    train_df["text"] = train_df.apply(lambda row: create_training_text(row, tokenizer), axis=1)
    train_dataset = Dataset.from_pandas(train_df[["text"]].reset_index(drop=True))
    bnb_config = make_bnb_config(resolved["quantization"], torch, BitsAndBytesConfig)
    model = AutoModelForCausalLM.from_pretrained(
        resolved["model_dir"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    if bool(resolved["training"]["gradient_checkpointing"]) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    if prepare_model_for_kbit_training is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(resolved["training"]["gradient_checkpointing"]))
    peft_config = LoraConfig(**resolved["lora"])
    trainer, trainer_backend = make_trainer(model, tokenizer, train_dataset, peft_config, resolved)
    result = trainer.train()
    trainer.model.save_pretrained(resolved["adapter_dir"])
    tokenizer.save_pretrained(resolved["adapter_dir"])
    adapter_dir = validate_adapter_dir(resolved["adapter_dir"])
    log_history = getattr(getattr(trainer, "state", None), "log_history", [])
    write_log_jsonl(Path(resolved["output_dir"]) / "train_log.jsonl", log_history)
    metrics = getattr(result, "metrics", {}) if result is not None else {}
    runtime = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trainer_backend": trainer_backend,
        "fallback_reason": getattr(trainer, "_hallu_fallback_reason", None),
        "metrics": metrics,
        "train_rows": int(len(train_df)),
        "adapter_dir": str(adapter_dir),
        "adapter_config": str(adapter_dir / "adapter_config.json"),
        "adapter_weight_files": [str(adapter_dir / name) for name in ADAPTER_WEIGHT_FILES if (adapter_dir / name).exists()],
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
    }
    write_json(Path(resolved["output_dir"]) / "train_runtime.json", runtime)
    print(f"TRAIN_OK adapter_dir={adapter_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/experiment_manifest.yaml")
    parser.add_argument("--model_key", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args)
        return
    train(args)


if __name__ == "__main__":
    main()
