import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


LABELS = ["no", "intrinsic", "extrinsic"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def load_yaml(path):
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a YAML mapping")
    return data


def git_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def flag_enabled(name):
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def validate_config(config):
    if config.get("labels") != LABELS:
        raise ValueError(f"Baseline labels must be {LABELS}")
    if "baselines" not in config or not isinstance(config["baselines"], dict):
        raise ValueError("Baseline config missing baselines mapping")
    for name, item in config["baselines"].items():
        for key in ["enabled", "type", "local_dir", "output_dir"]:
            if key not in item:
                raise ValueError(f"Baseline {name} missing {key}")
    return True


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {p}")
    return p


def write_failure(out_dir, name, exc):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "failure_report.md"
    path.write_text(
        "\n".join([
            f"# Baseline Failure: {name}",
            "",
            f"* timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"* error_type: {type(exc).__name__}",
            f"* error: {exc}",
            "",
            "```text",
            traceback.format_exc(),
            "```",
            "",
        ]),
        encoding="utf-8",
    )
    return path


def build_text(row):
    return f"{row.get('context', '')}\n\n{row.get('prompt', '')}\n\n{row.get('response', '')}"


def run_encoder_baseline(name, item, config, args):
    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
    from src.data.vihallu import read_csv_robust, validate_gold_df
    from src.evaluation.latency import Timer, save_latency_summary
    from src.utils.seed import set_seed

    set_seed(args.seed)
    out = Path(args.out_root) / name if args.out_root else Path(item["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    train = read_csv_robust(args.train_csv or config["train_csv"])
    test = read_csv_robust(args.gold_csv or config["test_csv"])
    validate_gold_df(train, args.train_csv or config["train_csv"], allow_duplicate_ids=False)
    validate_gold_df(test, args.gold_csv or config["test_csv"], allow_duplicate_ids=True)
    model_dir = Path(item["local_dir"])
    if not model_dir.exists():
        raise FileNotFoundError(f"Baseline model directory not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    train_df = train.copy()
    train_df["text"] = train_df.apply(build_text, axis=1)
    train_df["labels"] = train_df["label"].map(LABEL2ID)
    dataset = Dataset.from_pandas(train_df[["text", "labels"]].reset_index(drop=True))
    max_length = int(config["encoder_defaults"]["max_length"])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    training = config["encoder_defaults"]
    train_args = TrainingArguments(
        output_dir=str(out / "trainer"),
        learning_rate=float(training["learning_rate"]),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        warmup_ratio=float(training["warmup_ratio"]),
        weight_decay=float(training["weight_decay"]),
        logging_steps=int(training["logging_steps"]),
        save_strategy=training["save_strategy"],
        report_to=training["report_to"],
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    rows = []
    with Timer(name, samples=len(test)) as timer:
        for row_index, row in enumerate(test.to_dict("records")):
            encoded = tokenizer(build_text(row), return_tensors="pt", truncation=True, max_length=max_length)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = model(**encoded)
            logits = outputs.logits[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(logits))
            rows.append({
                "row_index": row_index,
                "id": row["id"],
                "label": row["label"],
                "predict_label": ID2LABEL[pred_idx],
                "context": row["context"],
                "prompt": row["prompt"],
                "response": row["response"],
                "raw_output": json.dumps(logits.tolist()),
            })
    pred_path = out / "predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    save_latency_summary([timer.summary()], out)
    build_cmd = [
        sys.executable,
        "scripts/build_paper_evidence.py",
        "--gold_csv",
        args.gold_csv or config["test_csv"],
        "--pred_csv",
        str(pred_path),
        "--out_dir",
        str(out),
        "--label_col",
        "label",
        "--pred_col",
        "predict_label",
        "--seed",
        str(args.seed),
        "--latency_seconds",
        str(timer.total_seconds or 0.0),
    ]
    if args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE"):
        build_cmd.append("--allow_known_public_split_leakage")
    subprocess.run(build_cmd, check=True)
    write_json(out / "run_metadata.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "baseline": name,
        "type": item["type"],
        "model_id": item.get("hf_id"),
        "model_dir": item["local_dir"],
        "train_csv": args.train_csv or config["train_csv"],
        "gold_csv": args.gold_csv or config["test_csv"],
        "seed": args.seed,
        "leakage_override": bool(args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE")),
        "challenge_style_evaluation": bool(args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE")),
    })
    return str(out)


def run_qwen_prompt_baseline(name, item, config, args):
    out = Path(args.out_root) / name if args.out_root else Path(item["output_dir"])
    env = dict(os.environ)
    env["RUN_QWEN3_BASELINE"] = "1"
    cmd = [
        sys.executable,
        "scripts/run_optional_qwen3_prompt_baseline.py",
        "--gold_csv",
        args.gold_csv or config["test_csv"],
        "--out_dir",
        str(out),
        "--seed",
        str(args.seed),
    ]
    subprocess.run(cmd, check=True, env=env)
    write_json(out / "run_metadata.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "baseline": name,
        "type": item["type"],
        "model_id": item.get("hf_id"),
        "model_dir": item["local_dir"],
        "gold_csv": args.gold_csv or config["test_csv"],
        "seed": args.seed,
        "leakage_override": bool(args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE")),
        "challenge_style_evaluation": bool(args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE")),
    })
    return str(out)


def run_one(name, item, config, args):
    if item["type"] == "encoder_classifier":
        return run_encoder_baseline(name, item, config, args)
    if item["type"] == "prompt_only_causal_lm":
        return run_qwen_prompt_baseline(name, item, config, args)
    raise RuntimeError(f"Baseline {name} type {item['type']} is disabled or unsupported for this pass.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_models.yaml")
    parser.add_argument("--gold_csv", default=None)
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--out_root", default="results/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow_known_public_split_leakage", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate_config(config)
    enabled = [name for name, item in config["baselines"].items() if item.get("enabled")]
    if args.dry_run:
        print(f"DRY_RUN_OK enabled_baselines={enabled} out_root={args.out_root}")
        return
    strict = bool(args.strict or flag_enabled("STRICT_BASELINES"))
    completed = []
    failed = []
    for name in enabled:
        item = config["baselines"][name]
        try:
            completed.append({"name": name, "out_dir": run_one(name, item, config, args)})
        except Exception as exc:
            failure_path = write_failure(Path(args.out_root) / name, name, exc)
            failed.append({"name": name, "failure_report": str(failure_path), "error": str(exc)})
            if strict:
                raise
    write_json(Path(args.out_root) / "baseline_run_summary.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "completed": completed,
        "failed": failed,
        "strict": strict,
    })
    print(f"BASELINES_DONE completed={len(completed)} failed={len(failed)} strict={strict}")


if __name__ == "__main__":
    main()
