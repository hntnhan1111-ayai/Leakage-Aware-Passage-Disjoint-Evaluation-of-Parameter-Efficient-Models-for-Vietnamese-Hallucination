#!/usr/bin/env python3
from __future__ import annotations

import argparse
from inspect import signature
import json
import shutil
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vihallu_repro import LABELS
from vihallu_repro.data import sha256_file
from vihallu_repro.metrics import save_metrics
from vihallu_repro.environment import environment_metadata

LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def load_config(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def model_reference(entry: dict) -> str:
    local = ROOT / entry["local_dir"]
    return str(local) if local.exists() and any(local.iterdir()) else entry["hf_id"]


def segment_frame(frame: pd.DataFrame, split_name: str, model_key: str, vncorenlp_dir: Path, segmenter=None) -> pd.DataFrame:
    if model_key != "phobert":
        return frame
    cache = ROOT / "data/cache/phobert_segmented" / f"{split_name}.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        cached = pd.read_csv(cache)
        if len(cached) == len(frame) and cached["id"].astype(str).tolist() == frame["id"].astype(str).tolist():
            return cached
    if not vncorenlp_dir.exists():
        raise FileNotFoundError(
            f"Missing VnCoreNLP at {vncorenlp_dir}. Run: python scripts/02_setup_vncorenlp.py --output-dir {vncorenlp_dir}"
        )
    if segmenter is None:
        raise RuntimeError(
            "PhoBERT segmentation requires one shared VnCoreNLP instance."
        )
    out = frame.copy()
    for column in ["context", "prompt", "response"]:
        out[column] = [" ".join(segmenter.word_segment(str(text))) for text in out[column]]
    temporary_cache = cache.with_suffix(".csv.tmp")
    out.to_csv(temporary_cache, index=False)
    temporary_cache.replace(cache)
    return out


def build_training_args(output_dir: Path, defaults: dict, seed: int):
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(defaults["epochs"]),
        "per_device_train_batch_size": int(defaults["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(defaults["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(defaults.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(defaults["learning_rate"]),
        "weight_decay": float(defaults["weight_decay"]),
        "warmup_ratio": float(defaults["warmup_ratio"]),
        "logging_steps": 20,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": int(defaults.get("save_total_limit", 1)),
        "report_to": "none",
        "seed": int(seed),
        "data_seed": int(seed),
    }
    params = signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in params})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=["phobert", "xlmr"], required=True)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vncorenlp-dir", default="models/vncorenlp")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    entry = config["models"][args.model_key]
    defaults = dict(config["encoder_defaults"])
    defaults["max_length"] = int(entry.get("max_length", defaults["max_length"]))
    output = ROOT / config["project"]["output_root"] / args.model_key / f"seed_{args.seed}"
    complete_marker = output / "COMPLETED.json"
    if complete_marker.exists() and not args.force:
        print(f"SKIP completed: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)

    # SINGLE_VNCORENLP_INSTANCE
    # PyJNIus permits JVM options/classpath configuration only before the
    # JVM starts. Create one VnCoreNLP object and reuse it across all splits.
    segmenter = None
    if args.model_key == "phobert":
        vncorenlp_dir = ROOT / args.vncorenlp_dir

        if not vncorenlp_dir.exists():
            raise FileNotFoundError(
                f"Missing VnCoreNLP at {vncorenlp_dir}. "
                f"Run: python scripts/02_setup_vncorenlp.py "
                f"--output-dir {vncorenlp_dir}"
            )

        import py_vncorenlp

        segmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=str(vncorenlp_dir),
        )

    frames = {}
    for split, key in [("train", "train_csv"), ("dev", "dev_csv"), ("test", "test_csv")]:
        frame = pd.read_csv(ROOT / config["project"][key])
        if args.debug_limit:
            frame = frame.head(args.debug_limit if split != "train" else max(64, args.debug_limit * 4)).copy()
        frames[split] = segment_frame(frame, split, args.model_key, ROOT / args.vncorenlp_dir, segmenter)

    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer
    import torch

    model_ref = model_reference(entry)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, revision=entry.get("revision") or None, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ref,
        revision=entry.get("revision") or None,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        trust_remote_code=True,
    )

    def tokenize_batch(batch):
        second = [f"USER_PROMPT: {p}\nMODEL_RESPONSE: {r}" for p, r in zip(batch["prompt"], batch["response"])]
        encoded = tokenizer(
            batch["context"],
            second,
            truncation="only_first",
            max_length=int(defaults["max_length"]),
            padding=False,
        )
        encoded["labels"] = [LABEL2ID[label] for label in batch["label"]]
        return encoded

    datasets = {}
    for split, frame in frames.items():
        dataset = Dataset.from_pandas(frame[["id", "context", "prompt", "response", "label"]], preserve_index=False)
        datasets[split] = dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro", labels=list(range(len(LABELS))), zero_division=0)),
            "weighted_f1": float(f1_score(labels, predictions, average="weighted", labels=list(range(len(LABELS))), zero_division=0)),
        }

    trainer_kwargs = {
        "model": model,
        "args": build_training_args(output / "trainer", defaults, args.seed),
        "train_dataset": datasets["train"],
        "eval_dataset": datasets["dev"],
        "compute_metrics": compute_metrics,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
    }
    params = signature(Trainer.__init__).parameters
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    started = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - started
    best_dir = output / "best_model"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    prediction_records = {}
    for split in ["dev", "test"]:
        prediction = trainer.predict(datasets[split])
        pred_ids = np.argmax(prediction.predictions, axis=-1)
        pred_frame = frames[split][["id", "context", "prompt", "response", "label"]].copy()
        pred_frame["predict_label"] = [ID2LABEL[int(value)] for value in pred_ids]
        pred_dir = output / split
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_frame.to_csv(pred_dir / "predictions.csv", index=False)
        metrics = save_metrics(pred_frame, pred_dir)
        prediction_records[split] = {key: value for key, value in metrics.items() if key not in {"classification_report", "confusion_matrix"}}

    trainable = sum(parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in trainer.model.parameters())
    resolved_commit = getattr(trainer.model.config, "_commit_hash", None)
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
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "checkpoint_selection": "maximum development Macro-F1; test evaluated after selection",
        "train_seconds": train_seconds,
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "max_length": int(defaults["max_length"]),
        "truncation": "paired encoding with truncation=only_first to preserve prompt+response",
        "phobert_word_segmentation": args.model_key == "phobert",
        "data_sha256": {split: sha256_file(ROOT / config["project"][f"{split}_csv"]) for split in ["train", "dev", "test"]},
        "metrics": prediction_records,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "environment": environment_metadata(),
    }
    complete_marker.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # CLEAN_TRAINER_CHECKPOINTS
    shutil.rmtree(output / "trainer", ignore_errors=True)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
