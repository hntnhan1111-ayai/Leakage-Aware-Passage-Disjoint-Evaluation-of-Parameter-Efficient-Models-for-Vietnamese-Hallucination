import argparse
import csv
from datetime import datetime, timezone
import gc
from inspect import signature
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

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
MALFORMED_COLUMNS = ["row_index", "id", "label", "predict_label", "malformed_reason", "raw_output"]


class SkipModel(Exception):
    pass


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


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {p}")
    return p


def write_empty_malformed(path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=MALFORMED_COLUMNS).writeheader()
    return p


def normalize_ref(value):
    return str(value).replace("\\", "/").strip().rstrip("/").lower()


def match_reference(actual, accepted_aliases):
    if not actual:
        return None
    actual_ref = normalize_ref(actual)
    for item in accepted_aliases:
        if actual_ref == normalize_ref(item):
            return item
    actual_name = Path(actual_ref).name.lower()
    for item in accepted_aliases:
        if actual_name == Path(normalize_ref(item)).name.lower():
            return item
    return None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_alias(entry):
    config_path = Path(entry["local_dir"]) / "config.json"
    if not config_path.exists():
        return {"actual_name_or_path": None, "matched_alias": None, "accepted_aliases": entry.get("accepted_aliases", [])}
    data = read_json(config_path)
    actual = data.get("_name_or_path") or data.get("name_or_path")
    if not actual:
        return {"actual_name_or_path": None, "matched_alias": None, "accepted_aliases": entry.get("accepted_aliases", [])}
    matched = match_reference(actual, entry.get("accepted_aliases", []))
    if not matched:
        raise SkipModel(f"invalid_alias: actual_name_or_path={actual} accepted_aliases={entry.get('accepted_aliases', [])}")
    return {"actual_name_or_path": actual, "matched_alias": matched, "accepted_aliases": entry.get("accepted_aliases", [])}


def local_model_ready(entry):
    from src.models.download import validate_local_model_entry

    result = validate_local_model_entry(entry, load_tokenizer=(entry.get("type") == "encoder_classifier"))
    if result["ok"]:
        return True, None
    if result["status"] == "missing_local_model":
        return False, "missing_local_model"
    if result["status"] == "tokenizer_load_failed":
        return False, "tokenizer_load_failed:" + str(result["reason"])
    if result["status"] == "unsupported_config":
        return False, "unsupported_config:" + str(result["reason"])
    return False, "incomplete_local_model:" + str(result["reason"])


def adapter_ready(entry):
    adapter_dir = entry.get("adapter_dir")
    if not adapter_dir:
        return True, None
    root = Path(adapter_dir)
    if not root.exists():
        return False, "missing_adapter_dir"
    if not (root / "adapter_config.json").exists():
        return False, "missing_adapter_config"
    if not (root / "adapter_model.safetensors").exists() and not (root / "adapter_model.bin").exists():
        return False, "missing_adapter_weights"
    return True, None


def validate_config(config):
    if config.get("labels") != LABELS:
        raise ValueError(f"Baseline labels must be {LABELS}")
    if "baselines" not in config or not isinstance(config["baselines"], dict):
        raise ValueError("Baseline config missing baselines mapping")
    required = ["model_key", "model_id", "local_dir", "model_family", "method_type", "enabled", "expected_loader", "accepted_aliases", "required_files", "inference_mode", "train_mode", "notes"]
    for name, item in config["baselines"].items():
        for key in required:
            if key not in item:
                raise ValueError(f"Baseline {name} missing {key}")
        if item["model_key"] != name:
            raise ValueError(f"Baseline key {name} must match model_key={item['model_key']}")
    return True


def build_text(row):
    return f"{row.get('context', '')}\n\n{row.get('prompt', '')}\n\n{row.get('response', '')}"


def load_encoder_tokenizer(auto_tokenizer, entry):
    try:
        return auto_tokenizer.from_pretrained(entry["local_dir"])
    except Exception:
        if entry.get("model_key") == "phobert":
            return auto_tokenizer.from_pretrained(entry["local_dir"], use_fast=False)
        raise


def build_label_prompt(row):
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


def release_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def summary_blank(entry, args, status, skip_reason="", rows=""):
    return {
        "model_key": entry["model_key"],
        "model_id": entry["model_id"],
        "local_dir": entry["local_dir"],
        "method_type": entry["method_type"],
        "model_family": entry["model_family"],
        "inference_mode": entry["inference_mode"],
        "train_mode": entry["train_mode"],
        "seed": args.seed,
        "split": "public_test",
        "dataset_path": args.gold_csv,
        "rows": rows,
        "accuracy": "",
        "macro_f1": "",
        "weighted_f1": "",
        "no_precision": "",
        "no_recall": "",
        "no_f1": "",
        "no_support": "",
        "intrinsic_precision": "",
        "intrinsic_recall": "",
        "intrinsic_f1": "",
        "intrinsic_support": "",
        "extrinsic_precision": "",
        "extrinsic_recall": "",
        "extrinsic_f1": "",
        "extrinsic_support": "",
        "malformed_count": "",
        "malformed_rate": "",
        "latency_mean_s": "",
        "latency_median_s": "",
        "latency_p95_s": "",
        "total_runtime_s": "",
        "rows_per_second": "",
        "peak_gpu_memory_gb": "",
        "dtype": entry.get("dtype", "bfloat16"),
        "quantization": entry.get("quantization", ""),
        "challenge_style_evaluation": bool(args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE")),
        "leakage_override": bool(args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE")),
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "artifact_dir": str(model_out_dir(args, entry)),
        "status": status,
        "skip_reason": skip_reason,
    }


def model_out_dir(args, entry):
    root = Path(args.out_root or "results/model_comparison")
    return root / entry["model_key"]


def write_status(out, row, extra=None):
    payload = dict(row)
    if extra:
        payload.update(extra)
    write_json(out / "status.json", payload)


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


def artifact_paths(out):
    return {
        "status": out / "status.json",
        "predictions": out / "predictions.csv",
        "summary": out / "summary_metrics.json",
        "report": out / "classification_report.json",
        "report_csv": out / "classification_report.csv",
        "confusion": out / "confusion_matrix.csv",
        "malformed": out / "malformed_predictions.csv",
    }


def compatible_completed_artifact(args, entry, expected_rows):
    out = model_out_dir(args, entry)
    paths = artifact_paths(out)
    if args.force_rerun_model or not paths["status"].exists():
        return None
    required = ["predictions", "summary", "report", "report_csv", "confusion", "malformed"]
    missing = [name for name in required if not paths[name].exists() or paths[name].stat().st_size == 0]
    if missing:
        print(f"STALE_ARTIFACT_MISSING {entry['model_key']} {','.join(missing)}")
        return None
    status = read_json(paths["status"])
    pred_rows = count_csv_rows(paths["predictions"])
    status_rows = int_or_none(status.get("rows"))
    expected_status_rows = int_or_none(status.get("expected_rows"))
    requested_limit = int_or_none(status.get("requested_limit"))
    current_limit = int_or_none(args.debug_limit)
    if pred_rows != int(expected_rows) or (status_rows is not None and status_rows != int(expected_rows)):
        print(f"STALE_ARTIFACT_ROWS_MISMATCH {entry['model_key']} expected={expected_rows} status_rows={status.get('rows')} prediction_rows={pred_rows}")
        return None
    if expected_status_rows is not None and expected_status_rows != int(expected_rows):
        print(f"STALE_ARTIFACT_ROWS_MISMATCH {entry['model_key']} expected={expected_rows} artifact_expected_rows={expected_status_rows}")
        return None
    if requested_limit != current_limit:
        print(f"STALE_ARTIFACT_LIMIT_MISMATCH {entry['model_key']} expected_limit={current_limit} artifact_limit={requested_limit}")
        return None
    checks = [
        status.get("status") == "completed",
        status.get("model_id") == entry["model_id"],
        status.get("method_type") == entry["method_type"],
        int_or_none(status.get("seed")) == int(args.seed),
        same_dataset_path(status.get("dataset_path"), args.gold_csv),
        int_or_none(status.get("malformed_count")) == 0,
    ]
    if not all(checks):
        print(f"STALE_ARTIFACT_METADATA_MISMATCH {entry['model_key']}")
        return None
    row = completed_summary(entry, args, out, extra_status={"artifact_source": status.get("artifact_source", "model_comparison_completed")})
    return row


def existing_resume_row(args, entry, expected_rows):
    return compatible_completed_artifact(args, entry, expected_rows)


def run_build_evidence(args, out, pred_path, malformed_path, latency_seconds, allow_partial):
    cmd = [
        sys.executable,
        "scripts/build_paper_evidence.py",
        "--gold_csv",
        args.gold_csv,
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
        str(latency_seconds),
        "--malformed_csv",
        str(malformed_path),
    ]
    if allow_partial:
        cmd.append("--allow_partial")
    if args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE"):
        cmd.append("--allow_known_public_split_leakage")
    subprocess.run(cmd, check=True)


def latency_fields(out, rows, fallback_seconds=0.0):
    path = out / "latency_summary.json"
    items = []
    if path.exists():
        loaded = read_json(path)
        if isinstance(loaded, list):
            items = loaded
    total = sum(float(item.get("total_seconds") or 0.0) for item in items) or float(fallback_seconds or 0.0)
    per_sample = [float(item.get("seconds_per_sample")) for item in items if item.get("seconds_per_sample") not in [None, ""]]
    mean_s = sum(per_sample) / len(per_sample) if per_sample else (total / rows if rows else 0.0)
    peak = 0
    for item in items:
        for key in ["cuda_max_memory_reserved", "cuda_max_memory_allocated"]:
            value = item.get(key)
            if value is not None:
                peak = max(peak, int(value))
    return {
        "latency_mean_s": float(mean_s),
        "latency_median_s": float(mean_s),
        "latency_p95_s": float(mean_s),
        "total_runtime_s": float(total),
        "rows_per_second": float(rows / total) if total > 0 and rows else 0.0,
        "peak_gpu_memory_gb": float(peak / (1024 ** 3)) if peak else 0.0,
    }


def completed_summary(entry, args, out, runtime_seconds=0.0, extra_status=None):
    summary = read_json(out / "summary_metrics.json")
    report = read_json(out / "classification_report.json")
    pred_rows = count_csv_rows(out / "predictions.csv")
    malformed_path = out / "malformed_predictions.csv"
    malformed_count = max(0, count_csv_rows(malformed_path)) if malformed_path.exists() else 0
    row = summary_blank(entry, args, "completed", "", pred_rows)
    row["accuracy"] = float(summary["accuracy"])
    row["macro_f1"] = float(summary["macro_f1"])
    row["weighted_f1"] = float(summary["weighted_f1"])
    for label in LABELS:
        values = report.get(label, {})
        row[f"{label}_precision"] = float(values.get("precision", 0.0))
        row[f"{label}_recall"] = float(values.get("recall", 0.0))
        row[f"{label}_f1"] = float(values.get("f1-score", 0.0))
        row[f"{label}_support"] = int(values.get("support", 0))
    row["malformed_count"] = int(malformed_count)
    row["malformed_rate"] = float(malformed_count / pred_rows) if pred_rows else 0.0
    row.update(latency_fields(out, pred_rows, runtime_seconds))
    for key in ["accuracy", "macro_f1", "weighted_f1", "malformed_rate"]:
        value = row[key]
        if not isinstance(value, (float, int)) or math.isnan(float(value)):
            raise ValueError(f"{entry['model_key']} produced invalid metric {key}: {value}")
    if int(row["malformed_count"]) != 0:
        raise ValueError(f"{entry['model_key']} produced malformed_count={row['malformed_count']}")
    status_extra = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "requested_limit": args.debug_limit,
        "expected_rows": int(pred_rows),
        "force_rerun": bool(args.force_rerun_model),
    }
    if extra_status:
        status_extra.update(extra_status)
    write_status(out, row, status_extra)
    return row


def count_csv_rows(path):
    import pandas as pd

    if not Path(path).exists():
        return 0
    return int(len(pd.read_csv(path)))


def copy_or_subset_predictions(source, target, limit=None):
    import pandas as pd

    df = pd.read_csv(source)
    if limit is not None:
        df = df.head(int(limit)).copy()
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return target


def validate_prediction_file(path, expected_rows):
    import pandas as pd

    df = pd.read_csv(path)
    if len(df) != int(expected_rows):
        raise SkipModel(f"source_prediction_row_count_mismatch:expected={expected_rows}:found={len(df)}")
    if "predict_label" not in df.columns:
        raise SkipModel("source_prediction_missing_predict_label")
    if df["predict_label"].isna().any():
        raise SkipModel("source_prediction_nan_predict_label")
    bad = sorted(set(df["predict_label"].astype(str)) - set(LABELS))
    if bad:
        raise SkipModel("source_prediction_invalid_labels:" + ",".join(bad))
    return True


def compatible_main_vistral_config(path, entry, args):
    if not Path(path).exists():
        raise SkipModel("main_current_best_missing_prediction_config")
    config = read_json(path)
    model_key = config.get("model_key")
    model_id = config.get("model_id") or config.get("canonical_model_id")
    if model_key and model_key != entry["model_key"]:
        raise SkipModel(f"main_current_best_model_key_mismatch:{model_key}")
    accepted = [entry["model_id"]] + list(entry.get("accepted_aliases", []))
    if model_id and not match_reference(model_id, accepted):
        raise SkipModel(f"main_current_best_model_id_mismatch:{model_id}")
    if not model_key and not model_id:
        raise SkipModel("main_current_best_missing_model_metadata")
    dataset_path = config.get("dataset_path")
    if dataset_path and not same_dataset_path(dataset_path, args.gold_csv):
        raise SkipModel(f"main_current_best_dataset_mismatch:{dataset_path}")
    malformed_count = int_or_none(config.get("malformed_count"))
    if malformed_count is not None and malformed_count != 0:
        raise SkipModel(f"main_current_best_malformed_count:{malformed_count}")
    return config


def copy_main_current_best_artifacts(out):
    sources = {
        "predictions.csv": Path("results/predictions.csv"),
        "prediction_config.json": Path("results/prediction_config.json"),
        "summary_metrics.json": Path("results/paper_evidence/summary_metrics.json"),
        "classification_report.json": Path("results/paper_evidence/classification_report.json"),
        "classification_report.csv": Path("results/paper_evidence/classification_report.csv"),
        "confusion_matrix.csv": Path("results/paper_evidence/confusion_matrix.csv"),
        "malformed_predictions.csv": Path("results/paper_evidence/malformed_predictions.csv"),
        "latency_summary.json": Path("results/paper_evidence/latency_summary.json"),
        "latency_summary.csv": Path("results/paper_evidence/latency_summary.csv"),
        "confusion_matrix.png": Path("results/paper_evidence/confusion_matrix.png"),
    }
    missing = [str(path) for path in sources.values() if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise SkipModel("main_current_best_missing_artifacts:" + ",".join(missing))
    out.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copy2(source, out / name)


def reuse_main_current_best(args, entry, expected_rows):
    if entry["model_key"] != "vistral" or args.force_rerun_model or args.debug_limit is not None:
        return None
    out = model_out_dir(args, entry)
    try:
        validate_prediction_file("results/predictions.csv", expected_rows)
        compatible_main_vistral_config("results/prediction_config.json", entry, args)
        malformed_rows = count_csv_rows("results/paper_evidence/malformed_predictions.csv")
        if malformed_rows != 0:
            raise SkipModel(f"main_current_best_malformed_rows:{malformed_rows}")
        copy_main_current_best_artifacts(out)
        row = completed_summary(entry, args, out, extra_status={
            "artifact_source": "main_current_best_reused",
            "source_predictions": "results/predictions.csv",
            "source_evidence_dir": "results/paper_evidence",
        })
        print("REUSED_MAIN_CURRENT_BEST vistral")
        return row
    except SkipModel as exc:
        print(f"MAIN_CURRENT_BEST_REUSE_UNAVAILABLE vistral {exc}")
        return None


def write_prediction_config(out, entry, args, extra):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "model_key": entry["model_key"],
        "model_id": entry["model_id"],
        "local_dir": entry["local_dir"],
        "method_type": entry["method_type"],
        "model_family": entry["model_family"],
        "inference_mode": entry["inference_mode"],
        "train_mode": entry["train_mode"],
        "seed": args.seed,
        "dataset_path": args.gold_csv,
        "dtype": entry.get("dtype", "bfloat16"),
        "quantization": entry.get("quantization", ""),
        "challenge_style_evaluation": bool(args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE")),
        "leakage_override": bool(args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE")),
    }
    payload.update(extra)
    write_json(out / "prediction_config.json", payload)


def run_vistral(entry, config, args, gold):
    out = model_out_dir(args, entry)
    out.mkdir(parents=True, exist_ok=True)
    expected_rows = len(gold)
    resumed = existing_resume_row(args, entry, expected_rows)
    if resumed is not None:
        print(f"MODEL_SKIPPED {entry['model_key']} resume_completed")
        return resumed
    pred_path = out / "predictions.csv"
    malformed_path = out / "malformed_predictions.csv"
    ready, reason = local_model_ready(entry)
    if not ready:
        raise SkipModel(reason)
    adapter_ok, adapter_reason = adapter_ready(entry)
    if not adapter_ok:
        raise SkipModel(adapter_reason)
    validate_alias(entry)
    cmd = [
        sys.executable,
        "scripts/generate_predictions_current_best.py",
        "--manifest",
        "configs/experiment_manifest.yaml",
        "--model-key",
        "vistral",
        "--model-dir",
        entry["local_dir"],
        "--adapter-dir",
        entry["adapter_dir"],
        "--gold-csv",
        args.gold_csv,
        "--out-csv",
        str(pred_path),
        "--config-json",
        str(out / "prediction_config.json"),
        "--malformed-csv",
        str(malformed_path),
        "--latency-out-dir",
        str(out),
        "--seed",
        str(args.seed),
    ]
    if args.debug_limit is not None:
        cmd += ["--limit", str(args.debug_limit)]
    elif pred_path.exists() and not args.force_rerun_model:
        current_rows = count_csv_rows(pred_path)
        if 0 < current_rows < expected_rows:
            print(f"PARTIAL_ARTIFACT_RESUME {entry['model_key']} rows={current_rows} expected={expected_rows}")
            cmd.append("--resume")
    if args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE"):
        cmd.append("--allow_known_public_split_leakage")
    subprocess.run(cmd, check=True)
    run_build_evidence(args, out, pred_path, malformed_path, 0.0, args.debug_limit is not None)
    return completed_summary(entry, args, out)


def run_encoder(entry, config, args, gold, train):
    ready, reason = local_model_ready(entry)
    if not ready:
        raise SkipModel(reason)
    validate_alias(entry)
    out = model_out_dir(args, entry)
    out.mkdir(parents=True, exist_ok=True)
    resumed = existing_resume_row(args, entry, len(gold))
    if resumed is not None:
        print(f"MODEL_SKIPPED {entry['model_key']} resume_completed")
        return resumed
    try:
        import numpy as np
        import pandas as pd
        import torch
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
        from src.evaluation.latency import save_latency_summary
        from src.utils.seed import set_seed
    except Exception as exc:
        raise SkipModel(f"missing_or_broken_dependency:{type(exc).__name__}: {exc}") from exc

    set_seed(args.seed)
    train_df = train.copy()
    if args.debug_limit is not None:
        train_df = train_df.head(max(16, int(args.debug_limit) * 4)).copy()
    train_df["text"] = train_df.apply(build_text, axis=1)
    train_df["labels"] = train_df["label"].map({label: index for index, label in enumerate(LABELS)})
    test_df = gold.copy()
    test_df["text"] = test_df.apply(build_text, axis=1)
    dataset = Dataset.from_pandas(train_df[["text", "labels"]].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[["text"]].reset_index(drop=True))
    tokenizer = load_encoder_tokenizer(AutoTokenizer, entry)
    max_length = int(config["encoder_defaults"]["max_length"])

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    test_tokenized = test_dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        entry["local_dir"],
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
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
        seed=int(args.seed),
        bf16=bool(torch.cuda.is_available()),
    )
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": tokenized,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
    }
    trainer_params = signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    start = time.perf_counter()
    trainer.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    infer_start = time.perf_counter()
    prediction = trainer.predict(test_tokenized)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - infer_start
    logits = prediction.predictions
    preds = np.argmax(logits, axis=1)
    rows = []
    for row_index, row in enumerate(gold.to_dict("records")):
        rows.append({
            "row_index": row_index,
            "id": row["id"],
            "label": row["label"],
            "predict_label": LABELS[int(preds[row_index])],
            "context": row["context"],
            "prompt": row["prompt"],
            "response": row["response"],
            "raw_output": json.dumps(logits[row_index].tolist()),
        })
    pred_path = out / "predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    memory = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
    save_latency_summary([{
        "name": entry["model_key"],
        "total_seconds": total_seconds,
        "samples": int(len(rows)),
        "seconds_per_sample": float(total_seconds / len(rows)) if rows else 0.0,
        "samples_per_second": float(len(rows) / total_seconds) if total_seconds > 0 and rows else 0.0,
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "cuda_max_memory_reserved": memory,
    }], out)
    malformed_path = write_empty_malformed(out / "malformed_predictions.csv")
    write_prediction_config(out, entry, args, {
        "train_rows": int(len(train_df)),
        "full_train_rows": int(len(train)),
        "runtime_seconds_training_plus_inference": float(time.perf_counter() - start),
        "resumed_from_existing_predictions": False,
        "skipped_rows": 0,
    })
    run_build_evidence(args, out, pred_path, malformed_path, total_seconds, args.debug_limit is not None)
    return completed_summary(entry, args, out, total_seconds)


def load_valid_prediction_rows(path):
    import pandas as pd

    rows = {}
    p = Path(path)
    if not p.exists():
        return rows, 0
    try:
        df = pd.read_csv(p)
    except Exception:
        return rows, 0
    required = {"row_index", "id", "predict_label"}
    if not required.issubset(df.columns):
        return rows, 0
    for item in df.to_dict("records"):
        try:
            row_index = int(item["row_index"])
        except Exception:
            continue
        label = str(item.get("predict_label", "")).strip()
        if label in LABELS:
            rows[row_index] = item
    return rows, len(rows)


def run_label_scoring(entry, config, args, gold):
    ready, reason = local_model_ready(entry)
    if not ready:
        raise SkipModel(reason)
    validate_alias(entry)
    out = model_out_dir(args, entry)
    out.mkdir(parents=True, exist_ok=True)
    resumed = existing_resume_row(args, entry, len(gold))
    if resumed is not None:
        print(f"MODEL_SKIPPED {entry['model_key']} resume_completed")
        return resumed
    try:
        import pandas as pd
        import torch
        from tqdm import tqdm
        from src.evaluation.latency import save_latency_summary
        from src.models.label_scoring import score_labels_causal_lm
        from src.models.loader import load_text_label_scoring_model
        from src.utils.seed import set_seed
    except Exception as exc:
        raise SkipModel(f"missing_or_broken_dependency:{type(exc).__name__}: {exc}") from exc

    set_seed(args.seed)
    pred_path = out / "predictions.csv"
    existing_rows, skipped_rows = load_valid_prediction_rows(pred_path)
    if args.force_rerun_model:
        existing_rows = {}
        skipped_rows = 0
    quantized = False
    try:
        model, tokenizer = load_text_label_scoring_model(entry["local_dir"], expected_loader=entry.get("expected_loader"), dtype=torch.bfloat16, quantized=False)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise SkipModel(f"unsupported_transformers_or_model_architecture:{type(exc).__name__}: {exc}") from exc
        release_cuda()
        try:
            model, tokenizer = load_text_label_scoring_model(entry["local_dir"], expected_loader=entry.get("expected_loader"), dtype=torch.bfloat16, quantized=True)
            quantized = True
        except Exception as retry_exc:
            raise SkipModel(f"oom_after_retry:{type(retry_exc).__name__}: {retry_exc}") from retry_exc
    except Exception as exc:
        raise SkipModel(f"unsupported_transformers_or_model_architecture:{type(exc).__name__}: {exc}") from exc

    device = next(model.parameters()).device
    rows = []
    debug_scores = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for row_index, row in enumerate(tqdm(gold.to_dict("records"), total=len(gold))):
        if row_index in existing_rows:
            rows.append(existing_rows[row_index])
            continue
        prompt = build_label_prompt(row)
        label, scores = score_labels_causal_lm(model, tokenizer, prompt, device=device, max_prompt_tokens=int(config.get("comparison_defaults", {}).get("max_prompt_tokens", 1024)))
        rows.append({
            "row_index": row_index,
            "id": row["id"],
            "label": row["label"],
            "predict_label": label,
            "context": row["context"],
            "prompt": row["prompt"],
            "response": row["response"],
            "raw_output": label,
        })
        if flag_enabled("DEBUG_LABEL_SCORES"):
            debug_scores.append({"row_index": row_index, "id": row["id"], "label_scores": scores})
        pd.DataFrame(sorted(rows, key=lambda item: int(item["row_index"]))).to_csv(pred_path, index=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - start
    rows = sorted(rows, key=lambda item: int(item["row_index"]))
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    if debug_scores:
        with (out / "label_scores.jsonl").open("w", encoding="utf-8") as handle:
            for item in debug_scores:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    save_latency_summary([{
        "name": entry["model_key"],
        "total_seconds": total_seconds,
        "samples": int(len(rows)),
        "seconds_per_sample": float(total_seconds / len(rows)) if rows else 0.0,
        "samples_per_second": float(len(rows) / total_seconds) if total_seconds > 0 and rows else 0.0,
        "cuda_max_memory_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "cuda_max_memory_reserved": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
    }], out)
    malformed_path = write_empty_malformed(out / "malformed_predictions.csv")
    write_prediction_config(out, entry, args, {
        "resumed_from_existing_predictions": bool(skipped_rows),
        "skipped_rows": int(skipped_rows),
        "completed_rows": int(len(rows)),
        "total_rows": int(len(gold)),
        "force_rerun": bool(args.force_rerun_model),
        "quantized_runtime": bool(quantized),
    })
    run_build_evidence(args, out, pred_path, malformed_path, total_seconds, args.debug_limit is not None)
    return completed_summary(entry, args, out, total_seconds)


def run_supervised_peft(entry, config, args, gold, train):
    out = model_out_dir(args, entry)
    out.mkdir(parents=True, exist_ok=True)
    resumed = existing_resume_row(args, entry, len(gold))
    if resumed is not None:
        print(f"MODEL_SKIPPED {entry['model_key']} resume_completed")
        return resumed
    cmd = [
        sys.executable,
        "scripts/train_eval_llm_peft_baseline.py",
        "--model-key",
        entry["model_key"],
        "--config",
        args.config,
        "--train-csv",
        args.train_csv,
        "--gold-csv",
        args.gold_csv,
        "--out-root",
        args.out_root,
        "--seed",
        str(args.seed),
        "--epochs",
        str(entry.get("epochs", 2)),
    ]
    if args.debug_limit is not None:
        cmd += ["--debug-limit", str(args.debug_limit)]
    if args.force_rerun_model:
        cmd.append("--force-rerun-model")
    subprocess.run(cmd, check=True)
    return completed_summary(entry, args, out)


def run_model(entry, config, args, gold, train):
    model_type = entry["type"]
    if model_type == "current_best_peft":
        return run_vistral(entry, config, args, gold)
    if model_type == "encoder_classifier":
        return run_encoder(entry, config, args, gold, train)
    if model_type == "label_scoring_lm":
        return run_label_scoring(entry, config, args, gold)
    if model_type == "supervised_peft_finetune":
        return run_supervised_peft(entry, config, args, gold, train)
    raise SkipModel(f"unsupported_model_type:{model_type}")


def write_failure(out, entry, args, status, reason):
    out.mkdir(parents=True, exist_ok=True)
    row = summary_blank(entry, args, status, reason)
    write_status(out, row, {"timestamp": datetime.now(timezone.utc).isoformat(), "traceback": traceback.format_exc()})
    path = out / "failure_report.md"
    path.write_text(
        "\n".join([
            f"# Model Comparison Failure: {entry['model_key']}",
            "",
            f"* timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"* status: {status}",
            f"* reason: {reason}",
            "",
        ]),
        encoding="utf-8",
    )
    return row


def write_summary_tables(rows, out_root):
    import pandas as pd

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[SUMMARY_COLUMNS]
    completed = df[df["status"] == "completed"].copy()
    skipped = df[df["status"] != "completed"].copy()
    if len(completed):
        completed["macro_f1_sort"] = completed["macro_f1"].astype(float)
        completed = completed.sort_values("macro_f1_sort", ascending=False).drop(columns=["macro_f1_sort"])
    ordered = pd.concat([completed, skipped], ignore_index=True)
    summary_csv = out / "model_comparison_summary.csv"
    ordered.to_csv(summary_csv, index=False)
    if len(skipped):
        skipped.to_csv(out / "skipped_models.csv", index=False)
    else:
        pd.DataFrame(columns=SUMMARY_COLUMNS).to_csv(out / "skipped_models.csv", index=False)
    lines = ["# Model Comparison Summary", ""]
    visible = ["model_key", "model_id", "method_type", "rows", "accuracy", "macro_f1", "weighted_f1", "status", "skip_reason"]
    lines.extend(markdown_table(ordered[visible]))
    lines.append("")
    (out / "model_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(out / "model_comparison_status.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "completed": int((ordered["status"] == "completed").sum()),
        "non_completed": int((ordered["status"] != "completed").sum()),
        "rows": ordered.to_dict("records"),
    })
    return ordered


def markdown_table(df):
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for item in df.fillna("").astype(str).to_dict("records"):
        rows.append("| " + " | ".join(item[col].replace("|", "\\|") for col in columns) + " |")
    return rows


def maybe_download_models(config, args):
    if not args.download_missing:
        return
    from src.models.download import download_entry
    from src.utils.env import load_hf_token

    token = load_hf_token(required=True)
    results = []
    for name, entry in config["baselines"].items():
        if not entry.get("enabled"):
            continue
        if args.only_model_key and name != args.only_model_key:
            continue
        results.append(download_entry(name, entry, token=token))
    write_json(Path(args.out_root) / "download_report.json", {"timestamp": datetime.now(timezone.utc).isoformat(), "results": results})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_models.yaml")
    parser.add_argument("--gold_csv", default=None)
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--out_root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--allow_known_public_split_leakage", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=None)
    parser.add_argument("--only-model-key", default=None)
    parser.add_argument("--force-rerun-model", action="store_true")
    parser.add_argument("--allow-stale-resume", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--rebuild-summary-only", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate_config(config)
    args.gold_csv = args.gold_csv or config["test_csv"]
    args.train_csv = args.train_csv or config["train_csv"]
    args.out_root = args.out_root or config.get("output_root", "results/model_comparison")
    args.debug_limit = args.debug_limit if args.debug_limit is not None else (int(os.getenv("DEBUG_LIMIT")) if os.getenv("DEBUG_LIMIT") else None)
    args.only_model_key = args.only_model_key or os.getenv("ONLY_MODEL_KEY")
    args.force_rerun_model = bool(args.force_rerun_model or flag_enabled("FORCE_RERUN_MODEL"))
    args.download_missing = bool(args.download_missing or flag_enabled("RUN_DOWNLOAD_MODELS"))
    args.rebuild_summary_only = bool(args.rebuild_summary_only or flag_enabled("REBUILD_MODEL_COMPARISON_SUMMARY"))
    enabled = [name for name, item in config["baselines"].items() if item.get("enabled") and (not args.only_model_key or name == args.only_model_key)]
    if args.dry_run:
        print(f"DRY_RUN_OK enabled_models={enabled} out_root={args.out_root}")
        return
    from src.data.vihallu import load_vihallu_split, read_csv_robust, validate_gold_df, validate_or_report_public_split_leakage

    gold_full = read_csv_robust(args.gold_csv)
    validate_gold_df(gold_full, args.gold_csv, allow_duplicate_ids=True)
    if len(gold_full) != 14000 and args.debug_limit is None:
        raise ValueError(f"{args.gold_csv} must contain exactly 14000 rows, found {len(gold_full)}")
    train = read_csv_robust(args.train_csv)
    validate_gold_df(train, args.train_csv, allow_duplicate_ids=False)
    validate_or_report_public_split_leakage(
        load_vihallu_split("train"),
        load_vihallu_split("test"),
        allow_known_public_split_leakage=args.allow_known_public_split_leakage,
        report_path=Path(args.out_root) / "leakage_report.md",
    )
    gold = gold_full.head(args.debug_limit).copy().reset_index(drop=True) if args.debug_limit is not None else gold_full.copy()
    strict = bool(args.strict or flag_enabled("STRICT_BASELINES"))
    if args.force_rerun_model:
        print("FORCE_RERUN_MODEL_ACTIVE")
    if not args.rebuild_summary_only:
        maybe_download_models(config, args)
    rows = []
    for name in enabled:
        entry = dict(config["baselines"][name])
        out = model_out_dir(args, entry)
        try:
            expected_rows = len(gold)
            row = compatible_completed_artifact(args, entry, expected_rows)
            if row is not None:
                rows.append(row)
                print(f"MODEL_ALREADY_COMPLETED {name}")
                continue
            row = reuse_main_current_best(args, entry, expected_rows)
            if row is not None:
                rows.append(row)
                print(f"MODEL_COMPLETED {name}")
                continue
            if args.rebuild_summary_only:
                rows.append(summary_blank(entry, args, "skipped", "no_compatible_completed_artifact"))
                print(f"MODEL_SKIPPED {name} no_compatible_completed_artifact")
                continue
            print(f"RUN_MODEL {name}")
            row = run_model(entry, config, args, gold, train)
            rows.append(row)
            print(f"MODEL_COMPLETED {name}")
        except SkipModel as exc:
            reason = str(exc)
            rows.append(write_failure(out, entry, args, "skipped", reason))
            print(f"MODEL_SKIPPED {name} {reason}")
            if strict:
                raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            rows.append(write_failure(out, entry, args, "failed", reason))
            print(f"MODEL_SKIPPED {name} {reason}")
            if strict:
                raise
        finally:
            release_cuda()
    ordered = write_summary_tables(rows, args.out_root)
    print("MODEL_COMPARISON_DONE")
    print(ordered[["model_key", "model_id", "method_type", "rows", "accuracy", "macro_f1", "status", "skip_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
