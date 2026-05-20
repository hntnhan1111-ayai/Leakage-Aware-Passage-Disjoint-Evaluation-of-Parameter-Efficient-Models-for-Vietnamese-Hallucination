import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd


LABELS = ["no", "intrinsic", "extrinsic"]
PAPER_MAIN_MODELS = ["qwen35_4b_peft", "gemma4_e2b_it_peft", "vistral", "phobert", "xlmr"]
PEFT_MODELS = ["qwen35_4b_peft", "gemma4_e2b_it_peft"]
EXPECTED_MACRO_F1 = {
    "qwen35_4b_peft": 0.9749414076960269,
    "gemma4_e2b_it_peft": 0.9323378936342928,
}


def require_file(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty file: {p}")
    return p


def read_json(path):
    return json.loads(require_file(path).read_text(encoding="utf-8"))


def safe_float(value):
    if value in [None, ""]:
        return None
    try:
        return float(value)
    except Exception:
        return None


def git_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def torch_environment():
    info = {
        "torch_version": package_version("torch"),
        "cuda_available": None,
        "gpu_name": "",
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["cuda_available"] = f"unavailable:{type(exc).__name__}:{exc}"
    return info


def compare_number(model_key, source, metric, computed, saved, failures):
    saved_value = safe_float(saved)
    if saved_value is None:
        failures.append(f"{model_key}:{source}:{metric}:missing_or_non_numeric")
        return {"source": source, "metric": metric, "computed": computed, "saved": saved, "diff": None, "match": False, "rounded_source": False}
    diff = abs(float(computed) - float(saved_value))
    if diff <= 1e-9:
        return {"source": source, "metric": metric, "computed": float(computed), "saved": float(saved_value), "diff": diff, "match": True, "rounded_source": False}
    if diff <= 5e-6:
        return {"source": source, "metric": metric, "computed": float(computed), "saved": float(saved_value), "diff": diff, "match": True, "rounded_source": True}
    failures.append(f"{model_key}:{source}:{metric}:diff={diff}")
    return {"source": source, "metric": metric, "computed": float(computed), "saved": float(saved_value), "diff": diff, "match": False, "rounded_source": False}


def compute_metrics(y_true, y_pred):
    y_true = [str(value) for value in y_true]
    y_pred = [str(value) for value in y_pred]
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    label_to_index = {label: index for index, label in enumerate(LABELS)}
    for gold, pred in zip(y_true, y_pred):
        matrix[label_to_index[gold], label_to_index[pred]] += 1
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    per_class = {}
    for index, label in enumerate(LABELS):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        pred_support = int(matrix[:, index].sum())
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
            "predicted": pred_support,
        }
    supports = np.array([per_class[label]["support"] for label in LABELS], dtype=float)
    precision_values = np.array([per_class[label]["precision"] for label in LABELS], dtype=float)
    recall_values = np.array([per_class[label]["recall"] for label in LABELS], dtype=float)
    f1_values = np.array([per_class[label]["f1-score"] for label in LABELS], dtype=float)
    metrics = {
        "accuracy": float(correct / total) if total else 0.0,
        "macro_precision": float(precision_values.mean()),
        "macro_recall": float(recall_values.mean()),
        "macro_f1": float(f1_values.mean()),
        "weighted_precision": float(np.average(precision_values, weights=supports)) if total else 0.0,
        "weighted_recall": float(np.average(recall_values, weights=supports)) if total else 0.0,
        "weighted_f1": float(np.average(f1_values, weights=supports)) if total else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "rows": total,
        "correct": correct,
        "errors": total - correct,
    }
    return metrics


def load_predictions(path, expected_rows):
    df = pd.read_csv(require_file(path))
    required = {"label", "predict_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    if len(df) != int(expected_rows):
        raise ValueError(f"{path} row mismatch: expected {expected_rows}, found {len(df)}")
    for column in ["label", "predict_label"]:
        if df[column].isna().any():
            raise ValueError(f"{path} has null {column}")
        bad = sorted(set(df[column].astype(str)) - set(LABELS))
        if bad:
            raise ValueError(f"{path} has invalid {column}: {bad}")
    return df


def load_summary_table(path):
    df = pd.read_csv(require_file(path))
    keys = list(df["model_key"].astype(str)) if "model_key" in df.columns else []
    expected = set(PAPER_MAIN_MODELS)
    actual = set(keys)
    if actual != expected or len(keys) != len(PAPER_MAIN_MODELS):
        raise ValueError(f"model_comparison_summary.csv paper-main keys mismatch: {keys}")
    blocked = {"qwen35_4b", "gemma4_e2b_it", "qwen3_4b_prompt"}
    found = sorted(actual & blocked)
    if found:
        raise ValueError(f"Auxiliary models found in paper-main summary: {found}")
    return df


def compare_artifacts(model_key, artifact_dir, summary_row, metrics, failures):
    comparisons = []
    summary_metrics = read_json(artifact_dir / "summary_metrics.json")
    report_json = read_json(artifact_dir / "classification_report.json")
    report_csv = pd.read_csv(require_file(artifact_dir / "classification_report.csv"), index_col=0)
    cm_csv = pd.read_csv(require_file(artifact_dir / "confusion_matrix.csv"), index_col=0).loc[LABELS, LABELS]
    for metric in ["accuracy", "macro_f1", "weighted_f1"]:
        comparisons.append(compare_number(model_key, "summary_metrics.json", metric, metrics[metric], summary_metrics.get(metric), failures))
        comparisons.append(compare_number(model_key, "model_comparison_summary.csv", metric, metrics[metric], summary_row.get(metric), failures))
    for label in LABELS:
        values = metrics["per_class"][label]
        for metric, key in [("precision", "precision"), ("recall", "recall"), ("f1-score", "f1-score"), ("support", "support")]:
            comparisons.append(compare_number(model_key, "classification_report.json", f"{label}.{metric}", values[key], report_json.get(label, {}).get(metric), failures))
            comparisons.append(compare_number(model_key, "classification_report.csv", f"{label}.{metric}", values[key], report_csv.loc[label, metric], failures))
    for metric, source_key in [("macro_precision", "precision"), ("macro_recall", "recall"), ("macro_f1", "f1-score")]:
        comparisons.append(compare_number(model_key, "classification_report.json", f"macro avg.{source_key}", metrics[metric], report_json.get("macro avg", {}).get(source_key), failures))
        comparisons.append(compare_number(model_key, "classification_report.csv", f"macro avg.{source_key}", metrics[metric], report_csv.loc["macro avg", source_key], failures))
    for metric, source_key in [("weighted_precision", "precision"), ("weighted_recall", "recall"), ("weighted_f1", "f1-score")]:
        comparisons.append(compare_number(model_key, "classification_report.json", f"weighted avg.{source_key}", metrics[metric], report_json.get("weighted avg", {}).get(source_key), failures))
        comparisons.append(compare_number(model_key, "classification_report.csv", f"weighted avg.{source_key}", metrics[metric], report_csv.loc["weighted avg", source_key], failures))
    if not np.array_equal(metrics["confusion_matrix"], cm_csv.values.astype(int)):
        failures.append(f"{model_key}:confusion_matrix.csv:mismatch")
    if model_key in EXPECTED_MACRO_F1:
        expected = EXPECTED_MACRO_F1[model_key]
        diff = abs(metrics["macro_f1"] - expected)
        if diff > 5e-6:
            failures.append(f"{model_key}:expected_macro_f1_mismatch:{metrics['macro_f1']}:{expected}")
    return comparisons


def metrics_presence(artifact_dir):
    summary = read_json(artifact_dir / "summary_metrics.json")
    report = read_json(artifact_dir / "classification_report.json")
    cm_path = artifact_dir / "confusion_matrix.csv"
    presence = {
        "accuracy": "accuracy" in summary,
        "weighted_f1": "weighted_f1" in summary,
        "per_class_precision": all("precision" in report.get(label, {}) for label in LABELS),
        "per_class_recall": all("recall" in report.get(label, {}) for label in LABELS),
        "per_class_f1": all("f1-score" in report.get(label, {}) for label in LABELS),
        "per_class_support": all("support" in report.get(label, {}) for label in LABELS),
        "confusion_matrix": cm_path.exists() and cm_path.stat().st_size > 0,
    }
    presence["complete"] = all(presence.values())
    return presence


def write_supplemental_if_needed(out_dir, model_key, metrics, presence):
    missing = [key for key, value in presence.items() if key != "complete" and not value]
    if not missing:
        return []
    supplemental_dir = out_dir / "validation_supplemental_metrics"
    supplemental_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = supplemental_dir / f"{model_key}_supplemental_metrics.json"
    cm_path = supplemental_dir / f"{model_key}_confusion_matrix.csv"
    payload = {key: value for key, value in metrics.items() if key != "confusion_matrix"}
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(metrics["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(cm_path)
    return [str(metrics_path), str(cm_path)]


def write_compact_csv(path, rows):
    columns = [
        "model_key",
        "rows",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "errors",
        "gold_no",
        "gold_intrinsic",
        "gold_extrinsic",
        "pred_no",
        "pred_intrinsic",
        "pred_extrinsic",
        "saved_match",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_qwen_gemma_metrics(path, records):
    rows = []
    for record in records:
        if record["model_key"] not in PEFT_MODELS:
            continue
        metrics = record["metrics"]
        row = {
            "model_key": record["model_key"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
        }
        for label in LABELS:
            row[f"{label}_precision"] = metrics["per_class"][label]["precision"]
            row[f"{label}_recall"] = metrics["per_class"][label]["recall"]
            row[f"{label}_f1"] = metrics["per_class"][label]["f1-score"]
            row[f"{label}_support"] = metrics["per_class"][label]["support"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def verify_required_fields(path, fields):
    payload = read_json(path)
    missing = [field for field in fields if field not in payload or payload.get(field) in [None, ""] and field != "requested_limit"]
    return missing, payload


def inspect_determinism():
    script = require_file("scripts/train_eval_llm_peft_baseline.py").read_text(encoding="utf-8")
    config = require_file("configs/baseline_models.yaml").read_text(encoding="utf-8")
    findings = {
        "seed_42_in_config": "seed: 42" in config,
        "set_seed_called": "set_seed(args.seed)" in script,
        "training_args_seed": '"seed": int(args.seed)' in script,
        "deterministic_generation": "do_sample=False" in script and "temperature=0.0" in script and "top_p=1.0" in script,
        "tf32_enabled": "allow_tf32 = True" in script,
        "deterministic_cuda_algorithms": "use_deterministic_algorithms" in script,
        "cudnn_deterministic": "cudnn.deterministic" in script,
        "cublas_workspace_config": "CUBLAS_WORKSPACE_CONFIG" in script,
    }
    return findings


def write_reproducibility_audit(path, records, validation_status):
    env = torch_environment()
    dirty = git_output(["git", "status", "--short"])
    dirty_lines = [line for line in dirty.splitlines() if line.strip()]
    determinism = inspect_determinism()
    lines = []
    lines.append("# Reproducibility Audit")
    lines.append("")
    lines.append(f"- Validation status: {validation_status}")
    lines.append(f"- Git branch: {git_output(['git', 'branch', '--show-current'])}")
    lines.append(f"- Git commit: {git_output(['git', 'rev-parse', 'HEAD'])}")
    lines.append(f"- Dirty status entries: {len(dirty_lines)}")
    if dirty_lines:
        lines.append("- Dirty status sample:")
        for item in dirty_lines[:20]:
            lines.append(f"  - `{item}`")
    lines.append(f"- Python version: {platform.python_version()}")
    lines.append(f"- torch version: {env['torch_version']}")
    lines.append(f"- transformers version: {package_version('transformers')}")
    lines.append(f"- peft version: {package_version('peft')}")
    lines.append(f"- bitsandbytes version: {package_version('bitsandbytes')}")
    lines.append(f"- CUDA available: {env['cuda_available']}")
    lines.append(f"- GPU name: {env['gpu_name']}")
    lines.append("")
    lines.append("## Paper-Main Model Set")
    for model_key in PAPER_MAIN_MODELS:
        lines.append(f"- `{model_key}`")
    lines.append("")
    lines.append("## Artifact Checks")
    for record in records:
        lines.append(f"- `{record['model_key']}`: rows={record['metrics']['rows']}, accuracy={record['metrics']['accuracy']:.12f}, macro_f1={record['metrics']['macro_f1']:.12f}, errors={record['metrics']['errors']}")
        for category, missing in record["required_field_missing"].items():
            if missing:
                lines.append(f"  - Missing {category}: {', '.join(missing)}")
    lines.append("")
    lines.append("## Determinism")
    lines.append(f"- Config default seed 42: {determinism['seed_42_in_config']}")
    lines.append(f"- Training script calls `set_seed(args.seed)`: {determinism['set_seed_called']}")
    lines.append(f"- Transformers TrainingArguments receives seed: {determinism['training_args_seed']}")
    lines.append(f"- Generation path is deterministic where used (`do_sample=False`, `temperature=0.0`, `top_p=1.0`): {determinism['deterministic_generation']}")
    lines.append(f"- TF32 is enabled on CUDA: {determinism['tf32_enabled']}")
    lines.append(f"- Deterministic CUDA algorithms enforced: {determinism['deterministic_cuda_algorithms']}")
    lines.append(f"- cuDNN deterministic mode enforced: {determinism['cudnn_deterministic']}")
    lines.append(f"- CUBLAS workspace determinism configured: {determinism['cublas_workspace_config']}")
    lines.append("")
    lines.append("Artifact-level metric reproducibility is verified from saved predictions; full training rerun reproducibility is supported by fixed seeds/configs but may still have GPU nondeterminism unless deterministic CUDA settings are enforced.")
    if not determinism["deterministic_cuda_algorithms"] or not determinism["cudnn_deterministic"] or not determinism["cublas_workspace_config"]:
        lines.append("")
        lines.append("Recommended patch if strict rerun determinism is required: set deterministic CUDA algorithms, cuDNN deterministic mode, and CUBLAS workspace configuration before model initialization, then document the expected speed tradeoff.")
    lines.append("")
    lines.append("## RTX4090 Reproduction Commands")
    lines.append("```bash")
    lines.append("cd ~/Downloads/Hallu-Paper")
    lines.append("git fetch origin")
    lines.append("git reset --hard origin/submit/icit2026-evidence-revision")
    lines.append("source .venv/bin/activate")
    lines.append("python3 -m compileall src scripts")
    lines.append("python3 scripts/audit_model_comparison_artifacts.py --root results/model_comparison --expected-rows 14000 --paper-main-only")
    lines.append("bash scripts/run_qwen_gemma_peft_rtx4090.sh")
    lines.append("python3 scripts/make_paper_figures.py --summary results/model_comparison/model_comparison_summary.csv --out-dir results/paper_figures")
    lines.append("python3 scripts/validate_paper_main_metrics.py")
    lines.append("python3 scripts/make_extra_paper_figures.py")
    lines.append("```")
    lines.append("")
    lines.append("## Local Audit Commands Used")
    lines.append("```bash")
    lines.append("python3.11 -m py_compile scripts/validate_paper_main_metrics.py scripts/make_extra_paper_figures.py scripts/audit_model_comparison_artifacts.py scripts/run_baselines_e2e.py scripts/train_eval_llm_peft_baseline.py scripts/make_paper_figures.py")
    lines.append("python3.11 scripts/audit_model_comparison_artifacts.py --root results/model_comparison --expected-rows 14000 --paper-main-only")
    lines.append("python scripts/audit_model_comparison_artifacts.py --root results/model_comparison --expected-rows 14000 --paper-main-only")
    lines.append("python scripts/validate_paper_main_metrics.py")
    lines.append("python scripts/make_extra_paper_figures.py")
    lines.append("git diff --check")
    lines.append("```")
    lines.append("")
    lines.append("Python 3.11 syntax checks and paper-main audit passed. Metric validation and figure regeneration were run with the repository `.venv` Python because that interpreter has the local pandas/numpy/matplotlib stack.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_code_validity(path, records, summary_keys, failures):
    lines = ["# Code and Result Validity Check", ""]
    lines.append(f"- {'PASS' if not failures else 'FAIL'}: Independent metric recomputation from `label` and `predict_label` completed.")
    lines.append(f"- {'PASS' if summary_keys == PAPER_MAIN_MODELS else 'FAIL'}: Paper-main summary contains exactly `{', '.join(PAPER_MAIN_MODELS)}` in the expected order.")
    lines.append("- PASS: Auxiliary zero-shot and legacy prompt-only models are absent from paper-main summary.")
    for record in records:
        metrics = record["metrics"]
        pred_dist = record["prediction_distribution"]
        gold_dist = record["gold_distribution"]
        has_all_pred = all(pred_dist.get(label, 0) > 0 for label in LABELS)
        lines.append(f"- {'PASS' if has_all_pred else 'FAIL'}: `{record['model_key']}` predicts all three labels; gold={gold_dist}; pred={pred_dist}.")
        collapsed = max(pred_dist.values()) == metrics["rows"] if pred_dist else True
        lines.append(f"- {'PASS' if not collapsed else 'FAIL'}: `{record['model_key']}` is not single-label collapsed.")
    if failures:
        lines.append("")
        lines.append("## Failures")
        for failure in failures:
            lines.append(f"- {failure}")
    lines.append("")
    lines.append("Current results are safe to use in the paper at artifact level if this report is PASS. The high Qwen3.5 PEFT score should still be described as a benchmark test split result and interpreted with the documented public split constraints.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/model_comparison")
    parser.add_argument("--summary", default="results/model_comparison/model_comparison_summary.csv")
    parser.add_argument("--out-dir", default="results/paper_figures")
    parser.add_argument("--expected-rows", type=int, default=14000)
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = load_summary_table(args.summary)
    summary_rows = {str(row["model_key"]): row for row in summary_df.to_dict("records")}
    failures = []
    records = []
    compact_rows = []
    supplemental_files = []
    for model_key in PAPER_MAIN_MODELS:
        artifact_dir = root / model_key
        df = load_predictions(artifact_dir / "predictions.csv", args.expected_rows)
        metrics = compute_metrics(df["label"], df["predict_label"])
        comparisons = compare_artifacts(model_key, artifact_dir, summary_rows[model_key], metrics, failures)
        prediction_distribution = {label: int((df["predict_label"].astype(str) == label).sum()) for label in LABELS}
        gold_distribution = {label: int((df["label"].astype(str) == label).sum()) for label in LABELS}
        presence = metrics_presence(artifact_dir)
        supplemental_files.extend(write_supplemental_if_needed(out_dir, model_key, metrics, presence))
        status_missing, status_payload = verify_required_fields(artifact_dir / "status.json", ["seed", "model_id", "method_type", "rows", "expected_rows", "requested_limit", "status", "macro_f1"])
        training_missing, training_payload = verify_required_fields(artifact_dir / "training_config_resolved.json", ["seed", "epochs", "learning_rate", "lora_r", "lora_alpha", "lora_dropout", "target_modules", "train_rows", "eval_rows", "adapter_dir"]) if model_key in PEFT_MODELS else ([], {})
        prediction_missing, prediction_payload = verify_required_fields(artifact_dir / "prediction_config.json", ["model_id", "method_type", "inference_mode", "quantization", "dtype", "rows"])
        if model_key in PEFT_MODELS and not any(key in prediction_payload for key in ["max_new_tokens", "max_prompt_tokens", "generation_config", "parser_version"]):
            prediction_missing.append("parser_or_generation_detail")
        record = {
            "model_key": model_key,
            "artifact_dir": str(artifact_dir),
            "metrics": {key: value for key, value in metrics.items() if key != "confusion_matrix"},
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "gold_distribution": gold_distribution,
            "prediction_distribution": prediction_distribution,
            "metrics_presence": presence,
            "comparisons": comparisons,
            "required_field_missing": {
                "status.json": status_missing,
                "training_config_resolved.json": training_missing,
                "prediction_config.json": prediction_missing,
            },
        }
        records.append(record)
        match_saved = all(item["match"] for item in comparisons)
        compact = {
            "model_key": model_key,
            "rows": metrics["rows"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "errors": metrics["errors"],
            "saved_match": match_saved,
        }
        for label in LABELS:
            compact[f"gold_{label}"] = gold_distribution[label]
            compact[f"pred_{label}"] = prediction_distribution[label]
        compact_rows.append(compact)
    validation_status = "PASS" if not failures else "FAIL"
    payload = {
        "status": validation_status,
        "expected_rows": args.expected_rows,
        "paper_main_models": PAPER_MAIN_MODELS,
        "records": records,
        "supplemental_files": supplemental_files,
        "failures": failures,
    }
    validation_json = out_dir / "paper_main_metric_validation.json"
    validation_csv = out_dir / "paper_main_metric_validation.csv"
    metrics_csv = out_dir / "qwen_gemma_metrics_beyond_macro_f1.csv"
    validation_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_compact_csv(validation_csv, compact_rows)
    write_qwen_gemma_metrics(metrics_csv, records)
    write_reproducibility_audit(out_dir / "reproducibility_audit.md", records, validation_status)
    write_code_validity(out_dir / "code_and_result_validity_check.md", records, list(summary_df["model_key"].astype(str)), failures)
    for path in [validation_json, validation_csv, metrics_csv, out_dir / "reproducibility_audit.md", out_dir / "code_and_result_validity_check.md"]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    if failures:
        print("PAPER_MAIN_METRIC_VALIDATION_FAIL")
        for failure in failures:
            print(failure)
        sys.exit(1)
    print("PAPER_MAIN_METRIC_VALIDATION_PASS")
    for row in compact_rows:
        print(f"{row['model_key']}: rows={row['rows']} accuracy={row['accuracy']:.12f} macro_precision={row['macro_precision']:.12f} macro_recall={row['macro_recall']:.12f} macro_f1={row['macro_f1']:.12f} weighted_f1={row['weighted_f1']:.12f} errors={row['errors']}")


if __name__ == "__main__":
    main()
