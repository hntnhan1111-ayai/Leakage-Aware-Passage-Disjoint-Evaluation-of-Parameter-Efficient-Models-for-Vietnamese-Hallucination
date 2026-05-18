import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


LABELS = ["no", "intrinsic", "extrinsic"]


def resolve_gold(path):
    if path:
        return Path(path)
    from src.data.vihallu import first_existing

    return first_existing(["vihallu-test.csv", "data/vihallu-test.csv", "materials/vihallu-test.csv"])


def resolve_pred(path):
    if path:
        return Path(path)
    from src.data.vihallu import first_existing

    return first_existing(["final_submission_scratch.csv", "results/predictions.csv", "results/paper_evidence/predictions.csv"])


def validate_columns(df, cols, path):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}. Available columns: {list(df.columns)}")
    return True


def write_error_cases(wrong, out_path, max_cases=9):
    import pandas as pd

    rows = []
    if len(wrong) > 0:
        wrong = wrong.copy()
        wrong["confusion_type"] = wrong["label"].astype(str) + "_to_" + wrong["predict_label"].astype(str)
        for _, group in wrong.groupby("confusion_type", sort=True):
            rows.append(group.head(1))
        sample = pd.concat(rows).head(max_cases) if rows else wrong.head(max_cases)
    else:
        sample = wrong
    lines = ["# Selected Error Cases", ""]
    if len(sample) == 0:
        lines.append("No wrong predictions found.")
    for i, row in enumerate(sample.to_dict("records"), 1):
        lines.append(f"## Case {i}: {row.get('label')} -> {row.get('predict_label')}")
        lines.append(f"- id: {row.get('id')}")
        lines.append(f"- context: {str(row.get('context', ''))[:1000]}")
        lines.append(f"- prompt: {str(row.get('prompt', ''))[:500]}")
        lines.append(f"- response: {str(row.get('response', ''))[:1000]}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def validate_prediction_labels(values, path):
    import pandas as pd

    bad = sorted(set(pd.Series(values).dropna().astype(str)) - set(LABELS))
    if bad:
        raise ValueError(f"{path} contains invalid prediction labels: {bad}")
    return True


def validate_summary(summary):
    import math

    for key in ["accuracy", "macro_f1", "weighted_f1"]:
        value = summary.get(key)
        if not isinstance(value, (float, int)) or math.isnan(float(value)):
            raise ValueError(f"summary_metrics.json contains invalid {key}: {value}")
    if summary.get("labels") != LABELS:
        raise ValueError(f"summary_metrics.json labels must be {LABELS}")
    if "classification_report" not in summary:
        raise ValueError("summary_metrics.json missing classification_report")
    return True


def write_validation_report(out, checks):
    lines = ["# Evidence Validation Report", ""]
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        lines.append(f"* {status}: {name}")
    lines.append("")
    lines.append("Overall: " + ("PASS" if not failed else "FAIL"))
    path = out / "validation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    if failed:
        raise RuntimeError("Evidence validation failed: " + ", ".join(failed))
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv", default=None)
    parser.add_argument("--pred_csv", default=None)
    parser.add_argument("--out_dir", default="results/paper_evidence")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--pred_col", default="predict_label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency_seconds", type=float, default=None)
    parser.add_argument("--allow_partial", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    from src.data.vihallu import read_csv_robust, validate_gold_df
    from src.evaluation.latency import save_latency_summary
    from src.evaluation.metrics import compute_and_save
    from src.utils.seed import set_seed

    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gold_path = resolve_gold(args.gold_csv)
    pred_path = resolve_pred(args.pred_csv)
    gold = read_csv_robust(gold_path)
    pred = read_csv_robust(pred_path)
    validate_gold_df(gold, gold_path)
    validate_columns(gold, ["id", "context", "prompt", "response", args.label_col], gold_path)
    validate_columns(pred, ["id", args.pred_col], pred_path)
    if gold["id"].duplicated().any():
        raise ValueError(f"{gold_path} contains duplicate id values")
    if pred["id"].duplicated().any():
        raise ValueError(f"{pred_path} contains duplicate id values")
    validate_prediction_labels(pred[args.pred_col], pred_path)
    keep_gold = [c for c in ["id", "context", "prompt", "response", args.label_col] if c in gold.columns]
    merged = gold[keep_gold].merge(pred[["id", args.pred_col]], on="id", how="inner")
    checks = []
    checks.append(("merged predictions are non-empty", len(merged) > 0))
    missing_ids = sorted(set(gold["id"]) - set(merged["id"]))
    checks.append(("no missing gold IDs", len(missing_ids) == 0 or args.allow_partial))
    if len(merged) == 0:
        write_validation_report(out, checks)
    if missing_ids and not args.allow_partial:
        write_validation_report(out, checks)
    merged = merged.rename(columns={args.label_col: "label", args.pred_col: "predict_label"})
    merged.to_csv(out / "predictions_merged.csv", index=False)
    wrong = merged[merged["label"] != merged["predict_label"]].copy()
    wrong.to_csv(out / "wrong_predictions.csv", index=False)
    write_error_cases(wrong, out / "selected_error_cases.md")
    summary = compute_and_save(merged["label"], merged["predict_label"], out)
    validate_summary(summary)
    cm = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    checks.append(("confusion matrix is 3x3", tuple(cm.shape) == (3, 3)))
    checks.append(("summary metrics are numeric and complete", True))
    latency_seconds = float(args.latency_seconds) if args.latency_seconds is not None else 0.0
    latency_row = {
        "name": "evidence_from_existing_predictions",
        "total_seconds": latency_seconds,
        "samples": int(len(merged)),
        "seconds_per_sample": float(latency_seconds / len(merged)) if latency_seconds > 0 and len(merged) else None,
        "samples_per_second": float(len(merged) / latency_seconds) if latency_seconds > 0 and len(merged) else None,
        "cuda_max_memory_allocated": None,
        "cuda_max_memory_reserved": None,
    }
    save_latency_summary([latency_row], out)
    required = [
        "predictions_merged.csv",
        "wrong_predictions.csv",
        "selected_error_cases.md",
        "classification_report.csv",
        "classification_report.json",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "summary_metrics.json",
        "latency_summary.csv",
        "latency_summary.json",
    ]
    for name in required:
        p = out / name
        checks.append((f"{p} exists and is non-empty", p.exists() and p.stat().st_size > 0))
    write_validation_report(out, checks)
    print(summary)


if __name__ == "__main__":
    main()
