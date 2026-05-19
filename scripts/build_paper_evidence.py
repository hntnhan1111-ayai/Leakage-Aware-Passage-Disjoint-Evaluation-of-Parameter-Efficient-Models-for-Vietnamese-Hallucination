import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


LABELS = ["no", "intrinsic", "extrinsic"]
MALFORMED_COLUMNS = [
    "row_index",
    "id",
    "label",
    "predict_label",
    "malformed_reason",
    "empty_vote_count",
    "unparsable_vote_count",
    "raw_output",
    "template_1_label",
    "template_1_raw_output",
    "template_2_label",
    "template_2_raw_output",
    "template_3_label",
    "template_3_raw_output",
]


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


def load_malformed_info(path, total_rows):
    if not path:
        return {"path": None, "rows": 0, "percentage": 0.0}
    malformed_path = Path(path)
    if not malformed_path.exists():
        import pandas as pd

        malformed_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=MALFORMED_COLUMNS).to_csv(malformed_path, index=False)
        return {"path": str(malformed_path), "rows": 0, "percentage": 0.0}
    import pandas as pd

    malformed = pd.read_csv(malformed_path)
    rows = int(len(malformed))
    percentage = (rows / total_rows) if total_rows else 0.0
    return {"path": str(malformed_path), "rows": rows, "percentage": percentage}


def write_validation_report(out, checks, leakage_info=None):
    lines = ["# Evidence Validation Report", ""]
    failed = [name for name, ok in checks if not ok]
    if leakage_info:
        lines.append(f"leakage_override={str(leakage_info.get('leakage_override', False)).lower()}")
        lines.append(f"challenge_style_evaluation={str(leakage_info.get('challenge_style_evaluation', False)).lower()}")
        lines.append("")
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
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--malformed_csv", default=None)
    parser.add_argument("--allow_known_public_split_leakage", action="store_true")
    parser.add_argument("--allow_malformed_debug", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    from src.data.vihallu import load_vihallu_split, make_id_occurrence_key, normalize_id_series, read_csv_robust, validate_gold_df, validate_or_report_public_split_leakage, validate_prediction_df
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
    gold_has_duplicate_ids = normalize_id_series(gold["id"]).duplicated(keep=False).any() if "id" in gold.columns else False
    validate_gold_df(gold, gold_path, allow_duplicate_ids=True)
    validate_columns(gold, ["id", "context", "prompt", "response", args.label_col], gold_path)
    validate_prediction_df(pred, pred_path, pred_col=args.pred_col, expected_ids=gold["id"], expected_count=len(gold), allow_partial=args.allow_partial, allow_duplicate_ids=gold_has_duplicate_ids)
    validate_prediction_labels(pred[args.pred_col], pred_path)
    leakage_info = validate_or_report_public_split_leakage(
        load_vihallu_split("train"),
        load_vihallu_split("test"),
        allow_known_public_split_leakage=args.allow_known_public_split_leakage,
        report_path=out / "leakage_report.md",
    )
    malformed_csv = args.malformed_csv or str(out / "malformed_predictions.csv")
    malformed_info = load_malformed_info(malformed_csv, len(gold))
    if malformed_info["rows"] > 0 and not args.allow_malformed_debug:
        raise RuntimeError(
            f"Malformed predictions detected before metrics computation: rows={malformed_info['rows']} "
            f"percentage={malformed_info['percentage']:.6f} path={malformed_info['path']}"
        )
    if "is_malformed" in pred.columns:
        malformed_mask = pred["is_malformed"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
        malformed_count = int(malformed_mask.sum())
        if malformed_count > 0 and not args.allow_malformed_debug:
            malformed_pct = malformed_count / len(pred) if len(pred) else 0.0
            raise RuntimeError(
                f"{pred_path} contains malformed predictions before metrics computation: "
                f"rows={malformed_count} percentage={malformed_pct:.6f}"
            )
    keep_gold = [c for c in ["id", "context", "prompt", "response", args.label_col] if c in gold.columns]
    gold_for_merge = gold[keep_gold].copy()
    pred_for_merge = pred[["id", args.pred_col]].copy()
    if "row_index" in pred.columns:
        pred_for_merge["row_index"] = pred["row_index"]
        gold_for_merge["__contract_id"] = gold_for_merge.index.astype(str)
        pred_for_merge["__contract_id"] = normalize_id_series(pred_for_merge["row_index"])
    else:
        gold_for_merge["__contract_id"] = make_id_occurrence_key(gold_for_merge)
        pred_for_merge["__contract_id"] = make_id_occurrence_key(pred_for_merge)
    merged = gold_for_merge.merge(pred_for_merge[["__contract_id", args.pred_col]], on="__contract_id", how="inner").drop(columns=["__contract_id"])
    checks = []
    checks.append(("merged predictions are non-empty", len(merged) > 0))
    missing_keys = sorted(set(gold_for_merge["__contract_id"]) - set(pred_for_merge["__contract_id"]))
    checks.append(("no missing gold rows", len(missing_keys) == 0 or args.allow_partial))
    checks.append((f"malformed predictions count is zero ({malformed_info['rows']} rows, {malformed_info['percentage']:.6%})", malformed_info["rows"] == 0))
    checks.append((f"challenge-style leakage override is {str(leakage_info['leakage_override']).lower()}", True))
    if args.validate_only:
        if len(merged) == 0:
            raise RuntimeError("Prediction contract validation failed: merged predictions are empty")
        if missing_keys and not args.allow_partial:
            raise RuntimeError("Prediction contract validation failed: missing gold rows")
        if malformed_info["rows"] > 0:
            raise RuntimeError("Prediction contract validation failed: malformed predictions detected")
        print(f"VALIDATE_ONLY_OK gold_rows={len(gold)} pred_rows={len(pred)} merged_rows={len(merged)} pred_csv={pred_path}")
        return
    if len(merged) == 0:
        write_validation_report(out, checks, leakage_info=leakage_info)
    if missing_keys and not args.allow_partial:
        write_validation_report(out, checks, leakage_info=leakage_info)
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
    existing_latency = []
    latency_json = out / "latency_summary.json"
    if latency_json.exists():
        loaded = json.loads(latency_json.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            existing_latency = loaded
    save_latency_summary(existing_latency + [latency_row], out)
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
        "malformed_predictions.csv",
    ]
    if leakage_info["leakage_override"]:
        required.append("leakage_report.md")
    for name in required:
        p = out / name
        checks.append((f"{p} exists and is non-empty", p.exists() and p.stat().st_size > 0))
    write_validation_report(out, checks, leakage_info=leakage_info)
    print(summary)


if __name__ == "__main__":
    main()
