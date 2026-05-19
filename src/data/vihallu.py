import os
from pathlib import Path

import pandas as pd


LABELS = ["no", "intrinsic", "extrinsic"]
LABEL_SET = set(LABELS)
ENCODINGS = ["utf-8", "utf-8-sig", "cp1258", "cp1252", "latin-1"]
GOLD_COLUMNS = ["id", "context", "prompt", "response", "label"]
PRIVATE_COLUMNS = ["id", "context", "prompt", "response"]
SPLIT_PATHS = {
    "train": ["vihallu-train.csv", "data/vihallu-train.csv", "materials/vihallu-train.csv"],
    "test": ["vihallu-test.csv", "data/vihallu-test.csv", "materials/vihallu-test.csv"],
    "private_test": ["vihallu-private-test.csv", "data/vihallu-private-test.csv", "materials/vihallu-private-test.csv"],
}
PUBLIC_EVAL_SPLITS = {"train", "test"}
LEAKAGE_OVERRIDE_ENV = "ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE"
LEAKAGE_LIMITATION_SENTENCE = "The public ViHallu split used in this challenge-style evaluation contains overlap with the released training split; therefore, we report it as a reproducible challenge-style evaluation rather than an independent holdout estimate."


def first_existing(paths):
    for path in paths:
        p = Path(path)
        if p.exists():
            return p
    raise FileNotFoundError("None of these paths exist: " + ", ".join(map(str, paths)))


def read_csv_robust(path):
    errors = []
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:
            errors.append(f"{enc}: {exc}")
    raise RuntimeError("Could not read CSV " + str(path) + "\n" + "\n".join(errors))


def load_vihallu_split(split):
    if split not in SPLIT_PATHS:
        raise KeyError(f"Unknown split: {split}")
    path = first_existing(SPLIT_PATHS[split])
    df = read_csv_robust(path)
    if split in {"train", "test"}:
        return validate_gold_df(df, path, allow_duplicate_ids=(split == "test"))
    return validate_private_df(df, path)


def infer_split_from_path(path):
    name = Path(path).name
    for split, candidates in SPLIT_PATHS.items():
        if any(Path(candidate).name == name for candidate in candidates):
            return split
    return None


def validate_gold_df(df, path, allow_duplicate_ids=False):
    split = infer_split_from_path(path)
    if split == "private_test":
        raise ValueError(f"{path} is the private test split and must not be used for evidence evaluation because it has no ground-truth label column.")
    validate_columns(df, GOLD_COLUMNS, path)
    validate_required_values(df, GOLD_COLUMNS, path)
    validate_labels(df["label"], path, column="label", require_exact=True)
    validate_unique_ids(df, path, allow_duplicates=allow_duplicate_ids)
    return df


def validate_private_df(df, path):
    validate_columns(df, PRIVATE_COLUMNS, path)
    validate_required_values(df, PRIVATE_COLUMNS, path)
    validate_unique_ids(df, path)
    return df


def validate_columns(df, required, path):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}. Available columns: {list(df.columns)}")
    return True


def csv_row_numbers(indexes, limit=20):
    rows = [int(i) + 2 for i in list(indexes)[:limit]]
    suffix = "" if len(indexes) <= limit else f" and {len(indexes) - limit} more"
    return f"{rows}{suffix}"


def empty_mask(series):
    return series.isna() | series.astype(str).str.strip().eq("")


def normalize_id_series(series):
    return series.astype(str).str.strip()


def make_id_occurrence_key(df, id_col="id"):
    ids = normalize_id_series(df[id_col])
    occurrence = ids.groupby(ids).cumcount().astype(str)
    return ids + "#" + occurrence


def validate_required_values(df, columns, path):
    failures = []
    for col in columns:
        mask = empty_mask(df[col])
        if mask.any():
            failures.append(f"{col}: CSV rows {csv_row_numbers(df.index[mask].tolist())}")
    if failures:
        raise ValueError(f"{path} contains empty required values: " + "; ".join(failures))
    all_empty = df[columns].apply(empty_mask).all(axis=1)
    if all_empty.any():
        raise ValueError(f"{path} contains empty rows at CSV rows {csv_row_numbers(df.index[all_empty].tolist())}")
    return True


def validate_unique_ids(df, path, allow_duplicates=False):
    ids = normalize_id_series(df["id"])
    dup_mask = ids.duplicated(keep=False)
    if dup_mask.any():
        if allow_duplicates:
            return False
        details = []
        for value in sorted(ids[dup_mask].unique()):
            rows = df.index[(ids == value)].tolist()
            details.append(f"{value}: CSV rows {csv_row_numbers(rows)}")
        raise ValueError(f"{path} contains duplicate id values: " + "; ".join(details[:20]))
    return True


def validate_labels(values, path, column="label", require_exact=False):
    series = pd.Series(values)
    missing = empty_mask(series)
    if missing.any():
        raise ValueError(f"{path} contains empty {column} values at CSV rows {csv_row_numbers(series.index[missing].tolist())}")
    normalized = series.astype(str).str.strip()
    bad_mask = ~normalized.isin(LABEL_SET)
    if bad_mask.any():
        details = []
        for idx, value in normalized[bad_mask].head(20).items():
            details.append(f"CSV row {int(idx) + 2}: {value}")
        raise ValueError(f"{path} contains invalid {column} values: " + "; ".join(details))
    observed = set(normalized)
    if require_exact and observed != LABEL_SET:
        missing_labels = sorted(LABEL_SET - observed)
        extra_labels = sorted(observed - LABEL_SET)
        raise ValueError(f"{path} label set must be exactly {LABELS}. Missing: {missing_labels}. Extra: {extra_labels}")
    return True


def validate_prediction_df(df, path, pred_col="predict_label", expected_ids=None, expected_count=None, allow_partial=False, allow_duplicate_ids=False):
    validate_columns(df, ["id", pred_col], path)
    validate_required_values(df, ["id", pred_col], path)
    validate_unique_ids(df, path, allow_duplicates=allow_duplicate_ids)
    validate_labels(df[pred_col], path, column=pred_col, require_exact=False)
    if "row_index" in df.columns:
        validate_required_values(df, ["row_index"], path)
        row_index = normalize_id_series(df["row_index"])
        dup_rows = row_index.duplicated(keep=False)
        if dup_rows.any():
            raise ValueError(f"{path} contains duplicate row_index values at CSV rows {csv_row_numbers(df.index[dup_rows].tolist())}")
    if expected_count is not None and not allow_partial and len(df) != int(expected_count):
        raise ValueError(f"{path} row count mismatch: expected {int(expected_count)}, found {len(df)}")
    if expected_ids is not None:
        expected_series = normalize_id_series(pd.Series(expected_ids))
        actual_series = normalize_id_series(df["id"])
        expected = set(expected_series)
        actual = set(actual_series)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing and not allow_partial:
            raise ValueError(f"{path} is missing prediction IDs: {missing[:20]}{' and more' if len(missing) > 20 else ''}")
        if extra:
            raise ValueError(f"{path} contains prediction IDs not present in gold CSV: {extra[:20]}{' and more' if len(extra) > 20 else ''}")
        if not allow_partial:
            expected_counts = expected_series.value_counts().sort_index()
            actual_counts = actual_series.value_counts().sort_index()
            if not expected_counts.equals(actual_counts):
                mismatched = []
                all_ids = sorted(set(expected_counts.index) | set(actual_counts.index))
                for value in all_ids:
                    expected_value = int(expected_counts.get(value, 0))
                    actual_value = int(actual_counts.get(value, 0))
                    if expected_value != actual_value:
                        mismatched.append(f"{value}: expected {expected_value}, found {actual_value}")
                raise ValueError(f"{path} prediction ID occurrence counts do not match gold CSV: " + "; ".join(mismatched[:20]))
    return True


def non_augmented_test_df(df):
    if "is_augmented" not in df.columns:
        return df
    mask = df["is_augmented"].astype(str).str.strip().str.lower().isin(["false", "0", "no"])
    if mask.any():
        return df.loc[mask].copy()
    return df


def leakage_overlap_report(train_df, test_df):
    train = train_df[["id", "context", "prompt", "response", "label"]].copy()
    test = non_augmented_test_df(test_df)[["id", "context", "prompt", "response", "label"]].copy()
    merged = train.merge(test, on=["id", "context", "prompt", "response", "label"], how="inner")
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test_df)),
        "test_non_augmented_rows": int(len(test)),
        "overlap_rows": int(len(merged)),
        "overlap_ids": int(merged["id"].astype(str).nunique()) if len(merged) else 0,
        "sample_ids": sorted(merged["id"].astype(str).unique()[:10]) if len(merged) else [],
    }


def env_flag_enabled(name):
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def leakage_override_enabled(cli_enabled=False):
    return bool(cli_enabled) or env_flag_enabled(LEAKAGE_OVERRIDE_ENV)


def validate_or_report_public_split_leakage(train_df, test_df, allow_known_public_split_leakage=False, report_path=None):
    leakage = leakage_overlap_report(train_df, test_df)
    override = leakage_override_enabled(allow_known_public_split_leakage)
    leakage["leakage_detected"] = leakage["overlap_rows"] > 0
    leakage["leakage_override"] = bool(override and leakage["leakage_detected"])
    leakage["challenge_style_evaluation"] = bool(override and leakage["leakage_detected"])
    if leakage["leakage_detected"] and not override:
        raise RuntimeError(
            "Detected train/test leakage in ViHallu public splits: "
            f"overlap_rows={leakage['overlap_rows']} overlap_ids={leakage['overlap_ids']} sample_ids={leakage['sample_ids']}. "
            f"Set {LEAKAGE_OVERRIDE_ENV}=1 only when intentionally reporting challenge-style public split evaluation."
        )
    if leakage["leakage_override"] and report_path is not None:
        write_leakage_report(report_path, leakage)
    return leakage


def write_leakage_report(path, leakage, train_path="vihallu-train.csv", test_path="vihallu-test.csv"):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Public Split Leakage Report",
        "",
        f"* train_file: `{train_path}`",
        f"* test_file: `{test_path}`",
        f"* overlap_rows: {leakage['overlap_rows']}",
        f"* overlap_ids: {leakage['overlap_ids']}",
        f"* train_rows: {leakage['train_rows']}",
        f"* test_rows: {leakage['test_rows']}",
        f"* test_non_augmented_rows: {leakage['test_non_augmented_rows']}",
        f"* sample_ids: {', '.join(map(str, leakage['sample_ids']))}",
        "",
        LEAKAGE_LIMITATION_SENTENCE,
        "",
        "leakage_override=true",
        "challenge_style_evaluation=true",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {out}")
    return out
