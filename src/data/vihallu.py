from pathlib import Path

import pandas as pd


LABELS = ["no", "intrinsic", "extrinsic"]
ENCODINGS = ["utf-8", "utf-8-sig", "cp1258", "cp1252", "latin-1"]
SPLIT_PATHS = {
    "train": ["vihallu-train.csv", "data/vihallu-train.csv", "materials/vihallu-train.csv"],
    "test": ["vihallu-test.csv", "data/vihallu-test.csv", "materials/vihallu-test.csv"],
    "private_test": ["vihallu-private-test.csv", "data/vihallu-private-test.csv", "materials/vihallu-private-test.csv"],
}


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
        return validate_gold_df(df, path)
    return validate_private_df(df, path)


def validate_gold_df(df, path):
    required = ["id", "context", "prompt", "response", "label"]
    validate_columns(df, required, path)
    validate_labels(df["label"], path)
    if df["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate id values")
    return df


def validate_private_df(df, path):
    required = ["id", "context", "prompt", "response"]
    validate_columns(df, required, path)
    if df["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate id values")
    return df


def validate_columns(df, required, path):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}. Available columns: {list(df.columns)}")
    return True


def validate_labels(values, path):
    bad = sorted(set(pd.Series(values).dropna().astype(str)) - set(LABELS))
    if bad:
        raise ValueError(f"{path} contains invalid labels: {bad}")
    return True
