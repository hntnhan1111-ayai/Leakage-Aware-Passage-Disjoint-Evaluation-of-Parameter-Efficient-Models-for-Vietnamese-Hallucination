from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import LABELS

REQUIRED_COLUMNS = ["id", "context", "prompt", "response", "label"]
TEXT_COLUMNS = ["context", "prompt", "response"]


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_labeled_csv(path: str | Path, *, allow_duplicate_ids: bool = False) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    for col in REQUIRED_COLUMNS:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{path} contains empty values in required column '{col}'")
    bad_labels = sorted(set(df["label"].astype(str).str.strip()) - set(LABELS))
    if bad_labels:
        raise ValueError(f"{path} contains invalid labels: {bad_labels}")
    if not allow_duplicate_ids and df["id"].astype(str).duplicated().any():
        duplicate_count = int(df["id"].astype(str).duplicated(keep=False).sum())
        raise ValueError(f"{path} contains duplicate IDs in {duplicate_count} rows")
    return df


def add_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TEXT_COLUMNS:
        out[f"{col}_normalized"] = out[col].map(normalize_text)
        out[f"{col}_hash"] = out[f"{col}_normalized"].map(sha256_text)
    out["triplet_hash"] = (
        out["context_normalized"]
        + "\x1f"
        + out["prompt_normalized"]
        + "\x1f"
        + out["response_normalized"]
    ).map(sha256_text)
    out["group_id"] = out["context_hash"]
    return out


def write_json(path: str | Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    return {label: int(counts[label]) for label in LABELS}


def class_proportions(df: pd.DataFrame) -> dict[str, float]:
    total = max(1, len(df))
    counts = class_counts(df)
    return {label: counts[label] / total for label in LABELS}


def assert_pairwise_disjoint(values_by_split: dict[str, Iterable[str]], name: str) -> None:
    split_names = list(values_by_split)
    sets = {key: set(values_by_split[key]) for key in split_names}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise RuntimeError(f"{name} overlap between {left} and {right}: {len(overlap)}")
