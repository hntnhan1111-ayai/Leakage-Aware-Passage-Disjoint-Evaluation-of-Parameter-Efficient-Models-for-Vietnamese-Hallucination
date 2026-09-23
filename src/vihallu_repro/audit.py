from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from . import LABELS
from .data import class_counts, class_proportions, sha256_file


def summarize_dataset(df: pd.DataFrame, path: str | Path) -> dict[str, object]:
    id_counts = df["id"].astype(str).value_counts()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(df)),
        "unique_ids": int(df["id"].astype(str).nunique()),
        "duplicate_id_rows": int((id_counts[id_counts > 1]).sum()),
        "unique_contexts": int(df["context_hash"].nunique()),
        "unique_prompts": int(df["prompt_hash"].nunique()),
        "unique_responses": int(df["response_hash"].nunique()),
        "unique_triplets": int(df["triplet_hash"].nunique()),
        "exact_duplicate_triplet_rows": int(df["triplet_hash"].duplicated(keep=False).sum()),
        "class_counts": class_counts(df),
        "class_proportions": class_proportions(df),
    }


def overlap_summary(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, int]:
    fields = ["id", "context_hash", "prompt_hash", "response_hash", "triplet_hash"]
    return {
        field: int(len(set(left[field].astype(str)) & set(right[field].astype(str))))
        for field in fields
    }


def quarantine_test_diagnosis(train: pd.DataFrame, suspect_test: pd.DataFrame) -> dict[str, object]:
    required = {"is_augmented", "aug_id"}
    diagnosis: dict[str, object] = {
        "has_augmentation_columns": required.issubset(suspect_test.columns),
        "rows": int(len(suspect_test)),
        "unique_ids": int(suspect_test["id"].astype(str).nunique()),
        "unique_triplets": int(suspect_test["triplet_hash"].nunique()),
        "id_frequency_distribution": {
            str(int(k)): int(v) for k, v in suspect_test.groupby("id").size().value_counts().sort_index().items()
        },
        "train_test_overlap": overlap_summary(train, suspect_test),
    }
    if required.issubset(suspect_test.columns):
        diagnosis["is_augmented_counts"] = {
            str(k): int(v) for k, v in suspect_test["is_augmented"].value_counts(dropna=False).items()
        }
        diagnosis["aug_id_counts"] = {
            str(k): int(v) for k, v in suspect_test["aug_id"].value_counts(dropna=False).sort_index().items()
        }
    merged = suspect_test.merge(
        train[["id", "context_hash", "prompt_hash", "response_hash", "triplet_hash", "label"]],
        on="id",
        suffixes=("_test", "_train"),
        how="left",
    )
    diagnosis["rows_with_train_id"] = int(merged["context_hash_train"].notna().sum())
    diagnosis["identical_to_train_by_field"] = {
        field: int((merged[f"{field}_test"] == merged[f"{field}_train"]).sum())
        for field in ["context_hash", "prompt_hash", "response_hash", "triplet_hash", "label"]
    }
    diagnosis["invalid_as_independent_test"] = bool(
        diagnosis["rows_with_train_id"] == len(suspect_test)
        and diagnosis["identical_to_train_by_field"]["triplet_hash"] == len(suspect_test)
    )
    return diagnosis


def validate_clean_splits(splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    report: dict[str, object] = {"pairwise": {}, "valid": True}
    for left_name, right_name in combinations(splits, 2):
        overlap = overlap_summary(splits[left_name], splits[right_name])
        report["pairwise"][f"{left_name}__{right_name}"] = overlap
        if overlap["context_hash"] or overlap["triplet_hash"] or overlap["id"]:
            report["valid"] = False
    report["splits"] = {
        name: {
            "rows": int(len(df)),
            "groups": int(df["group_id"].nunique()),
            "class_counts": class_counts(df),
            "class_proportions": class_proportions(df),
        }
        for name, df in splits.items()
    }
    return report


def nearest_cross_split_contexts(
    splits: dict[str, pd.DataFrame],
    *,
    thresholds: tuple[float, ...] = (0.90, 0.95, 0.98),
    top_k_pairs: int = 50,
) -> dict[str, object]:
    unique_frames: dict[str, pd.DataFrame] = {}
    all_texts: list[str] = []
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, df in splits.items():
        frame = df.drop_duplicates("group_id")[["group_id", "context_normalized", "context"]].reset_index(drop=True)
        unique_frames[name] = frame
        start = cursor
        all_texts.extend(frame["context_normalized"].tolist())
        cursor += len(frame)
        offsets[name] = (start, cursor)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=1, max_features=100_000, sublinear_tf=True, dtype=np.float32)
    matrix = vectorizer.fit_transform(all_texts)
    report: dict[str, object] = {}
    for left_name, right_name in combinations(splits, 2):
        left_start, left_end = offsets[left_name]
        right_start, right_end = offsets[right_name]
        left_matrix = matrix[left_start:left_end]
        right_matrix = matrix[right_start:right_end]
        nearest = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
        nearest.fit(right_matrix)
        distances, indices = nearest.kneighbors(left_matrix)
        similarities = 1.0 - distances[:, 0]
        left_frame = unique_frames[left_name]
        right_frame = unique_frames[right_name]
        order = np.argsort(-similarities)[:top_k_pairs]
        pairs = []
        for left_idx in order:
            right_idx = int(indices[left_idx, 0])
            pairs.append(
                {
                    "similarity": float(similarities[left_idx]),
                    "left_group_id": str(left_frame.iloc[left_idx]["group_id"]),
                    "right_group_id": str(right_frame.iloc[right_idx]["group_id"]),
                    "left_context_excerpt": str(left_frame.iloc[left_idx]["context"])[:500],
                    "right_context_excerpt": str(right_frame.iloc[right_idx]["context"])[:500],
                }
            )
        report[f"{left_name}__{right_name}"] = {
            "left_unique_contexts": int(len(left_frame)),
            "right_unique_contexts": int(len(right_frame)),
            "max_similarity": float(similarities.max()) if len(similarities) else None,
            "mean_nearest_similarity": float(similarities.mean()) if len(similarities) else None,
            "threshold_counts": {
                str(threshold): int((similarities >= threshold).sum()) for threshold in thresholds
            },
            "top_pairs": pairs,
        }
    return report
