from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import LABELS


@dataclass(frozen=True)
class SplitSearchResult:
    split_indices: dict[str, np.ndarray]
    selected_trial: int
    effective_seed: int
    objective: float
    row_counts: dict[str, int]
    group_counts: dict[str, int]
    class_counts: dict[str, dict[str, int]]


def _group_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("group_id", sort=True)
    summary = grouped.size().rename("rows").to_frame()
    for label in LABELS:
        summary[label] = grouped["label"].apply(lambda values, label=label: int((values == label).sum()))
    return summary.reset_index()


def search_balanced_group_split(
    df: pd.DataFrame,
    *,
    proportions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    base_seed: int = 42,
    trials: int = 20_000,
) -> SplitSearchResult:
    if abs(sum(proportions) - 1.0) > 1e-9:
        raise ValueError(f"Split proportions must sum to 1, got {proportions}")
    if "group_id" not in df.columns:
        raise ValueError("DataFrame must include group_id")

    summary = _group_summary(df)
    matrix = summary[["rows", *LABELS]].to_numpy(dtype=np.int64)
    total_rows = len(df)
    target_rows = np.asarray(proportions, dtype=np.float64) * total_rows
    overall_class_prop = (
        df["label"].value_counts(normalize=True).reindex(LABELS, fill_value=0).to_numpy(dtype=np.float64)
    )

    best: tuple[float, int, list[np.ndarray], np.ndarray, np.ndarray] | None = None
    for trial in range(int(trials)):
        rng = np.random.default_rng(base_seed + trial)
        permutation = rng.permutation(len(summary))
        cumulative_rows = np.cumsum(matrix[permutation, 0])
        first_cut = int(np.argmin(np.abs(cumulative_rows - target_rows[0]))) + 1
        second_target = target_rows[0] + target_rows[1]
        second_cut = int(np.argmin(np.abs(cumulative_rows - second_target))) + 1
        if second_cut <= first_cut or second_cut >= len(permutation):
            continue

        parts = [
            permutation[:first_cut],
            permutation[first_cut:second_cut],
            permutation[second_cut:],
        ]
        row_counts = np.asarray([matrix[part, 0].sum() for part in parts], dtype=np.float64)
        class_props = np.asarray(
            [matrix[part, 1:].sum(axis=0) / max(1.0, matrix[part, 0].sum()) for part in parts]
        )
        row_error = np.mean(((row_counts - target_rows) / target_rows) ** 2)
        class_error = np.mean(((class_props - overall_class_prop) / (overall_class_prop + 1e-12)) ** 2)
        objective = float(10.0 * row_error + class_error)
        if best is None or objective < best[0]:
            best = (objective, trial, parts, row_counts, class_props)

    if best is None:
        raise RuntimeError("Could not construct a valid group-disjoint split")

    objective, selected_trial, parts, _, _ = best
    split_names = ["train", "dev", "test"]
    group_sets = {
        name: set(summary.iloc[part]["group_id"].astype(str)) for name, part in zip(split_names, parts)
    }
    indices = {
        name: df.index[df["group_id"].astype(str).isin(group_sets[name])].to_numpy(dtype=np.int64)
        for name in split_names
    }
    class_counts = {
        name: {
            label: int((df.loc[index, "label"] == label).sum())
            for label in LABELS
        }
        for name, index in indices.items()
    }
    return SplitSearchResult(
        split_indices=indices,
        selected_trial=int(selected_trial),
        effective_seed=int(base_seed + selected_trial),
        objective=float(objective),
        row_counts={name: int(len(index)) for name, index in indices.items()},
        group_counts={name: int(df.loc[index, "group_id"].nunique()) for name, index in indices.items()},
        class_counts=class_counts,
    )
