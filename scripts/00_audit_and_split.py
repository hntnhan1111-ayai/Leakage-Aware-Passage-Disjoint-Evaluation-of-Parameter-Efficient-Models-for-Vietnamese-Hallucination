#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vihallu_repro.audit import (
    nearest_cross_split_contexts,
    overlap_summary,
    quarantine_test_diagnosis,
    summarize_dataset,
    validate_clean_splits,
)
from vihallu_repro.data import add_fingerprints, read_labeled_csv, sha256_file, write_json
from vihallu_repro.split import search_balanced_group_split


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def markdown_summary(original: dict, quarantine: dict, split_report: dict, split_meta: dict) -> str:
    lines = [
        "# Dataset audit and passage-disjoint split",
        "",
        "## P0 findings",
        "",
        f"- Source labeled rows: **{original['train']['rows']}**.",
        f"- Source labeled unique contexts: **{original['train']['unique_contexts']}**.",
        f"- Quarantined legacy test rows: **{quarantine['rows']}**.",
        f"- Quarantined legacy test unique IDs: **{quarantine['unique_ids']}**.",
        f"- Quarantined legacy test unique triplets: **{quarantine['unique_triplets']}**.",
        f"- Every quarantined row exactly matches a labeled training triplet: **{quarantine['invalid_as_independent_test']}**.",
        "- The quarantined file must not be used for model selection or final evaluation.",
        f"- The unlabeled private file shares **{original['unlabeled_private']['overlap_with_labeled_source']['context_hash']}** unique contexts with the labeled source and is retained only for optional blind inference; it is not used for metrics.",
        "",
        "## P1 clean split",
        "",
        f"- Split objective: `{split_meta['objective']:.10f}`.",
        f"- Base seed: `{split_meta['base_seed']}`; selected trial: `{split_meta['selected_trial']}`; effective seed: `{split_meta['effective_seed']}`.",
        "- Group key: SHA-256 of normalized context.",
        "- Exact ID, context, and triplet overlap across train/dev/test: zero.",
        "",
        "| split | rows | unique contexts | no | intrinsic | extrinsic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "dev", "test"]:
        info = split_report["splits"][split]
        counts = info["class_counts"]
        lines.append(
            f"| {split} | {info['rows']} | {info['groups']} | {counts['no']} | {counts['intrinsic']} | {counts['extrinsic']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The clean test split is a ViHallu-derived, passage-disjoint evaluation split constructed from the released labeled data. It is not the official hidden challenge test set and must not be described as an official ViHallu leaderboard result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit legacy datasets and create a clean passage-disjoint split.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--skip-near-duplicate-audit", action="store_true")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)
    raw = config["raw"]
    paths = {name: PROJECT_ROOT / value for name, value in raw.items()}

    train = add_fingerprints(read_labeled_csv(paths["labeled_source"]))
    suspect = add_fingerprints(read_labeled_csv(paths["quarantined_duplicate_test"], allow_duplicate_ids=True))
    private = add_fingerprints(pd.read_csv(paths["unlabeled_private"]))

    reports_dir = PROJECT_ROOT / config["reports_dir"]
    processed_dir = PROJECT_ROOT / config["processed_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    original = {
        "train": summarize_dataset(train, paths["labeled_source"]),
        "quarantined_duplicate_test": summarize_dataset(suspect, paths["quarantined_duplicate_test"]),
        "unlabeled_private": {
            "path": str(paths["unlabeled_private"]),
            "sha256": sha256_file(paths["unlabeled_private"]),
            "rows": int(len(private)),
            "unique_ids": int(private["id"].astype(str).nunique()),
            "unique_contexts": int(private["context_hash"].nunique()),
            "unique_triplets": int(private["triplet_hash"].nunique()),
            "has_gold_labels": bool("label" in private.columns and private["label"].notna().any()),
            "overlap_with_labeled_source": overlap_summary(train, private),
            "rows_with_labeled_source_context": int(private["context_hash"].isin(set(train["context_hash"])).sum()),
        },
    }
    original["train"]["path"] = raw["labeled_source"]
    original["quarantined_duplicate_test"]["path"] = raw["quarantined_duplicate_test"]
    original["unlabeled_private"]["path"] = raw["unlabeled_private"]
    quarantine = quarantine_test_diagnosis(train, suspect)
    if not quarantine["invalid_as_independent_test"]:
        raise RuntimeError("Legacy duplicate test diagnosis changed; inspect before continuing")

    split_cfg = config["split"]
    split_result = search_balanced_group_split(
        train,
        proportions=tuple(float(value) for value in split_cfg["proportions"]),
        base_seed=int(split_cfg["base_seed"]),
        trials=int(split_cfg["search_trials"]),
    )
    output_columns = ["id", "context", "prompt", "response", "label", "group_id", "triplet_hash"]
    splits = {
        name: train.loc[index].sort_values("id").reset_index(drop=True)
        for name, index in split_result.split_indices.items()
    }
    validation = validate_clean_splits(splits)
    if not validation["valid"]:
        raise RuntimeError(f"Generated split failed leakage validation: {validation}")

    for name, frame in splits.items():
        frame[output_columns].to_csv(processed_dir / f"{name}.csv", index=False)

    split_meta = {
        "source_file": raw["labeled_source"],
        "source_sha256": sha256_file(paths["labeled_source"]),
        "group_key": "sha256(normalize_nfkc_lower_whitespace(context))",
        "proportions": split_cfg["proportions"],
        "base_seed": int(split_cfg["base_seed"]),
        "search_trials": int(split_cfg["search_trials"]),
        "selected_trial": split_result.selected_trial,
        "effective_seed": split_result.effective_seed,
        "objective": split_result.objective,
        "row_counts": split_result.row_counts,
        "group_counts": split_result.group_counts,
        "class_counts": split_result.class_counts,
    }

    write_json(reports_dir / "original_dataset_audit.json", original)
    write_json(reports_dir / "quarantined_duplicate_test_diagnosis.json", quarantine)
    write_json(reports_dir / "clean_split_metadata.json", split_meta)
    write_json(reports_dir / "clean_split_validation.json", validation)

    if not args.skip_near_duplicate_audit:
        near_cfg = config.get("near_duplicate", {})
        near_report = nearest_cross_split_contexts(
            splits,
            thresholds=tuple(float(value) for value in near_cfg.get("thresholds", [0.90, 0.95, 0.98])),
            top_k_pairs=int(near_cfg.get("top_k_pairs", 50)),
        )
        write_json(reports_dir / "near_duplicate_context_audit.json", near_report)

    summary = markdown_summary(original, quarantine, validation, split_meta)
    (reports_dir / "AUDIT_SUMMARY.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
