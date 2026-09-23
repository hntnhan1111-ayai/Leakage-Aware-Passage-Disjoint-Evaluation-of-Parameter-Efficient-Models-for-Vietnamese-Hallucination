#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vihallu_repro.error_analysis import suggest_category


def stratified_sample(errors: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if len(errors) <= count:
        return errors.copy()
    groups = list(errors.groupby(["label", "predict_label"], sort=True))
    per_group = max(1, count // max(1, len(groups)))
    sampled = []
    used = set()
    for _, group in groups:
        take = min(per_group, len(group))
        selected = group.sample(n=take, random_state=seed)
        sampled.append(selected)
        used.update(selected.index.tolist())
    combined = pd.concat(sampled) if sampled else errors.iloc[0:0]
    remaining = count - len(combined)
    if remaining > 0:
        pool = errors.drop(index=list(used))
        if len(pool):
            combined = pd.concat([combined, pool.sample(n=min(remaining, len(pool)), random_state=seed + 1)])
    return combined.head(count).sort_values(["label", "predict_label", "id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--model-key", default="qwen35_peft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    path = ROOT / config["project"]["output_root"] / args.model_key / f"seed_{args.seed}" / "test/predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    predictions = pd.read_csv(path)
    errors = predictions[predictions["label"] != predictions["predict_label"]].copy()
    errors["error_pair"] = errors["label"].astype(str) + "→" + errors["predict_label"].astype(str)
    errors["suggested_category"] = errors.apply(suggest_category, axis=1)
    review = stratified_sample(errors, args.count, args.sample_seed)
    review["human_category"] = ""
    review["review_notes"] = ""
    review["label_quality_flag"] = ""

    output = ROOT / "results/error_analysis" / args.model_key / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    review.to_csv(output / "error_review_queue.csv", index=False)
    errors.groupby(["error_pair", "suggested_category"]).size().rename("count").reset_index().sort_values("count", ascending=False).to_csv(output / "suggested_error_category_counts.csv", index=False)

    lines = [
        "# Qualitative error review queue",
        "",
        f"Model: `{args.model_key}`",
        f"Seed: `{args.seed}`",
        f"Total errors: **{len(errors)}**",
        f"Sampled for manual review: **{len(review)}**",
        "",
        "The `suggested_category` column is heuristic. Before reporting categories in the paper, manually fill `human_category`, `review_notes`, and `label_quality_flag` for every selected row.",
        "",
        errors.groupby("error_pair").size().rename("count").reset_index().to_markdown(index=False),
        "",
    ]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
