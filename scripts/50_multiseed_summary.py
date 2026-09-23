#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--models", nargs="*", default=["qwen35_peft", "gemma4_peft"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    rows = []
    for model_key in args.models:
        for seed in args.seeds:
            directory = ROOT / config["project"]["output_root"] / model_key / f"seed_{seed}" / "test"
            summary_path = directory / "summary_metrics.json"
            report_path = directory / "classification_report.json"
            if not summary_path.exists() or not report_path.exists():
                raise FileNotFoundError(f"Missing seed result: {model_key} seed={seed}: {directory}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows.append({
                "model_key": model_key,
                "seed": seed,
                "accuracy": summary["accuracy"],
                "macro_f1": summary["macro_f1"],
                "weighted_f1": summary["weighted_f1"],
                "no_f1": report["no"]["f1-score"],
                "intrinsic_f1": report["intrinsic"]["f1-score"],
                "extrinsic_f1": report["extrinsic"]["f1-score"],
            })
    raw = pd.DataFrame(rows)
    metrics = ["accuracy", "macro_f1", "weighted_f1", "no_f1", "intrinsic_f1", "extrinsic_f1"]
    summary_rows = []
    for model_key, group in raw.groupby("model_key"):
        row = {"model_key": model_key, "seeds": ",".join(map(str, sorted(group["seed"].tolist())))}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("macro_f1_mean", ascending=False)
    output = ROOT / "results/multiseed"
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "seed_level_metrics.csv", index=False)
    summary.to_csv(output / "mean_std_summary.csv", index=False)
    (output / "mean_std_summary.md").write_text("# Three-seed stability\n\n" + summary.to_markdown(index=False, floatfmt=".6f") + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
