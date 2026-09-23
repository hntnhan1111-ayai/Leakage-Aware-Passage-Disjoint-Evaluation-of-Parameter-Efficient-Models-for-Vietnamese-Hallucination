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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    rows = []
    for model_key, entry in config["models"].items():
        directory = ROOT / config["project"]["output_root"] / model_key / f"seed_{args.seed}" / "test"
        summary_path = directory / "summary_metrics.json"
        report_path = directory / "classification_report.json"
        if not summary_path.exists() or not report_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row = {
            "model_key": model_key,
            "hf_id": entry["hf_id"],
            "seed": args.seed,
            "accuracy": summary["accuracy"],
            "macro_f1": summary["macro_f1"],
            "weighted_f1": summary["weighted_f1"],
        }
        for label in ["no", "intrinsic", "extrinsic"]:
            row[f"{label}_f1"] = report[label]["f1-score"]
            row[f"{label}_precision"] = report[label]["precision"]
            row[f"{label}_recall"] = report[label]["recall"]
            row[f"{label}_support"] = int(report[label]["support"])
        rows.append(row)
    if not rows:
        raise FileNotFoundError("No completed model test artifacts were found")
    frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    output = ROOT / "results/main_results"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"main_results_seed_{args.seed}.csv", index=False)
    lines = ["# Main results", "", frame.to_markdown(index=False, floatfmt=".4f"), ""]
    (output / f"main_results_seed_{args.seed}.md").write_text("\n".join(lines), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
