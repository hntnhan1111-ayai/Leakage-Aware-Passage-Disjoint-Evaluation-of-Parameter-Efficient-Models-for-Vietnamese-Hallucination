#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vihallu_repro.stats import mcnemar_exact, paired_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--models", nargs="*", default=["qwen35_peft", "gemma4_peft"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    frames = {}
    for model_key in args.models:
        path = ROOT / config["project"]["output_root"] / model_key / f"seed_{args.seed}" / "test/predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)[["id", "label", "predict_label"]]
        if frame["id"].duplicated().any():
            raise ValueError(f"Duplicate IDs in {path}")
        frames[model_key] = frame

    reference_name = args.models[0]
    merged = frames[reference_name].rename(columns={"predict_label": reference_name})
    for model_key in args.models[1:]:
        merged = merged.merge(
            frames[model_key].rename(columns={"label": f"label_{model_key}", "predict_label": model_key}),
            on="id",
            how="inner",
            validate="one_to_one",
        )
        if not (merged["label"] == merged[f"label_{model_key}"]).all():
            raise ValueError(f"Gold labels do not align for {model_key}")
        merged = merged.drop(columns=[f"label_{model_key}"])
    expected_rows = len(frames[reference_name])
    if len(merged) != expected_rows:
        raise ValueError(f"Prediction alignment lost rows: expected {expected_rows}, got {len(merged)}")

    predictions = {model_key: merged[model_key].to_numpy() for model_key in args.models}
    bootstrap = paired_bootstrap(
        merged["label"].to_numpy(),
        predictions,
        iterations=args.iterations,
        seed=args.bootstrap_seed,
    )
    mcnemar = {
        f"{left}__vs__{right}": mcnemar_exact(merged["label"], merged[left], merged[right])
        for left, right in combinations(args.models, 2)
    }
    payload = {
        "models": args.models,
        "model_seed": args.seed,
        "rows": len(merged),
        "bootstrap": bootstrap,
        "mcnemar": mcnemar,
    }
    output = ROOT / "results/statistics" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "statistical_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    merged.to_csv(output / "aligned_predictions.csv", index=False)

    lines = ["# Statistical validation", "", f"Rows: **{len(merged)}**", "", "## Bootstrap Macro-F1", ""]
    for model_key, values in bootstrap["models"].items():
        lines.append(f"- `{model_key}`: {values['point']:.6f} [{values['ci_low']:.6f}, {values['ci_high']:.6f}]")
    lines.extend(["", "## Paired differences", ""])
    for pair, values in bootstrap["differences"].items():
        lines.append(f"- `{pair}`: {values['point']:.6f} [{values['ci_low']:.6f}, {values['ci_high']:.6f}], bootstrap p={values['two_sided_bootstrap_p']:.6g}")
    lines.extend(["", "## Exact McNemar tests", ""])
    for pair, values in mcnemar.items():
        lines.append(f"- `{pair}`: b={values['a_correct_b_wrong']}, c={values['a_wrong_b_correct']}, p={values['exact_p_value']:.6g}")
    (output / "statistical_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
