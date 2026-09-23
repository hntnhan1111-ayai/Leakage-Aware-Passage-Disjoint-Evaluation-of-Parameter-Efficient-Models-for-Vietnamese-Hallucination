#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def latex_escape(value: str) -> str:
    return str(value).replace("_", "\\_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = ROOT / "paper/generated"
    output.mkdir(parents=True, exist_ok=True)
    generated = []

    main_path = ROOT / "results/main_results" / f"main_results_seed_{args.seed}.csv"
    if main_path.exists():
        frame = pd.read_csv(main_path)
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Performance on the passage-disjoint ViHallu-derived test split.}",
            "\\label{tab:clean-main-results}",
            "\\begin{tabular}{lccccc}",
            "\\hline",
            "Model & Accuracy & Macro-F1 & $F1_{no}$ & $F1_{int}$ & $F1_{ext}$ \\\\",
            "\\hline",
        ]
        for _, row in frame.iterrows():
            lines.append(
                f"{latex_escape(row['model_key'])} & {row['accuracy']:.4f} & {row['macro_f1']:.4f} & {row['no_f1']:.4f} & {row['intrinsic_f1']:.4f} & {row['extrinsic_f1']:.4f} \\\\"
            )
        lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
        path = output / "main_results_table.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        generated.append(str(path.relative_to(ROOT)))

    ablation_candidates = list((ROOT / "results/context_ablation").glob(f"*/seed_{args.seed}/context_ablation_summary.csv"))
    for ablation_path in ablation_candidates:
        frame = pd.read_csv(ablation_path)
        model_key = ablation_path.parents[1].name
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Context-sensitivity ablation for {latex_escape(model_key)}.}}",
            "\\label{tab:context-ablation}",
            "\\begin{tabular}{lccccc}",
            "\\hline",
            "Input condition & Accuracy & Macro-F1 & $F1_{no}$ & $F1_{int}$ & $F1_{ext}$ \\\\",
            "\\hline",
        ]
        for _, row in frame.iterrows():
            lines.append(
                f"{latex_escape(row['condition'])} & {row['accuracy']:.4f} & {row['macro_f1']:.4f} & {row['no_f1']:.4f} & {row['intrinsic_f1']:.4f} & {row['extrinsic_f1']:.4f} \\\\"
            )
        lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
        path = output / f"context_ablation_{model_key}.tex"
        path.write_text("\n".join(lines), encoding="utf-8")
        generated.append(str(path.relative_to(ROOT)))

    stat_path = ROOT / "results/statistics" / f"seed_{args.seed}/statistical_validation.json"
    if stat_path.exists():
        stats = json.loads(stat_path.read_text(encoding="utf-8"))
        (output / "statistical_results.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        generated.append(str((output / "statistical_results.json").relative_to(ROOT)))

    manifest = {"seed": args.seed, "generated": generated}
    (output / "generated_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
