#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vihallu_repro.peft_runtime import attach_adapter, evaluate_rows, load_base_model, load_tokenizer_or_processor
from vihallu_repro.prompts import ABLATION_CONDITIONS, make_ablation_frame


def model_reference(entry: dict) -> str:
    local = ROOT / entry["local_dir"]
    return str(local) if local.exists() and any(local.iterdir()) else entry["hf_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=["vistral_peft", "qwen35_peft", "gemma4_peft"], default="qwen35_peft")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions", nargs="*", default=ABLATION_CONDITIONS)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    entry = config["models"][args.model_key]
    defaults = config["peft_defaults"]
    trained_dir = ROOT / config["project"]["output_root"] / args.model_key / f"seed_{args.seed}"
    adapter_dir = trained_dir / entry.get("adapter_subdir", "adapter")
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing trained adapter: {adapter_dir}")
    test = pd.read_csv(ROOT / config["project"]["test_csv"])

    model_ref = model_reference(entry)
    tokenizer = load_tokenizer_or_processor(model_ref)
    model = load_base_model(model_ref, quantized=True, torch_dtype="bfloat16")
    model = attach_adapter(model, adapter_dir)

    output_root = ROOT / "results/context_ablation" / args.model_key / f"seed_{args.seed}"
    summary_rows = []
    for condition in args.conditions:
        frame = make_ablation_frame(test, condition, seed=args.seed)
        condition_dir = output_root / condition
        predictions = evaluate_rows(
            model,
            tokenizer,
            frame.to_dict("records"),
            output_dir=condition_dir,
            condition=condition,
            max_prompt_tokens=int(defaults["max_prompt_tokens"]),
            add_zero_mm_token_type_ids=bool(entry.get("add_zero_mm_token_type_ids", False)),
        )
        if condition == "shuffled_context":
            frame[["id", "shuffled_from_id", "original_context", "context"]].to_csv(condition_dir / "shuffle_mapping.csv", index=False)
        metrics = json.loads((condition_dir / "summary_metrics.json").read_text(encoding="utf-8"))
        report = json.loads((condition_dir / "classification_report.json").read_text(encoding="utf-8"))
        summary_rows.append({
            "condition": condition,
            "rows": len(predictions),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "no_f1": report["no"]["f1-score"],
            "intrinsic_f1": report["intrinsic"]["f1-score"],
            "extrinsic_f1": report["extrinsic"]["f1-score"],
        })
    summary = pd.DataFrame(summary_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "context_ablation_summary.csv", index=False)
    (output_root / "context_ablation_summary.md").write_text(
        "# Context-sensitivity ablation\n\n" + summary.to_markdown(index=False, floatfmt=".4f") + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
