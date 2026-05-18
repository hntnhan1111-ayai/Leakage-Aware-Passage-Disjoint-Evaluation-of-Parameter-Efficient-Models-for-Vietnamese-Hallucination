import argparse
from pathlib import Path
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from tqdm import tqdm

from src.data.vihallu import LABELS, read_csv_robust
from src.evaluation.latency import Timer, save_latency_summary
from src.models.loader import load_causal_lm
from src.utils.seed import set_seed


MODEL_DIR = "models/Qwen3-4B-Instruct-2507"


def build_prompt(row):
    return (
        "Phan loai cau tra loi tieng Viet ve ao giac thanh dung mot trong ba nhan: no, intrinsic, extrinsic.\n"
        "Chi tra ve dung mot nhan.\n"
        f"Context: {row.get('context', '')}\n"
        f"Prompt: {row.get('prompt', '')}\n"
        f"Response: {row.get('response', '')}\n"
        "Label:"
    )


def normalize_label(text):
    value = str(text).strip().lower()
    first = value.splitlines()[0].strip() if value else ""
    if first in LABELS:
        return first
    for label in LABELS:
        if re_match_label(label, value):
            return label
    raise ValueError(f"Could not parse model output as a valid label: {text[:200]}")


def re_match_label(label, value):
    import re

    return re.search(r"(^|[^a-z])" + re.escape(label) + r"([^a-z]|$)", value) is not None


def encode_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        except Exception:
            pass
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)


def main():
    if os.getenv("RUN_QWEN3_BASELINE") != "1":
        raise SystemExit("RUN_QWEN3_BASELINE=1 is required to run this optional baseline.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv", default="vihallu-test.csv")
    parser.add_argument("--out_dir", default="results/qwen3_prompt_baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_csv_robust(args.gold_csv)
    required = ["id", "context", "prompt", "response", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{args.gold_csv} missing required columns: {missing}. Available columns: {list(df.columns)}")
    model, tokenizer = load_causal_lm(MODEL_DIR, quantized=True)
    device = next(model.parameters()).device
    rows = []
    with Timer("qwen3_prompt_baseline", samples=len(df)) as timer:
        for row in tqdm(df.to_dict("records"), total=len(df)):
            prompt = build_prompt(row)
            inputs = encode_prompt(tokenizer, prompt)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=10, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
            generated = output[0][inputs["input_ids"].shape[1]:]
            raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
            pred = normalize_label(raw)
            rows.append({
                "id": row["id"],
                "label": row["label"],
                "predict_label": pred,
                "context": row["context"],
                "prompt": row["prompt"],
                "response": row["response"],
                "raw_output": raw,
            })
    pred_path = out / "predictions.csv"
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    save_latency_summary([timer.summary()], out)
    subprocess.run([
        sys.executable,
        "scripts/build_paper_evidence.py",
        "--gold_csv",
        args.gold_csv,
        "--pred_csv",
        str(pred_path),
        "--out_dir",
        str(out),
        "--label_col",
        "label",
        "--pred_col",
        "predict_label",
        "--seed",
        str(args.seed),
        "--latency_seconds",
        str(timer.total_seconds or 0.0),
    ], check=True)


if __name__ == "__main__":
    main()
