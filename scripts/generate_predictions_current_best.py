import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


LABELS = ["no", "intrinsic", "extrinsic"]
CLASS_WEIGHTS = {"intrinsic": 0.3521, "extrinsic": 0.3520, "no": 0.2959}


def normalize_label(text):
    value = str(text).strip().lower()
    first = value.splitlines()[0].strip() if value else ""
    if first in LABELS:
        return first
    for label in LABELS:
        if re.search(r"(^|[^a-z])" + re.escape(label) + r"([^a-z]|$)", value):
            return label
    raise ValueError(f"Could not parse model output as a valid label: {str(text)[:200]}")


def compute_jaccard_similarity(text1, text2, tokenizer):
    tokens1 = set(tokenizer.tokenize(str(text1)))
    tokens2 = set(tokenizer.tokenize(str(text2)))
    union = tokens1.union(tokens2)
    if not union:
        return 0.0
    return len(tokens1.intersection(tokens2)) / len(union)


def create_prompt(row, tokenizer, template_id):
    context = row.get("context", "")
    response = row.get("response", "")
    score = compute_jaccard_similarity(context, response, tokenizer)
    if template_id == 1:
        return f"Phân tích hallucination bằng cách so sánh trực tiếp CONTEXT và RESPONSE.\n\nJaccard score: {score:.4f}\n\nCONTEXT:\n{context}\n\nRESPONSE:\n{response}\n\nPHÂN TÍCH:\n1. RESPONSE có thông tin nào được thêm vào không có trong CONTEXT không?\n2. RESPONSE có mâu thuẫn với CONTEXT không?\n\nKẾT LUẬN (no/intrinsic/extrinsic):"
    if template_id == 2:
        return f"Dựa vào CONTEXT dưới đây, hãy đánh giá xem RESPONSE có chứa hallucination không.\n\nCONTEXT: {context}\nRESPONSE: {response}\n\nCÂU HỎI:\n- Response có thêm thông tin không có trong context không?\n- Response có mâu thuẫn với context không?\n\nTRẢ LỜI (no/intrinsic/extrinsic):"
    return f"So sánh CONTEXT và RESPONSE để phân loại.\n\nCONTEXT: \"{context}\"\nRESPONSE: \"{response}\"\n\nCHỌN MỘT TRONG CÁC LOẠI SAU:\n1. no\n2. intrinsic\n3. extrinsic\n\nPHÂN LOẠI:"


def find_adapter(explicit):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Adapter directory not found: {path}")
        return path
    candidates = []
    for root in [Path("adapters"), Path("checkpoints"), Path("results_scratch_train_pipeline")]:
        if root.exists():
            candidates.extend([p for p in root.rglob("*") if p.is_dir() and ((p / "adapter_config.json").exists() or (p / "config.json").exists())])
    if candidates:
        return sorted(candidates, key=lambda p: len(str(p)))[0]
    raise FileNotFoundError("No adapter/checkpoint found. Provide --adapter_dir on the target machine, set PRED_CSV to an existing prediction CSV, or explicitly run a training workflow before generation.")


def validate_dry_run(args):
    gold_path = Path(args.gold_csv)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold CSV not found: {gold_path}")
    import csv

    encodings = ["utf-8", "utf-8-sig", "cp1258", "cp1252", "latin-1"]
    header = None
    rows = 0
    last_error = None
    for enc in encodings:
        try:
            with gold_path.open("r", encoding=enc, errors="strict", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = sum(1 for _ in reader)
            break
        except Exception as exc:
            last_error = exc
    if header is None:
        raise RuntimeError(f"Could not read {gold_path}: {last_error}")
    missing = [c for c in ["id", "context", "prompt", "response", "label"] if c not in header]
    if missing:
        raise ValueError(f"{gold_path} missing columns {missing}. Available columns: {header}")
    if args.sample_frac is not None and not (0 < args.sample_frac <= 1):
        raise ValueError("--sample_frac must be in (0, 1]")
    adapter_status = "provided" if args.adapter_dir else "auto-detect on target"
    print(f"DRY_RUN_OK gold_rows={rows} out_csv={args.out_csv} adapter={adapter_status}")


def load_model_and_tokenizer(args, adapter_dir):
    import torch
    from peft import PeftModel
    from src.models.loader import load_causal_lm

    model, tokenizer = load_causal_lm(args.model_dir, quantized=not args.no_quant)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, torch


def predict_one(model, tokenizer, torch, row, seed, max_new_tokens):
    votes = []
    device = next(model.parameters()).device
    from src.utils.seed import set_seed

    for template_id in [1, 2, 3]:
        prompt = create_prompt(row, tokenizer, template_id)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        set_seed(seed + template_id)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, pad_token_id=tokenizer.eos_token_id)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            label = normalize_label(raw)
        except ValueError:
            label = "no"
        votes.append((label, raw))
    scores = {label: 0.0 for label in LABELS}
    for label, _ in votes:
        scores[label] += CLASS_WEIGHTS[label]
    best = max(scores, key=scores.get)
    raw_joined = " || ".join(raw for _, raw in votes)
    return best, raw_joined


def run_generation(args):
    import pandas as pd
    from tqdm import tqdm
    from src.data.vihallu import read_csv_robust
    from src.evaluation.latency import Timer, save_latency_summary
    from src.utils.seed import set_seed

    set_seed(args.seed)
    adapter_dir = find_adapter(args.adapter_dir)
    model, tokenizer, torch = load_model_and_tokenizer(args, adapter_dir)
    df = read_csv_robust(args.gold_csv)
    if args.sample_frac is not None:
        df = df.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    rows = []
    with Timer("generate_predictions_current_best", samples=len(df)) as timer:
        for row in tqdm(df.to_dict("records"), total=len(df)):
            pred, raw = predict_one(model, tokenizer, torch, row, args.seed, args.max_new_tokens)
            rows.append({
                "id": row["id"],
                "label": row["label"],
                "predict_label": pred,
                "context": row["context"],
                "prompt": row["prompt"],
                "response": row["response"],
                "raw_output": raw,
            })
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    bad = sorted(set(out_df["predict_label"].astype(str)) - set(LABELS))
    if bad:
        raise ValueError(f"Invalid predict_label values generated: {bad}")
    out_df.to_csv(out_csv, index=False)
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {out_csv}")
    config = {
        "seed": args.seed,
        "model_dir": args.model_dir,
        "adapter_dir": str(adapter_dir),
        "generation": {"do_sample": False, "temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "dataset_path": args.gold_csv,
        "limit": args.limit,
        "sample_frac": args.sample_frac,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": "bfloat16",
        "quantization": None if args.no_quant else "nf4_4bit",
    }
    config_path = Path(args.config_json)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    save_latency_summary([timer.summary()], out_csv.parent)
    print(f"Wrote {out_csv}")
    print(f"Wrote {config_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv", default="vihallu-test.csv")
    parser.add_argument("--out_csv", default="results/predictions.csv")
    parser.add_argument("--config_json", default="results/prediction_config.json")
    parser.add_argument("--model_dir", default="models/Vistral-7B-Chat")
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_frac", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--no_quant", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        validate_dry_run(args)
        return
    run_generation(args)


if __name__ == "__main__":
    main()
