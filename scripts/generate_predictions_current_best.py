import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


LABELS = ["no", "intrinsic", "extrinsic"]
CLASS_WEIGHTS = {"intrinsic": 0.3521, "extrinsic": 0.3520, "no": 0.2959}
TEMPLATE_IDS = [1, 2, 3]
MALFORMED_COLUMNS = [
    "row_index",
    "id",
    "label",
    "predict_label",
    "malformed_reason",
    "empty_vote_count",
    "unparsable_vote_count",
    "raw_output",
    "template_1_label",
    "template_1_raw_output",
    "template_2_label",
    "template_2_raw_output",
    "template_3_label",
    "template_3_raw_output",
]


def git_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def flag_enabled(name):
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_label(text):
    value = str(text).strip().lower()
    first = value.splitlines()[0].strip() if value else ""
    if first in LABELS:
        return first
    for label in LABELS:
        if re.search(r"(^|[^a-z])" + re.escape(label) + r"([^a-z]|$)", value):
            return label
    raise ValueError(f"Could not parse model output as a valid label: {str(text)[:200]}")


def validate_generation_settings(args):
    if args.sample_frac is not None and not (0 < args.sample_frac <= 1):
        raise ValueError("--sample_frac must be in (0, 1]")
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be > 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if not (0 < args.top_p <= 1):
        raise ValueError("--top_p must be in (0, 1]")
    if args.max_malformed_count < 0:
        raise ValueError("--max_malformed_count must be >= 0")
    if not (0 <= args.max_malformed_rate <= 1):
        raise ValueError("--max_malformed_rate must be in [0, 1]")


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


def resolve_contract(args, require_model_dir=False, require_adapter=False, allow_missing_adapter=False):
    import preflight_target_run as preflight

    manifest = preflight.verify_environment.load_manifest(args.manifest)
    model_key, model_entry = preflight.find_manifest_model(manifest, args.model_key, args.model_dir)
    model_info = None
    model_dir = args.full_model_dir or args.model_dir
    if require_model_dir or Path(model_dir).exists():
        model_info = preflight.validate_model_dir(model_dir, model_entry)
    adapter_dir = None
    adapter_info = None
    if args.full_model_dir:
        return manifest, model_key, model_entry, model_info, adapter_dir, adapter_info
    if require_adapter:
        adapter_dir = preflight.find_adapter_candidate(manifest, args.adapter_dir)
        adapter_info = preflight.validate_adapter_dir(adapter_dir, model_entry, args.model_dir)
    elif args.adapter_dir:
        adapter_dir = Path(args.adapter_dir)
        if adapter_dir.exists():
            adapter_info = preflight.validate_adapter_dir(adapter_dir, model_entry, args.model_dir)
        elif not allow_missing_adapter:
            raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    return manifest, model_key, model_entry, model_info, adapter_dir, adapter_info


def ensure_malformed_file(path):
    malformed_path = Path(path)
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    with malformed_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MALFORMED_COLUMNS)
        writer.writeheader()
    return malformed_path


def write_malformed_rows(path, rows):
    malformed_path = ensure_malformed_file(path)
    if not rows:
        return malformed_path
    with malformed_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MALFORMED_COLUMNS)
        for row in rows:
            writer.writerow({key: row.get(key) for key in MALFORMED_COLUMNS})
    return malformed_path


def build_generation_config(args, adapter_dir, model_key, status, processed_rows=0, total_rows=0, malformed_count=0, malformed_percentage=0.0):
    leakage_override = bool(args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE"))
    return {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "seed": args.seed,
        "manifest": args.manifest,
        "model_key": model_key,
        "model_id": getattr(args, "model_id", None),
        "model_dir": args.model_dir,
        "full_model_dir": args.full_model_dir,
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "dataset_path": args.gold_csv,
        "out_csv": args.out_csv,
        "malformed_csv": args.malformed_csv,
        "processed_rows": int(processed_rows),
        "total_rows": int(total_rows),
        "malformed_count": int(malformed_count),
        "malformed_percentage": float(malformed_percentage),
        "limit": args.limit,
        "sample_frac": args.sample_frac,
        "generation": {
            "do_sample": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "dtype": "bfloat16",
        "quantization": None if args.no_quant else "nf4_4bit",
        "leakage_override": leakage_override,
        "challenge_style_evaluation": leakage_override,
        "malformed_policy": {
            "max_malformed_count": args.max_malformed_count,
            "max_malformed_rate": args.max_malformed_rate,
        },
    }


def write_generation_config(path, config):
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if not config_path.exists() or config_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {config_path}")
    return config_path


def apply_manifest_defaults(args):
    import verify_environment

    manifest = verify_environment.load_manifest(args.manifest)
    model_key = args.model_key or manifest.get("model_key") or manifest["runtime"]["inference_model_key"]
    model_entry = manifest["models"][model_key]
    artifacts = manifest.get("artifacts", {})
    generation = manifest["generation"]
    args.model_key = model_key
    if args.model_dir is None:
        args.model_dir = manifest.get("model_dir") or model_entry["local_dir"]
    if args.adapter_dir is None and args.full_model_dir is None:
        args.adapter_dir = manifest.get("adapter_dir") or manifest["lora"].get("adapter_dir")
    if args.max_new_tokens is None:
        args.max_new_tokens = int(generation["max_new_tokens"])
    if args.temperature is None:
        args.temperature = float(generation["temperature"])
    if args.top_p is None:
        args.top_p = float(generation["top_p"])
    if args.max_malformed_count is None:
        args.max_malformed_count = int(generation.get("max_malformed_count", 0))
    if args.max_malformed_rate is None:
        args.max_malformed_rate = float(generation.get("max_malformed_rate", 0.0))
    if args.config_json is None:
        args.config_json = artifacts.get("prediction_config_json", "results/prediction_config.json")
    if args.malformed_csv is None:
        args.malformed_csv = artifacts.get("malformed_predictions_csv", "results/paper_evidence/malformed_predictions.csv")
    if args.latency_out_dir is None:
        args.latency_out_dir = artifacts.get("evidence_dir", "results/paper_evidence")
    args.model_id = model_entry["hf_id"]
    return manifest


def validate_manifest_contract(manifest, manifest_path):
    if manifest.get("labels") != LABELS:
        raise ValueError(f"{manifest_path} labels must be exactly {LABELS}, found {manifest.get('labels')}")
    generation = manifest.get("generation", {})
    for key in ["do_sample", "temperature", "top_p", "max_new_tokens"]:
        if key not in generation:
            raise ValueError(f"{manifest_path} generation.{key} is required")
    if generation.get("do_sample") is not False:
        raise ValueError(f"{manifest_path} generation.do_sample must be false for deterministic prediction")
    return True


def ensure_parent_dir(path, label):
    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.exists() or not parent.is_dir():
        raise RuntimeError(f"{label} parent directory is not writable: {parent}")
    return parent


def validate_dry_run(args):
    gold_path = Path(args.gold_csv)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold CSV not found: {gold_path}")
    from src.data.vihallu import read_csv_robust, validate_gold_df

    manifest = apply_manifest_defaults(args)
    validate_manifest_contract(manifest, args.manifest)
    validate_generation_settings(args)
    df = read_csv_robust(gold_path)
    validate_gold_df(df, gold_path, allow_duplicate_ids=True)
    ensure_parent_dir(args.out_csv, "Prediction CSV")
    ensure_parent_dir(args.config_json, "Prediction config")
    ensure_parent_dir(args.malformed_csv, "Malformed prediction CSV")
    ensure_parent_dir(Path(args.latency_out_dir) / "latency_summary.csv", "Latency summary")
    if args.full_model_dir and args.adapter_dir:
        raise ValueError("--full_model_dir and --adapter_dir are mutually exclusive")
    summary = {
        "status": "DRY_RUN_OK",
        "manifest": args.manifest,
        "gold_csv": str(gold_path),
        "gold_rows": int(len(df)),
        "out_csv": args.out_csv,
        "config_json": args.config_json,
        "malformed_csv": args.malformed_csv,
        "latency_out_dir": args.latency_out_dir,
        "model_key": args.model_key,
        "model_id": args.model_id,
        "model_dir": args.model_dir,
        "adapter_dir": args.adapter_dir,
        "full_model_dir": args.full_model_dir,
        "adapter_validation": "skipped_in_dry_run",
        "model_loading": "skipped_in_dry_run",
        "generation": {
            "do_sample": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "labels": LABELS,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def log_runtime_config(config):
    print("RUNTIME_CONFIG " + json.dumps(config, ensure_ascii=False, sort_keys=True))


def load_model_and_tokenizer(args, adapter_dir):
    from peft import PeftModel
    from src.models.loader import load_causal_lm

    model_dir = args.full_model_dir or args.model_dir
    model, tokenizer = load_causal_lm(model_dir, quantized=not args.no_quant)
    if args.full_model_dir is None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    import torch

    return model, tokenizer, torch


def predict_one(model, tokenizer, torch, row, args):
    votes = []
    device = next(model.parameters()).device
    from src.utils.seed import set_seed

    for template_id in TEMPLATE_IDS:
        prompt = create_prompt(row, tokenizer, template_id)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        set_seed(args.seed + template_id)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not raw:
            votes.append({"template_id": template_id, "label": None, "raw_output": raw, "malformed_reason": "empty_output"})
            continue
        try:
            label = normalize_label(raw)
            votes.append({"template_id": template_id, "label": label, "raw_output": raw, "malformed_reason": None})
        except ValueError:
            votes.append({"template_id": template_id, "label": None, "raw_output": raw, "malformed_reason": "unparsable_output"})
    malformed_votes = [vote for vote in votes if vote["malformed_reason"]]
    raw_joined = " || ".join(vote["raw_output"] for vote in votes)
    if malformed_votes:
        reasons = [f"template_{vote['template_id']}:{vote['malformed_reason']}" for vote in malformed_votes]
        valid_votes = [vote["label"] for vote in votes if vote["label"] in LABELS]
        fallback_label = valid_votes[0] if valid_votes else None
        return {
            "predict_label": fallback_label,
            "raw_output": raw_joined,
            "votes": votes,
            "malformed_reason": "; ".join(reasons),
            "empty_vote_count": sum(1 for vote in votes if vote["malformed_reason"] == "empty_output"),
            "unparsable_vote_count": sum(1 for vote in votes if vote["malformed_reason"] == "unparsable_output"),
        }
    scores = {label: 0.0 for label in LABELS}
    for vote in votes:
        scores[vote["label"]] += CLASS_WEIGHTS[vote["label"]]
    best = max(scores, key=scores.get)
    return {
        "predict_label": best,
        "raw_output": raw_joined,
        "votes": votes,
        "malformed_reason": None,
        "empty_vote_count": 0,
        "unparsable_vote_count": 0,
    }


def malformed_row_payload(row_index, row, result):
    payload = {
        "row_index": row_index,
        "id": row["id"],
        "label": row["label"],
        "predict_label": result["predict_label"],
        "malformed_reason": result["malformed_reason"],
        "empty_vote_count": result["empty_vote_count"],
        "unparsable_vote_count": result["unparsable_vote_count"],
        "raw_output": result["raw_output"],
    }
    for vote in result["votes"]:
        payload[f"template_{vote['template_id']}_label"] = vote["label"]
        payload[f"template_{vote['template_id']}_raw_output"] = vote["raw_output"]
    return payload


def run_generation(args):
    import pandas as pd
    from tqdm import tqdm
    from src.data.vihallu import read_csv_robust
    from src.evaluation.latency import Timer, save_latency_summary
    from src.utils.seed import set_seed

    validate_generation_settings(args)
    set_seed(args.seed)
    manifest, model_key, model_entry, _, adapter_dir, _ = resolve_contract(
        args,
        require_model_dir=True,
        require_adapter=args.full_model_dir is None,
    )
    args.model_id = model_entry["hf_id"]
    malformed_path = ensure_malformed_file(args.malformed_csv)
    runtime_config = build_generation_config(args, adapter_dir, model_key, status="starting")
    write_generation_config(args.config_json, runtime_config)
    log_runtime_config(runtime_config)
    model, tokenizer, torch = load_model_and_tokenizer(args, adapter_dir)
    df = read_csv_robust(args.gold_csv)
    if args.sample_frac is not None:
        df = df.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    rows = []
    malformed_rows = []
    with Timer("generate_predictions_current_best", samples=len(df)) as timer:
        for row_index, row in enumerate(tqdm(df.to_dict("records"), total=len(df))):
            result = predict_one(model, tokenizer, torch, row, args)
            if result["malformed_reason"] is not None:
                malformed_rows.append(malformed_row_payload(row_index, row, result))
                write_malformed_rows(malformed_path, [malformed_rows[-1]])
                processed = row_index + 1
                malformed_pct = len(malformed_rows) / processed if processed else 0.0
                failed_config = build_generation_config(
                    args,
                    adapter_dir,
                    model_key,
                    status="failed_malformed_generation",
                    processed_rows=processed,
                    total_rows=len(df),
                    malformed_count=len(malformed_rows),
                    malformed_percentage=malformed_pct,
                )
                write_generation_config(args.config_json, failed_config)
                if len(malformed_rows) > args.max_malformed_count or malformed_pct > args.max_malformed_rate:
                    raise RuntimeError(
                        f"Malformed generation detected at row_index={row_index}, id={row['id']}. "
                        f"malformed_count={len(malformed_rows)} malformed_percentage={malformed_pct:.6f}. "
                        f"See {malformed_path}"
                    )
            rows.append({
                "row_index": row_index,
                "id": row["id"],
                "label": row["label"],
                "predict_label": result["predict_label"],
                "context": row["context"],
                "prompt": row["prompt"],
                "response": row["response"],
                "raw_output": result["raw_output"],
                "template_id": ",".join(map(str, TEMPLATE_IDS)),
                "ensemble": manifest["generation"].get("ensemble", "weighted_vote"),
                "is_malformed": result["malformed_reason"] is not None,
                "malformed_reason": result["malformed_reason"],
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
    config = build_generation_config(
        args,
        adapter_dir,
        model_key,
        status="completed",
        processed_rows=len(out_df),
        total_rows=len(df),
        malformed_count=len(malformed_rows),
        malformed_percentage=(len(malformed_rows) / len(out_df)) if len(out_df) else 0.0,
    )
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    write_generation_config(args.config_json, config)
    save_latency_summary([timer.summary()], args.latency_out_dir)
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.config_json}")
    print(f"Wrote {malformed_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv", default="vihallu-test.csv")
    parser.add_argument("--out_csv", default="results/predictions.csv")
    parser.add_argument("--config_json", default=None)
    parser.add_argument("--manifest", default="configs/experiment_manifest.yaml")
    parser.add_argument("--model_key", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--full_model_dir", default=None)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--malformed_csv", default=None)
    parser.add_argument("--latency_out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_frac", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_malformed_count", type=int, default=None)
    parser.add_argument("--max_malformed_rate", type=float, default=None)
    parser.add_argument("--no_quant", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-adapter", action="store_true")
    parser.add_argument("--require-model-dir", action="store_true")
    parser.add_argument("--allow_known_public_split_leakage", action="store_true")
    args = parser.parse_args()
    apply_manifest_defaults(args)
    if args.dry_run:
        validate_dry_run(args)
        return
    run_generation(args)


if __name__ == "__main__":
    main()
