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
PARSER_VERSION = "strict_label_v2_retry_scoring"
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
    "template_1_retry_raw_output",
    "template_1_parser_normalization_reason",
    "template_1_fallback_method",
    "template_2_label",
    "template_2_raw_output",
    "template_2_retry_raw_output",
    "template_2_parser_normalization_reason",
    "template_2_fallback_method",
    "template_3_label",
    "template_3_raw_output",
    "template_3_retry_raw_output",
    "template_3_parser_normalization_reason",
    "template_3_fallback_method",
]


def git_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def flag_enabled(name):
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def strip_outer_formatting(text):
    value = str(text).strip().lower()
    value = value.strip(" \t\r\n`'\"[](){}:.,;")
    value = re.sub(r"^(label|answer|nhan|nhãn)\s*[:：-]\s*", "", value, flags=re.IGNORECASE)
    value = value.strip(" \t\r\n`'\"[](){}:.,;")
    return value


def parse_label(text):
    raw = "" if text is None else str(text)
    normalized = strip_outer_formatting(raw)
    if normalized in LABELS:
        return normalized, "canonical_or_harmless_formatting"
    lowered = raw.strip().lower()
    label_hits = []
    for label in LABELS:
        if re.search(r"(?<![a-z])" + re.escape(label) + r"(?![a-z])", lowered):
            label_hits.append(label)
    unique_hits = sorted(set(label_hits), key=LABELS.index)
    if len(unique_hits) == 1:
        return unique_hits[0], "single_unambiguous_label_in_text"
    if len(unique_hits) > 1:
        raise ValueError(f"Conflicting labels in generated text: {unique_hits}")
    raise ValueError(f"Could not parse model output as a valid label: {raw[:200]}")


def normalize_label(text):
    label, _ = parse_label(text)
    return label


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


def create_prompt(row, template_id, retry=False):
    context = row.get("context", "")
    user_prompt = row.get("prompt", "")
    response = row.get("response", "")
    framing = {
        1: "Classify MODEL_RESPONSE by faithfulness to CONTEXT.",
        2: "Decide whether MODEL_RESPONSE is supported by CONTEXT for USER_PROMPT.",
        3: "Choose the hallucination label for MODEL_RESPONSE using only CONTEXT as evidence.",
    }.get(template_id, "Classify MODEL_RESPONSE by faithfulness to CONTEXT.")
    body = (
        "You are a strict Vietnamese hallucination detection classifier.\n\n"
        "Task:\n"
        f"{framing}\n\n"
        "Labels:\n"
        "no = MODEL_RESPONSE is fully supported by CONTEXT.\n"
        "intrinsic = MODEL_RESPONSE contradicts or distorts information in CONTEXT.\n"
        "extrinsic = MODEL_RESPONSE adds information not supported by CONTEXT.\n\n"
        "Rules:\n"
        "Return exactly one label.\n"
        "Allowed outputs: no, intrinsic, extrinsic.\n"
        "Do not explain.\n"
        "Do not copy CONTEXT.\n"
        "Do not answer the USER_PROMPT.\n"
        "Do not add punctuation.\n\n"
        "<CONTEXT>\n"
        f"{context}\n"
        "</CONTEXT>\n\n"
        "<USER_PROMPT>\n"
        f"{user_prompt}\n"
        "</USER_PROMPT>\n\n"
        "<MODEL_RESPONSE>\n"
        f"{response}\n"
        "</MODEL_RESPONSE>\n\n"
    )
    if retry:
        return (
            body
            + "Return exactly one label from this list:\n"
            + "no\n"
            + "intrinsic\n"
            + "extrinsic\n\n"
            + "No explanation. No punctuation. No extra words.\n\n"
            + "Label:"
        )
    return body + "Allowed outputs: no, intrinsic, extrinsic.\nAnswer with exactly one label.\n\nLabel:"


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


def build_generation_config(
    args,
    adapter_dir,
    model_key,
    status,
    processed_rows=0,
    total_rows=0,
    malformed_count=0,
    malformed_percentage=0.0,
    retry_count=0,
    label_scoring_count=0,
    adapter_info=None,
    model_info=None,
    resumed_from_existing_predictions=False,
    skipped_rows=0,
):
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
        "retry_count": int(retry_count),
        "label_scoring_count": int(label_scoring_count),
        "parser_version": PARSER_VERSION,
        "prompt_template_ids": TEMPLATE_IDS,
        "resumed_from_existing_predictions": bool(resumed_from_existing_predictions),
        "skipped_rows": int(skipped_rows),
        "limit": args.limit,
        "sample_frac": args.sample_frac,
        "generation": {
            "do_sample": False,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "retry_max_new_tokens": 3,
            "max_prompt_tokens": args.max_prompt_tokens,
            "label_scoring_fallback": not args.disable_label_scoring,
        },
        "dtype": "bfloat16",
        "quantization": None if args.no_quant else "nf4_4bit",
        "adapter_validation": adapter_info,
        "model_alias_validation": model_info.get("reference_validation") if isinstance(model_info, dict) else None,
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
    if args.max_prompt_tokens is None:
        args.max_prompt_tokens = int(manifest.get("max_length", 1024))
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
            "retry_max_new_tokens": 3,
            "max_prompt_tokens": args.max_prompt_tokens,
            "label_scoring_fallback": not args.disable_label_scoring,
        },
        "parser_version": PARSER_VERSION,
        "labels": LABELS,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def log_runtime_config(config):
    print("RUNTIME_CONFIG " + json.dumps(config, ensure_ascii=False, sort_keys=True))


def load_model_and_tokenizer(args, adapter_dir):
    from src.models.loader import load_causal_lm

    model_dir = args.full_model_dir or args.model_dir
    model, tokenizer = load_causal_lm(
        model_dir,
        quantized=not args.no_quant,
        adapter_dir=None if args.full_model_dir else adapter_dir,
        adapter_required=args.full_model_dir is None,
    )
    import torch

    return model, tokenizer, torch


def encode_prompt(tokenizer, prompt, max_length):
    original_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        return tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    finally:
        tokenizer.truncation_side = original_side


def generate_label_text(model, tokenizer, torch, prompt, args, max_new_tokens):
    device = next(model.parameters()).device
    inputs = encode_prompt(tokenizer, prompt, args.max_prompt_tokens)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def score_labels(model, tokenizer, torch, prompt, args):
    import torch.nn.functional as functional

    device = next(model.parameters()).device
    scores = {}
    for label in LABELS:
        label_ids = tokenizer(" " + label, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
        max_prompt_length = max(1, args.max_prompt_tokens - int(label_ids.numel()))
        prompt_inputs = encode_prompt(tokenizer, prompt, max_prompt_length)
        prompt_ids = prompt_inputs["input_ids"][0].to(device)
        input_ids = torch.cat([prompt_ids, label_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        losses = []
        prompt_len = int(prompt_ids.numel())
        for offset, token_id in enumerate(label_ids):
            position = prompt_len + offset - 1
            token_loss = functional.cross_entropy(logits[0, position, :].unsqueeze(0), token_id.view(1), reduction="mean")
            losses.append(float(token_loss.detach().cpu()))
        scores[label] = sum(losses) / max(1, len(losses))
    best = min(scores, key=scores.get)
    return best, scores


def predict_template(model, tokenizer, torch, row, args, template_id):
    from src.utils.seed import set_seed

    prompt = create_prompt(row, template_id, retry=False)
    retry_prompt = create_prompt(row, template_id, retry=True)
    set_seed(args.seed + template_id)
    raw = generate_label_text(model, tokenizer, torch, prompt, args, args.max_new_tokens)
    try:
        label, reason = parse_label(raw)
        return {
            "template_id": template_id,
            "label": label,
            "raw_output": raw,
            "retry_raw_output": None,
            "parser_normalization_reason": reason,
            "fallback_method": None,
            "label_scores": None,
            "malformed_reason": None,
        }
    except ValueError as first_error:
        set_seed(args.seed + 1000 + template_id)
        retry_raw = generate_label_text(model, tokenizer, torch, retry_prompt, args, 3)
        try:
            label, reason = parse_label(retry_raw)
            return {
                "template_id": template_id,
                "label": label,
                "raw_output": raw,
                "retry_raw_output": retry_raw,
                "parser_normalization_reason": "retry_" + reason,
                "fallback_method": "retry_generation",
                "label_scores": None,
                "malformed_reason": None,
            }
        except ValueError as retry_error:
            if args.disable_label_scoring:
                return {
                    "template_id": template_id,
                    "label": None,
                    "raw_output": raw,
                    "retry_raw_output": retry_raw,
                    "parser_normalization_reason": None,
                    "fallback_method": None,
                    "label_scores": None,
                    "malformed_reason": f"unparsable_output; first={first_error}; retry={retry_error}",
                }
            label, scores = score_labels(model, tokenizer, torch, retry_prompt, args)
            return {
                "template_id": template_id,
                "label": label,
                "raw_output": raw,
                "retry_raw_output": retry_raw,
                "parser_normalization_reason": "label_scoring_after_unparsable_generation",
                "fallback_method": "label_scoring",
                "label_scores": scores,
                "malformed_reason": None,
            }


def predict_one(model, tokenizer, torch, row, args):
    votes = [predict_template(model, tokenizer, torch, row, args, template_id) for template_id in TEMPLATE_IDS]
    malformed_votes = [vote for vote in votes if vote["malformed_reason"]]
    raw_joined = " || ".join(vote["raw_output"] or "" for vote in votes)
    retry_count = sum(1 for vote in votes if vote["retry_raw_output"] is not None)
    label_scoring_count = sum(1 for vote in votes if vote["fallback_method"] == "label_scoring")
    if malformed_votes:
        reasons = [f"template_{vote['template_id']}:{vote['malformed_reason']}" for vote in malformed_votes]
        return {
            "predict_label": None,
            "raw_output": raw_joined,
            "votes": votes,
            "malformed_reason": "; ".join(reasons),
            "empty_vote_count": sum(1 for vote in votes if vote["malformed_reason"] == "empty_output"),
            "unparsable_vote_count": sum(1 for vote in votes if vote["malformed_reason"]),
            "retry_count": retry_count,
            "label_scoring_count": label_scoring_count,
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
        "retry_count": retry_count,
        "label_scoring_count": label_scoring_count,
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
        payload[f"template_{vote['template_id']}_retry_raw_output"] = vote["retry_raw_output"]
        payload[f"template_{vote['template_id']}_parser_normalization_reason"] = vote["parser_normalization_reason"]
        payload[f"template_{vote['template_id']}_fallback_method"] = vote["fallback_method"]
    return payload


def load_resumable_predictions(path):
    pred_path = Path(path)
    if not pred_path.exists():
        return {}, 0
    import pandas as pd

    pred = pd.read_csv(pred_path)
    if "row_index" not in pred.columns or "predict_label" not in pred.columns:
        return {}, 0
    valid = pred[pred["predict_label"].astype(str).str.strip().isin(LABELS)].copy()
    if "is_malformed" in valid.columns:
        malformed = valid["is_malformed"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
        valid = valid[~malformed]
    rows = {}
    for record in valid.to_dict("records"):
        try:
            rows[int(record["row_index"])] = record
        except Exception:
            continue
    return rows, len(rows)


def run_generation(args):
    import pandas as pd
    from tqdm import tqdm
    from src.data.vihallu import read_csv_robust, validate_gold_df
    from src.evaluation.latency import Timer, save_latency_summary
    from src.utils.seed import set_seed

    validate_generation_settings(args)
    set_seed(args.seed)
    manifest, model_key, model_entry, model_info, adapter_dir, adapter_info = resolve_contract(
        args,
        require_model_dir=True,
        require_adapter=args.full_model_dir is None,
    )
    validate_manifest_contract(manifest, args.manifest)
    args.model_id = model_entry["hf_id"]
    malformed_path = ensure_malformed_file(args.malformed_csv)
    resume_enabled = bool(args.resume or flag_enabled("RESUME_PREDICTIONS"))
    resumed_rows, skipped_rows = load_resumable_predictions(args.out_csv) if resume_enabled else ({}, 0)
    df = read_csv_robust(args.gold_csv)
    validate_gold_df(df, args.gold_csv, allow_duplicate_ids=True)
    if args.sample_frac is not None:
        df = df.sample(frac=args.sample_frac, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    runtime_config = build_generation_config(
        args,
        adapter_dir,
        model_key,
        status="starting",
        adapter_info=adapter_info,
        model_info=model_info,
        resumed_from_existing_predictions=resume_enabled,
        skipped_rows=skipped_rows,
    )
    write_generation_config(args.config_json, runtime_config)
    log_runtime_config(runtime_config)
    model, tokenizer, torch = load_model_and_tokenizer(args, adapter_dir)
    rows = []
    malformed_rows = []
    retry_count = 0
    label_scoring_count = 0
    with Timer("generate_predictions_current_best", samples=len(df)) as timer:
        for row_index, row in enumerate(tqdm(df.to_dict("records"), total=len(df))):
            if row_index in resumed_rows:
                rows.append(resumed_rows[row_index])
                continue
            result = predict_one(model, tokenizer, torch, row, args)
            retry_count += result["retry_count"]
            label_scoring_count += result["label_scoring_count"]
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
                    retry_count=retry_count,
                    label_scoring_count=label_scoring_count,
                    adapter_info=adapter_info,
                    model_info=model_info,
                    resumed_from_existing_predictions=resume_enabled,
                    skipped_rows=skipped_rows,
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
                "retry_count": result["retry_count"],
                "label_scoring_count": result["label_scoring_count"],
                "fallback_methods": ",".join(sorted({vote["fallback_method"] for vote in result["votes"] if vote["fallback_method"]})),
                "label_scores": json.dumps({f"template_{vote['template_id']}": vote["label_scores"] for vote in result["votes"] if vote["label_scores"]}, ensure_ascii=False, sort_keys=True),
                "template_1_raw_output": next((vote["raw_output"] for vote in result["votes"] if vote["template_id"] == 1), None),
                "template_1_retry_raw_output": next((vote["retry_raw_output"] for vote in result["votes"] if vote["template_id"] == 1), None),
                "template_1_label": next((vote["label"] for vote in result["votes"] if vote["template_id"] == 1), None),
                "template_1_fallback_method": next((vote["fallback_method"] for vote in result["votes"] if vote["template_id"] == 1), None),
                "template_2_raw_output": next((vote["raw_output"] for vote in result["votes"] if vote["template_id"] == 2), None),
                "template_2_retry_raw_output": next((vote["retry_raw_output"] for vote in result["votes"] if vote["template_id"] == 2), None),
                "template_2_label": next((vote["label"] for vote in result["votes"] if vote["template_id"] == 2), None),
                "template_2_fallback_method": next((vote["fallback_method"] for vote in result["votes"] if vote["template_id"] == 2), None),
                "template_3_raw_output": next((vote["raw_output"] for vote in result["votes"] if vote["template_id"] == 3), None),
                "template_3_retry_raw_output": next((vote["retry_raw_output"] for vote in result["votes"] if vote["template_id"] == 3), None),
                "template_3_label": next((vote["label"] for vote in result["votes"] if vote["template_id"] == 3), None),
                "template_3_fallback_method": next((vote["fallback_method"] for vote in result["votes"] if vote["template_id"] == 3), None),
            })
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    if args.limit is None and len(out_df) != len(df):
        raise RuntimeError(f"Prediction row count mismatch: expected {len(df)}, found {len(out_df)}")
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
        retry_count=retry_count,
        label_scoring_count=label_scoring_count,
        adapter_info=adapter_info,
        model_info=model_info,
        resumed_from_existing_predictions=resume_enabled,
        skipped_rows=skipped_rows,
    )
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    write_generation_config(args.config_json, config)
    save_latency_summary([timer.summary()], args.latency_out_dir)
    print(f"Wrote {out_csv}")
    print(f"Wrote {args.config_json}")
    print(f"Wrote {malformed_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_csv", "--gold-csv", dest="gold_csv", default="vihallu-test.csv")
    parser.add_argument("--out_csv", "--out-csv", dest="out_csv", default="results/predictions.csv")
    parser.add_argument("--config_json", "--config-json", dest="config_json", default=None)
    parser.add_argument("--manifest", default="configs/experiment_manifest.yaml")
    parser.add_argument("--model_key", "--model-key", dest="model_key", default=None)
    parser.add_argument("--model_dir", "--model-dir", dest="model_dir", default=None)
    parser.add_argument("--full_model_dir", "--full-model-dir", dest="full_model_dir", default=None)
    parser.add_argument("--adapter_dir", "--adapter-dir", dest="adapter_dir", default=None)
    parser.add_argument("--malformed_csv", "--malformed-csv", dest="malformed_csv", default=None)
    parser.add_argument("--latency_out_dir", "--latency-out-dir", dest="latency_out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_frac", "--sample-frac", dest="sample_frac", type=float, default=None)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", dest="max_new_tokens", type=int, default=None)
    parser.add_argument("--max_prompt_tokens", "--max-prompt-tokens", dest="max_prompt_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", "--top-p", dest="top_p", type=float, default=None)
    parser.add_argument("--max_malformed_count", "--max-malformed-count", dest="max_malformed_count", type=int, default=None)
    parser.add_argument("--max_malformed_rate", "--max-malformed-rate", dest="max_malformed_rate", type=float, default=None)
    parser.add_argument("--no_quant", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--disable-label-scoring", action="store_true")
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
