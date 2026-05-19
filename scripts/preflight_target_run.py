import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_environment


TOKENIZER_REQUIRED_FILES = ["tokenizer_config.json"]
TOKENIZER_ASSET_FILES = ["tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model", "vocab.txt", "vocab.json"]
MODEL_REQUIRED_FILES = ["config.json"]
ADAPTER_REQUIRED_FILES = ["adapter_config.json"]
ADAPTER_WEIGHT_FILES = ["adapter_model.safetensors", "adapter_model.bin"]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_ref(value):
    return str(value).replace("\\", "/").strip().rstrip("/").lower()


def ref_aliases(value):
    text = normalize_ref(value)
    return {text, Path(text).name.lower()}


def accepted_model_refs(model_entry, model_dir):
    values = [model_entry["hf_id"], model_entry["local_dir"], model_dir]
    aliases = set()
    display = []
    for value in values:
        if not value:
            continue
        aliases.update(ref_aliases(value))
        display.append(str(value))
    return aliases, display


def find_manifest_model(manifest, model_key, model_dir):
    models = manifest["models"]
    if model_key:
        if model_key not in models:
            raise KeyError(f"Model key {model_key} not found in manifest. Available keys: {sorted(models)}")
        return model_key, models[model_key]
    model_aliases = ref_aliases(model_dir)
    matches = []
    for key, entry in models.items():
        if model_aliases & ref_aliases(entry["local_dir"]):
            matches.append((key, entry))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"Could not resolve manifest model entry for model_dir={model_dir}. Available local_dir values: {[models[k]['local_dir'] for k in sorted(models)]}")
    raise KeyError(f"Multiple manifest model entries match model_dir={model_dir}: {[key for key, _ in matches]}")


def require_file(path, label):
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is not a file: {candidate}")
    return candidate


def require_dir(path, label):
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"{label} not found: {candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {candidate}")
    return candidate


def require_any(parent, names, label):
    candidates = [Path(parent) / name for name in names]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{label} not found. Checked: {', '.join(str(item) for item in candidates)}")


def find_adapter_candidate(manifest, explicit=None):
    if explicit:
        return Path(explicit)
    candidates = []
    for root in manifest["lora"]["adapter_search_paths"]:
        root_path = Path(root)
        if not root_path.exists():
            continue
        candidates.extend(sorted(path.parent for path in root_path.rglob("adapter_config.json")))
    if candidates:
        return sorted(candidates, key=lambda path: (len(str(path)), str(path)))[0]
    checked = ", ".join(manifest["lora"]["adapter_search_paths"])
    raise FileNotFoundError(f"No adapter directory found. Checked search roots: {checked}")


def validate_reference_match(actual, label, allowed_aliases, allowed_display):
    if not actual:
        return None
    actual_aliases = ref_aliases(actual)
    if actual_aliases & allowed_aliases:
        return str(actual)
    raise RuntimeError(f"{label} mismatch: found {actual}. Expected one of: {allowed_display}")


def validate_model_dir(model_dir, model_entry):
    model_root = require_dir(model_dir, "Model directory")
    require_file(model_root / MODEL_REQUIRED_FILES[0], "Model config")
    require_file(model_root / TOKENIZER_REQUIRED_FILES[0], "Tokenizer config")
    tokenizer_asset = require_any(model_root, TOKENIZER_ASSET_FILES, "Tokenizer asset")
    config = load_json(model_root / MODEL_REQUIRED_FILES[0])
    tokenizer_config = load_json(model_root / TOKENIZER_REQUIRED_FILES[0])
    allowed_aliases, allowed_display = accepted_model_refs(model_entry, model_dir)
    validate_reference_match(config.get("_name_or_path") or config.get("name_or_path"), f"{model_root / MODEL_REQUIRED_FILES[0]} base model reference", allowed_aliases, allowed_display)
    validate_reference_match(tokenizer_config.get("name_or_path"), f"{model_root / TOKENIZER_REQUIRED_FILES[0]} tokenizer reference", allowed_aliases, allowed_display)
    return {
        "model_dir": str(model_root),
        "model_config": str(model_root / MODEL_REQUIRED_FILES[0]),
        "tokenizer_config": str(model_root / TOKENIZER_REQUIRED_FILES[0]),
        "tokenizer_asset": str(tokenizer_asset),
    }


def validate_adapter_dir(adapter_dir, model_entry, model_dir):
    adapter_root = require_dir(adapter_dir, "Adapter directory")
    adapter_config_path = require_file(adapter_root / ADAPTER_REQUIRED_FILES[0], "Adapter config")
    adapter_weights = require_any(adapter_root, ADAPTER_WEIGHT_FILES, "Adapter weights")
    adapter_config = load_json(adapter_config_path)
    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"{adapter_config_path} missing base_model_name_or_path")
    allowed_aliases, allowed_display = accepted_model_refs(model_entry, model_dir)
    validate_reference_match(base_model, f"{adapter_config_path} base_model_name_or_path", allowed_aliases, allowed_display)
    if "r" not in adapter_config:
        raise ValueError(f"{adapter_config_path} missing LoRA rank field r")
    return {
        "adapter_dir": str(adapter_root),
        "adapter_config": str(adapter_config_path),
        "adapter_weights": str(adapter_weights),
        "base_model_name_or_path": str(base_model),
        "lora_r": int(adapter_config["r"]),
    }


def resolve_gold_csv(manifest, explicit):
    if explicit:
        gold_path = Path(explicit)
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold CSV not found: {gold_path}")
        return gold_path
    split_paths = manifest["datasets"]["test"]["fallback_paths"]
    return Path(verify_environment.resolve_dataset({"datasets": {"test": {"fallback_paths": split_paths}}}, "test")["test"])


def validate_gold_csv(manifest, gold_csv):
    from src.data.vihallu import infer_split_from_path, read_csv_robust, validate_gold_df

    gold_path = resolve_gold_csv(manifest, gold_csv)
    split = infer_split_from_path(gold_path)
    if split == "private_test":
        raise RuntimeError(f"{gold_path} is the private test split and must not be used for evidence evaluation.")
    df = read_csv_robust(gold_path)
    test_name = Path(manifest["datasets"]["test"]["canonical"]).name
    allow_duplicate_ids = gold_path.name == test_name
    validate_gold_df(df, gold_path, allow_duplicate_ids=allow_duplicate_ids)
    return gold_path, df, allow_duplicate_ids, split


def validate_dataset_semantics():
    from src.data.vihallu import LABELS, leakage_overlap_report, load_vihallu_split

    train_df = load_vihallu_split("train")
    test_df = load_vihallu_split("test")
    private_df = load_vihallu_split("private_test")
    if "label" in private_df.columns:
        raise RuntimeError("vihallu-private-test.csv unexpectedly contains a label column and must not be used for evidence evaluation.")
    if "predict_label" not in private_df.columns:
        raise RuntimeError("vihallu-private-test.csv is missing predict_label and its semantics are unclear.")
    leakage = leakage_overlap_report(train_df, test_df)
    if leakage["overlap_rows"] > 0:
        raise RuntimeError(
            "Detected train/test leakage in ViHallu public splits: "
            f"overlap_rows={leakage['overlap_rows']} overlap_ids={leakage['overlap_ids']} sample_ids={leakage['sample_ids']}. "
            "Benchmark execution should not continue until the evaluation split is corrected."
        )
    return {
        "labels": LABELS,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "private_test_rows": int(len(private_df)),
        "private_has_label": "label" in private_df.columns,
        "private_has_predict_label": "predict_label" in private_df.columns,
        "leakage": leakage,
    }


def validate_malformed_csv(path):
    from src.data.vihallu import read_csv_robust

    malformed_path = Path(path)
    if not malformed_path.exists():
        return {"path": str(malformed_path), "exists": False, "rows": 0}
    df = read_csv_robust(malformed_path)
    rows = int(len(df))
    if rows > 0:
        raise RuntimeError(f"{malformed_path} contains {rows} malformed prediction rows. Fix generation parsing failures before evidence generation.")
    return {"path": str(malformed_path), "exists": True, "rows": rows}


def validate_prediction_csv(pred_csv, gold_df, pred_col, allow_duplicate_ids, allow_partial=False):
    from src.data.vihallu import read_csv_robust, validate_prediction_df

    pred_path = Path(pred_csv)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {pred_path}")
    pred = read_csv_robust(pred_path)
    validate_prediction_df(pred, pred_path, pred_col=pred_col, expected_ids=gold_df["id"], expected_count=len(gold_df), allow_partial=allow_partial, allow_duplicate_ids=allow_duplicate_ids)
    if "raw_output" in pred.columns:
        empty_raw = pred["raw_output"].isna() | pred["raw_output"].astype(str).str.strip().eq("")
        if empty_raw.any():
            rows = [int(idx) + 2 for idx in pred.index[empty_raw].tolist()[:20]]
            raise ValueError(f"{pred_path} contains empty raw_output values at CSV rows {rows}")
    if "is_malformed" in pred.columns:
        malformed_mask = pred["is_malformed"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
        malformed_count = int(malformed_mask.sum())
        if malformed_count > 0:
            malformed_pct = 100.0 * malformed_count / len(pred) if len(pred) else 0.0
            raise RuntimeError(f"{pred_path} reports {malformed_count} malformed predictions ({malformed_pct:.4f}%).")
    return {"path": str(pred_path), "rows": int(len(pred))}


def validate_output_dir(path):
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise RuntimeError(f"Output directory path is not a directory: {output_dir}")
    probe = output_dir / ".preflight_write_test"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    return str(output_dir)


def validate_output_paths(out_dir, pred_out_csv, config_json, malformed_csv=None):
    results = {
        "out_dir": validate_output_dir(out_dir),
        "prediction_parent": validate_output_dir(Path(pred_out_csv).parent),
        "config_parent": validate_output_dir(Path(config_json).parent),
    }
    if malformed_csv:
        results["malformed_parent"] = validate_output_dir(Path(malformed_csv).parent)
    return results


def build_summary(args, env_summary, model_key, model_entry, gold_info, output_info, adapter_info, model_info, prediction_info, malformed_info, dataset_semantics):
    return {
        "mode": "precheck_only" if args.precheck_only else "preflight",
        "manifest": args.manifest,
        "model_key": model_key,
        "model_hf_id": model_entry["hf_id"],
        "model_dir": args.model_dir,
        "gold_csv": str(gold_info["path"]),
        "gold_rows": int(gold_info["rows"]),
        "gold_split": gold_info["split"],
        "prediction_csv": prediction_info,
        "malformed_csv": malformed_info,
        "output_paths": output_info,
        "adapter": adapter_info,
        "model_files": model_info,
        "environment": env_summary,
        "dataset_semantics": dataset_semantics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/experiment_manifest.yaml")
    parser.add_argument("--model_key", default="vistral")
    parser.add_argument("--model_dir", default="models/Vistral-7B-Chat")
    parser.add_argument("--gold_csv", default=None)
    parser.add_argument("--pred_csv", default=None)
    parser.add_argument("--pred_col", default="predict_label")
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--out_dir", default="results/paper_evidence")
    parser.add_argument("--pred_out_csv", default="results/predictions.csv")
    parser.add_argument("--config_json", default="results/prediction_config.json")
    parser.add_argument("--malformed_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--require_model_dir", action="store_true")
    parser.add_argument("--require_adapter", action="store_true")
    parser.add_argument("--require_hf_token", action="store_true")
    parser.add_argument("--require_active_env", action="store_true")
    parser.add_argument("--require_bitsandbytes", action="store_true")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--require_bf16", action="store_true")
    parser.add_argument("--precheck_only", action="store_true")
    args = parser.parse_args()

    modules, versions, warnings = verify_environment.check_imports(require_bitsandbytes=args.require_bitsandbytes)
    manifest = verify_environment.load_manifest(args.manifest)
    token_source = verify_environment.find_token_source()
    if args.require_hf_token and token_source is None:
        raise RuntimeError("HF_TOKEN is required in .env or the process environment")
    active_env = verify_environment.active_environment()
    if args.require_active_env and not active_env:
        raise RuntimeError("No active Python environment detected. Activate uv venv or conda before target execution.")
    cuda_summary = verify_environment.check_cuda(modules["torch"], require_cuda=args.require_cuda, require_bf16=args.require_bf16)
    model_key, model_entry = find_manifest_model(manifest, args.model_key, args.model_dir)
    dataset_semantics = validate_dataset_semantics()
    gold_path, gold_df, allow_duplicate_ids, gold_split = validate_gold_csv(manifest, args.gold_csv)
    output_info = validate_output_paths(args.out_dir, args.pred_out_csv, args.config_json, args.malformed_csv)
    model_info = None
    model_dir_path = Path(args.model_dir)
    if args.require_model_dir or model_dir_path.exists():
        model_info = validate_model_dir(args.model_dir, model_entry)
    adapter_info = None
    if args.require_adapter or args.adapter_dir:
        adapter_path = find_adapter_candidate(manifest, args.adapter_dir)
        adapter_info = validate_adapter_dir(adapter_path, model_entry, args.model_dir)
        if model_info is None and args.require_model_dir:
            raise RuntimeError("Adapter validation requires a validated model directory, but model validation did not run.")
    prediction_info = None
    if args.pred_csv:
        prediction_info = validate_prediction_csv(args.pred_csv, gold_df, args.pred_col, allow_duplicate_ids, allow_partial=args.allow_partial)
    malformed_info = validate_malformed_csv(args.malformed_csv)
    env_summary = {
        "python": sys.version.split()[0],
        "platform": verify_environment.platform.platform(),
        "active_environment": bool(active_env),
        "token_key_present": token_source is not None,
        "package_versions": versions,
        "warnings": warnings,
        "cuda": cuda_summary,
    }
    summary = build_summary(
        args,
        env_summary,
        model_key,
        model_entry,
        {"path": gold_path, "rows": len(gold_df), "split": gold_split},
        output_info,
        adapter_info,
        model_info,
        prediction_info,
        malformed_info,
        dataset_semantics,
    )
    if args.summary_json:
        verify_environment.write_summary(args.summary_json, summary)
    print("PREFLIGHT_OK")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
