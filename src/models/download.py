from pathlib import Path

MIN_WEIGHT_BYTES = 1024 * 1024
WEIGHT_FILES = ["pytorch_model.bin", "model.safetensors"]
WEIGHT_INDEX_FILES = ["model.safetensors.index.json", "pytorch_model.bin.index.json"]
TOKENIZER_ASSETS = [
    "tokenizer.json",
    "tokenizer.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
]


MODEL_SPECS = {
    "vistral": {
        "repo_id": "Viet-Mistral/Vistral-7B-Chat",
        "local_dir": "models/Vistral-7B-Chat",
        "aliases": ("Viet-Mistral/Vistral-7B-Chat", "uonlp/viet-mistral-sft-v1"),
    },
    "qwen3_4b": {
        "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
        "local_dir": "models/Qwen3-4B-Instruct-2507",
    },
    "qwen35_4b": {
        "repo_id": "Qwen/Qwen3.5-4B",
        "local_dir": "models/Qwen3.5-4B",
    },
    "gemma4_e2b": {
        "repo_id": "google/gemma-4-E2B-it",
        "local_dir": "models/gemma-4-E2B-it",
    },
    "phobert": {
        "repo_id": "vinai/phobert-base-v2",
        "local_dir": "models/phobert-base-v2",
    },
    "xlmr": {
        "repo_id": "FacebookAI/xlm-roberta-base",
        "local_dir": "models/xlm-roberta-base",
    },
}

MODELS = {name: (spec["repo_id"], spec["local_dir"]) for name, spec in MODEL_SPECS.items()}


def model_ready(local_dir):
    return validate_local_model_entry({"local_dir": local_dir, "required_files": ["config.json"]})["ok"]


def result_template(entry):
    local_dir = Path(entry.get("local_dir", ""))
    return {
        "ok": False,
        "ready": False,
        "status": "missing_local_model",
        "reason": "",
        "local_dir": str(local_dir),
        "missing": [],
        "files": {},
    }


def nonempty_file(path, min_bytes=1):
    p = Path(path)
    return p.exists() and p.is_file() and p.stat().st_size >= min_bytes


def present_files(root, names):
    return [name for name in names if (Path(root) / name).exists()]


def tokenizer_files(root):
    p = Path(root)
    files = present_files(p, TOKENIZER_ASSETS)
    if (p / "vocab.txt").exists() and (p / "bpe.codes").exists():
        files.extend(["vocab.txt", "bpe.codes"])
    return sorted(set(files))


def read_index_shards(root, index_name):
    path = Path(root) / index_name
    if not path.exists():
        return []
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    values = sorted(set((data.get("weight_map") or {}).values()))
    return [Path(root) / value for value in values]


def weight_files(root):
    p = Path(root)
    direct = [name for name in WEIGHT_FILES if nonempty_file(p / name, MIN_WEIGHT_BYTES)]
    if direct:
        return direct, []
    for index_name in WEIGHT_INDEX_FILES:
        shards = read_index_shards(p, index_name)
        if shards:
            missing = [str(item) for item in shards if not nonempty_file(item, MIN_WEIGHT_BYTES)]
            if missing:
                return [], missing
            return [index_name] + [str(item.relative_to(p)) for item in shards], []
    candidates = []
    for pattern in ["*.safetensors", "*.bin"]:
        candidates.extend([item.name for item in p.glob(pattern) if item.name != "training_args.bin" and nonempty_file(item, MIN_WEIGHT_BYTES)])
    return sorted(set(candidates)), []


def tokenizer_load_status(entry):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    local_dir = str(entry["local_dir"])
    try:
        AutoTokenizer.from_pretrained(local_dir)
        return True, ""
    except Exception as first_exc:
        if entry.get("model_key") == "phobert":
            try:
                AutoTokenizer.from_pretrained(local_dir, use_fast=False)
                return True, ""
            except Exception as second_exc:
                return False, f"{type(second_exc).__name__}: {second_exc}"
        return False, f"{type(first_exc).__name__}: {first_exc}"


def validate_local_model_entry(entry, load_tokenizer=False):
    result = result_template(entry)
    local_dir = Path(entry.get("local_dir", ""))
    result["local_dir"] = str(local_dir)
    if not local_dir.exists():
        result["reason"] = "missing_local_model"
        result["missing"] = [str(local_dir)]
        return result
    if not local_dir.is_dir():
        result["status"] = "incomplete_local_model"
        result["reason"] = "local_path_is_not_directory"
        result["missing"] = [str(local_dir)]
        return result
    required_files = entry.get("required_files") or ["config.json"]
    missing_required = [name for name in required_files if not (local_dir / name).is_file()]
    result["files"]["required_files"] = present_files(local_dir, required_files)
    if missing_required:
        result["status"] = "incomplete_local_model"
        result["reason"] = "missing_required_files:" + ",".join(missing_required)
        result["missing"] = missing_required
        return result
    tokens = tokenizer_files(local_dir)
    result["files"]["tokenizer_assets"] = tokens
    if not tokens:
        result["status"] = "incomplete_local_model"
        result["reason"] = "missing_tokenizer_assets"
        result["missing"] = ["tokenizer_assets"]
        return result
    weights, missing_shards = weight_files(local_dir)
    result["files"]["weights"] = weights
    if missing_shards:
        result["status"] = "incomplete_local_model"
        result["reason"] = "missing_or_tiny_weight_shards:" + ",".join(missing_shards[:10])
        result["missing"] = missing_shards
        return result
    if not weights:
        result["status"] = "incomplete_local_model"
        result["reason"] = "missing_model_weights"
        result["missing"] = ["model_weights"]
        return result
    if load_tokenizer:
        ok, reason = tokenizer_load_status(entry)
        if not ok:
            result["status"] = "tokenizer_load_failed"
            result["reason"] = reason
            result["missing"] = ["loadable_tokenizer"]
            return result
    result["ok"] = True
    result["ready"] = True
    result["status"] = "present"
    result["reason"] = ""
    return result


def model_ready_for_entry(entry):
    result = validate_local_model_entry(entry)
    if result["ok"]:
        return True, None
    return False, result["reason"]


def download_hf_snapshot(model_id, local_dir, token=None, allow_patterns=None):
    from huggingface_hub import snapshot_download

    p = Path(local_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(p),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=allow_patterns,
    )
    return p


def download_entry(name, entry, token=None):
    from src.utils.env import load_hf_token

    hf_token = token or load_hf_token(required=True)
    before = validate_local_model_entry(entry)
    if before["ok"]:
        return {"model_key": name, "model_id": entry["model_id"], "local_dir": entry["local_dir"], "status": "present", "skip_reason": None}
    try:
        download_hf_snapshot(entry["model_id"], entry["local_dir"], token=hf_token)
        after = validate_local_model_entry(entry)
        if not after["ok"]:
            return {"model_key": name, "model_id": entry["model_id"], "local_dir": entry["local_dir"], "status": "failed", "previous_status": before["status"], "skip_reason": "incomplete_local_model:" + str(after["reason"])}
        return {"model_key": name, "model_id": entry["model_id"], "local_dir": entry["local_dir"], "status": "downloaded", "previous_status": before["status"], "skip_reason": None}
    except Exception as exc:
        return {"model_key": name, "model_id": entry["model_id"], "local_dir": entry["local_dir"], "status": "failed", "previous_status": before["status"], "skip_reason": f"{type(exc).__name__}: {exc}"}


def download_one(name, token=None):
    from src.utils.env import load_hf_token

    if name not in MODELS:
        raise KeyError(f"Unknown model key: {name}")
    spec = MODEL_SPECS[name]
    repo_id = spec["repo_id"]
    local_dir = spec["local_dir"]
    if model_ready(local_dir):
        print(f"SKIP {name} {local_dir}")
        return Path(local_dir)
    hf_token = token or load_hf_token(required=True)
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    print(f"DOWNLOAD {name} {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, token=hf_token)
    if not model_ready(local_dir):
        raise RuntimeError(f"Downloaded model is missing expected marker files: {local_dir}")
    return Path(local_dir)
