from pathlib import Path

from huggingface_hub import snapshot_download

from src.utils.env import load_hf_token


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
    "gemma4_e2b_it": {
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
    p = Path(local_dir)
    if not p.exists():
        return False
    markers = ["config.json", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model", "vocab.txt", "processor_config.json", "preprocessor_config.json"]
    return (p / "config.json").exists() and any((p / marker).exists() for marker in markers if marker != "config.json")


def model_ready_for_entry(entry):
    p = Path(entry["local_dir"])
    if not p.exists():
        return False, "missing_local_model"
    missing = [name for name in entry.get("required_files", ["config.json"]) if not (p / name).exists()]
    if missing:
        return False, "missing_required_files:" + ",".join(missing)
    assets = ["tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model", "vocab.txt", "vocab.json", "processor_config.json", "preprocessor_config.json", "tokenizer_config.json"]
    if not any((p / name).exists() for name in assets):
        return False, "missing_tokenizer_or_processor_assets"
    return True, None


def download_hf_snapshot(model_id, local_dir, token=None, allow_patterns=None):
    local_path = Path(local_dir)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_path),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=allow_patterns,
    )
    return local_path


def download_one(name, token=None):
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


def download_entry(name, entry, token=None):
    hf_token = token or load_hf_token(required=True)
    ready, reason = model_ready_for_entry(entry)
    if ready:
        return {"model_key": name, "model_id": entry["model_id"], "local_dir": entry["local_dir"], "status": "present", "skip_reason": None}
    try:
        download_hf_snapshot(entry["model_id"], entry["local_dir"], token=hf_token)
        ready, reason = model_ready_for_entry(entry)
        if not ready:
            return {"model_key": name, "model_id": entry["model_id"], "local_dir": entry["local_dir"], "status": "failed", "skip_reason": reason}
        return {"model_key": name, "model_id": entry["model_id"], "local_dir": entry["local_dir"], "status": "downloaded", "skip_reason": None}
    except Exception as exc:
        return {"model_key": name, "model_id": entry["model_id"], "local_dir": entry["local_dir"], "status": "failed", "skip_reason": f"{type(exc).__name__}: {exc}"}
