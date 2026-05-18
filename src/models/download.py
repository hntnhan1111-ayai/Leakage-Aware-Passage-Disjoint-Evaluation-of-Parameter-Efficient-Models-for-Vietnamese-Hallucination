from pathlib import Path

from huggingface_hub import snapshot_download

from src.utils.env import load_hf_token


MODELS = {
    "vistral": ("Viet-Mistral/Vistral-7B-Chat", "models/Vistral-7B-Chat"),
    "qwen3_4b": ("Qwen/Qwen3-4B-Instruct-2507", "models/Qwen3-4B-Instruct-2507"),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "models/Qwen3.5-4B"),
    "gemma4_e2b": ("google/gemma-4-E2B-it", "models/gemma-4-E2B-it"),
    "phobert": ("vinai/phobert-base-v2", "models/phobert-base-v2"),
    "xlmr": ("FacebookAI/xlm-roberta-base", "models/xlm-roberta-base"),
}


def model_ready(local_dir):
    p = Path(local_dir)
    if not p.exists():
        return False
    markers = ["config.json", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model", "vocab.txt"]
    return any((p / marker).exists() for marker in markers)


def download_one(name, token=None):
    if name not in MODELS:
        raise KeyError(f"Unknown model key: {name}")
    repo_id, local_dir = MODELS[name]
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
