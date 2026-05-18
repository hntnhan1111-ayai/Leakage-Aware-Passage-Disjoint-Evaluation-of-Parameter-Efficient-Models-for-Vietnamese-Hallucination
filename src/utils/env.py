import os
from pathlib import Path

from dotenv import load_dotenv


def load_hf_token(required=True):
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
    token = os.getenv("HF_TOKEN")
    if required and not token:
        raise RuntimeError("HF_TOKEN is required. Add HF_TOKEN to .env or export it in the shell.")
    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    return token


def assert_env_key_exists(path=".env", key="HF_TOKEN"):
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"Missing {path}. Create it with {key} before running.")
    keys = []
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.append(stripped.split("=", 1)[0].strip())
    if key not in keys:
        raise RuntimeError(f"Missing {key} key in {path}")
    return True
