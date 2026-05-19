import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_environment


def git_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_json_if_exists(path):
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def flag_enabled(name):
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/experiment_manifest.yaml")
    parser.add_argument("--model_key", default="vistral")
    parser.add_argument("--model_dir", default="models/Vistral-7B-Chat")
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--config_json", default="results/prediction_config.json")
    parser.add_argument("--out", default="results/paper_evidence/run_metadata.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_known_public_split_leakage", action="store_true")
    args = parser.parse_args()

    import torch

    manifest = verify_environment.load_manifest(args.manifest)
    model_entry = manifest["models"][args.model_key]
    config = load_json_if_exists(args.config_json) or {}
    leakage_override = bool(args.allow_known_public_split_leakage or flag_enabled("ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE") or config.get("leakage_override"))
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_key": args.model_key,
        "model_id": model_entry["hf_id"],
        "model_dir": args.model_dir,
        "adapter_path": args.adapter_dir or config.get("adapter_dir"),
        "dtype": config.get("dtype", manifest["runtime"]["torch_dtype"]),
        "seed": config.get("seed", args.seed),
        "generation_parameters": config.get("generation", {}),
        "quantization": config.get("quantization", manifest["runtime"]["quantization"]),
        "manifest": args.manifest,
        "prediction_config_path": args.config_json,
        "leakage_override": leakage_override,
        "challenge_style_evaluation": leakage_override,
        "leakage_report": manifest.get("artifacts", {}).get("leakage_report_md", "results/paper_evidence/leakage_report.md") if leakage_override else None,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {out}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
