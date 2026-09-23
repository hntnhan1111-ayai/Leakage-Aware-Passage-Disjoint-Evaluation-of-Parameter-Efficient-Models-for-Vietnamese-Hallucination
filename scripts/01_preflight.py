#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vihallu_repro import LABELS
from vihallu_repro.data import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-local-models", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    checks = []
    for split in ["train_csv", "dev_csv", "test_csv"]:
        path = ROOT / config["project"][split]
        ok = path.exists()
        detail = {"path": str(path), "exists": ok}
        if ok:
            frame = pd.read_csv(path)
            detail.update({"rows": len(frame), "sha256": sha256_file(path), "labels": sorted(frame["label"].unique().tolist())})
            ok = sorted(frame["label"].unique().tolist()) == sorted(LABELS)
        checks.append({"name": split, "ok": ok, "detail": detail})

    if not args.data_only:
        try:
            import torch

            cuda = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda else None
            torch_version = torch.__version__
        except Exception as exc:
            cuda = False
            gpu_name = None
            torch_version = f"unavailable: {exc}"
        checks.append({"name": "cuda", "ok": cuda or not args.require_cuda, "detail": {"available": cuda, "gpu": gpu_name, "torch": torch_version}})

        for model_key, entry in config["models"].items():
            local = ROOT / entry["local_dir"]
            ok = local.exists() and any(local.iterdir())
            checks.append({"name": f"model:{model_key}", "ok": ok or not args.require_local_models, "detail": {"path": str(local), "ready": ok, "hf_id": entry["hf_id"]}})

    packages = {}
    for package in ["torch", "transformers", "datasets", "accelerate", "peft", "bitsandbytes", "scikit-learn", "pandas", "numpy", "scipy", "pyyaml"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "checks": checks,
        "passed": all(item["ok"] for item in checks),
    }
    output = ROOT / "reports/preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
