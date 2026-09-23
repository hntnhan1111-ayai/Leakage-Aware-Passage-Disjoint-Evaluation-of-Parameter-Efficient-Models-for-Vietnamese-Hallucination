#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="*", default=["phobert", "xlmr", "vistral_peft", "qwen35_peft", "gemma4_peft"])
    parser.add_argument("--vncorenlp-dir", default="models/vncorenlp")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug-limit", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    for model_key in args.models:
        family = config["models"][model_key]["family"]
        if family == "encoder":
            command = [sys.executable, "scripts/10_train_encoder.py", "--model-key", model_key, "--seed", str(args.seed), "--vncorenlp-dir", args.vncorenlp_dir]
        else:
            command = [sys.executable, "scripts/11_train_peft.py", "--model-key", model_key, "--seed", str(args.seed)]
        if args.force:
            command.append("--force")
        if args.debug_limit:
            command.extend(["--debug-limit", str(args.debug_limit)])
        run(command)
    run([sys.executable, "scripts/13_collect_main_results.py", "--seed", str(args.seed)])


if __name__ == "__main__":
    main()
