#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official VnCoreNLP components used for PhoBERT word segmentation.")
    parser.add_argument("--output-dir", default="models/vncorenlp")
    args = parser.parse_args()
    import py_vncorenlp

    path = Path(args.output_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    py_vncorenlp.download_model(save_dir=str(path))
    print(path)


if __name__ == "__main__":
    main()
