#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review_csv")
    args = parser.parse_args()
    path = Path(args.review_csv)
    frame = pd.read_csv(path).fillna("")
    missing = frame["human_category"].astype(str).str.strip().eq("")
    if missing.any():
        raise ValueError(f"{int(missing.sum())} rows still have empty human_category")
    counts = frame.groupby(["error_pair", "human_category"]).size().rename("count").reset_index().sort_values("count", ascending=False)
    counts.to_csv(path.with_name("validated_error_category_counts.csv"), index=False)
    lines = ["# Validated qualitative error analysis", "", counts.to_markdown(index=False), ""]
    path.with_name("validated_error_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
