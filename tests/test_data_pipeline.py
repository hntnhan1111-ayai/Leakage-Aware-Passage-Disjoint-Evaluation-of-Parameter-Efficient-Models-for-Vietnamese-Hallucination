from pathlib import Path

import pandas as pd

from vihallu_repro.data import add_fingerprints, normalize_text
from vihallu_repro.split import search_balanced_group_split
from vihallu_repro.audit import validate_clean_splits


def test_normalize_text_is_stable():
    assert normalize_text("  Xin   CHÀO\n") == "xin chào"


def test_group_split_is_disjoint_and_complete():
    frame = pd.DataFrame(
        {
            "id": [f"id-{i}" for i in range(30)],
            "context": [f"context {i // 3}" for i in range(30)],
            "prompt": [f"prompt {i}" for i in range(30)],
            "response": [f"response {i}" for i in range(30)],
            "label": ["no", "intrinsic", "extrinsic"] * 10,
        }
    )
    frame = add_fingerprints(frame)
    result = search_balanced_group_split(frame, proportions=(0.7, 0.15, 0.15), base_seed=42, trials=200)
    splits = {name: frame.loc[index].copy() for name, index in result.split_indices.items()}
    assert sum(len(split) for split in splits.values()) == len(frame)
    validation = validate_clean_splits(splits)
    assert validation["valid"]
