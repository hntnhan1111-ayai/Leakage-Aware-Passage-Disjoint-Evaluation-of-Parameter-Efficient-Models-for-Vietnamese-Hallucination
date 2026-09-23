from __future__ import annotations

import numpy as np
import pandas as pd

PROMPT_TEMPLATE_VERSION = "vihallu_strict_v1"
ABLATION_CONDITIONS = ["full", "no_prompt", "response_only", "shuffled_context"]


def build_prompt(row: dict[str, object], condition: str = "full") -> str:
    if condition not in ABLATION_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    context = str(row.get("context", "")) if condition in {"full", "no_prompt", "shuffled_context"} else "[REMOVED]"
    user_prompt = str(row.get("prompt", "")) if condition in {"full", "shuffled_context"} else "[REMOVED]"
    response = str(row.get("response", ""))
    return (
        "You are a strict Vietnamese hallucination detection classifier.\n\n"
        "Task:\n"
        "Given CONTEXT, USER_PROMPT, and MODEL_RESPONSE, classify MODEL_RESPONSE by faithfulness to CONTEXT.\n\n"
        "Labels:\n"
        "no = MODEL_RESPONSE is fully supported by CONTEXT.\n"
        "intrinsic = MODEL_RESPONSE contradicts or distorts information in CONTEXT.\n"
        "extrinsic = MODEL_RESPONSE adds information not supported by CONTEXT.\n\n"
        "Rules:\n"
        "Return exactly one label.\n"
        "Allowed outputs: no, intrinsic, extrinsic.\n"
        "Do not explain.\n"
        "Do not copy CONTEXT.\n"
        "Do not answer USER_PROMPT.\n"
        "Do not add punctuation.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER_PROMPT:\n{user_prompt}\n\n"
        f"MODEL_RESPONSE:\n{response}\n\n"
        "Label:"
    )


def make_ablation_frame(df: pd.DataFrame, condition: str, seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    if condition == "shuffled_context":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(out))
        if len(out) > 1:
            for _ in range(100):
                if np.all(order != np.arange(len(out))):
                    break
                order = rng.permutation(len(out))
            if np.any(order == np.arange(len(out))):
                order = np.roll(np.arange(len(out)), 1)
        out["original_context"] = out["context"]
        out["context"] = out.iloc[order]["context"].to_numpy()
        out["shuffled_from_id"] = out.iloc[order]["id"].astype(str).to_numpy()
    return out
