from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from . import LABELS


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def paired_bootstrap(
    y_true,
    predictions: dict[str, np.ndarray],
    *,
    iterations: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, object]:
    y_true = np.asarray(y_true)
    names = list(predictions)
    pred = {name: np.asarray(values) for name, values in predictions.items()}
    if any(len(values) != len(y_true) for values in pred.values()):
        raise ValueError("Prediction arrays must have the same length as y_true")
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(iterations, dtype=float) for name in names}
    differences: dict[str, np.ndarray] = {}
    for left_idx, left in enumerate(names):
        for right in names[left_idx + 1 :]:
            differences[f"{left}__minus__{right}"] = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        index = rng.integers(0, len(y_true), size=len(y_true))
        true_sample = y_true[index]
        values = {}
        for name in names:
            values[name] = macro_f1(true_sample, pred[name][index])
            samples[name][iteration] = values[name]
        for key in differences:
            left, right = key.split("__minus__")
            differences[key][iteration] = values[left] - values[right]
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, object] = {"iterations": iterations, "seed": seed, "confidence": confidence, "models": {}, "differences": {}}
    for name in names:
        point = macro_f1(y_true, pred[name])
        low, high = np.quantile(samples[name], [alpha, 1.0 - alpha])
        result["models"][name] = {"point": point, "ci_low": float(low), "ci_high": float(high)}
    for key, values in differences.items():
        low, high = np.quantile(values, [alpha, 1.0 - alpha])
        left, right = key.split("__minus__")
        point_difference = macro_f1(y_true, pred[left]) - macro_f1(y_true, pred[right])
        result["differences"][key] = {
            "point": float(point_difference),
            "ci_low": float(low),
            "ci_high": float(high),
            "two_sided_bootstrap_p": float(min(1.0, 2.0 * min((values <= 0).mean(), (values >= 0).mean()))),
        }
    return result


def mcnemar_exact(y_true, pred_a, pred_b) -> dict[str, object]:
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    a_correct_b_wrong = int(np.sum(correct_a & ~correct_b))
    a_wrong_b_correct = int(np.sum(~correct_a & correct_b))
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(min(a_correct_b_wrong, a_wrong_b_correct), discordant, p=0.5, alternative="two-sided").pvalue)
    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "discordant": discordant,
        "exact_p_value": p_value,
    }
