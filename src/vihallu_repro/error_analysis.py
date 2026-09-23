from __future__ import annotations

import re

import pandas as pd

YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
NEGATION_TERMS = {"không", "chưa", "không phải", "sai", "không có"}


def suggest_category(row: pd.Series) -> str:
    context = str(row.get("context", "")).lower()
    response = str(row.get("response", "")).lower()
    context_numbers = set(NUMBER_RE.findall(context))
    response_numbers = set(NUMBER_RE.findall(response))
    if response_numbers - context_numbers:
        if any(YEAR_RE.fullmatch(value) for value in response_numbers - context_numbers):
            return "temporal_or_date_error"
        return "numerical_or_unsupported_value"
    response_negation = any(term in response for term in NEGATION_TERMS)
    context_negation = any(term in context for term in NEGATION_TERMS)
    if response_negation != context_negation:
        return "negation_or_contradiction"
    gold = str(row.get("label", ""))
    pred = str(row.get("predict_label", ""))
    if {gold, pred} == {"intrinsic", "extrinsic"}:
        return "intrinsic_extrinsic_boundary"
    if pred == "no" and gold != "no":
        return "missed_hallucination"
    if gold == "no" and pred != "no":
        return "false_hallucination_alarm"
    return "other_requires_manual_review"
