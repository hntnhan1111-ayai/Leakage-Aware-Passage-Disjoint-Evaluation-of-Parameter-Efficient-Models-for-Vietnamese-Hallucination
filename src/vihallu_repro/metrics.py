from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from . import LABELS


def compute_metrics(y_true, y_pred) -> dict[str, object]:
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_metrics(predictions: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(predictions["label"], predictions["predict_label"])
    (output_dir / "summary_metrics.json").write_text(
        json.dumps({key: value for key, value in metrics.items() if key not in {"classification_report", "confusion_matrix"}}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(metrics["classification_report"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(metrics["classification_report"]).T.to_csv(output_dir / "classification_report.csv")
    cm = np.asarray(metrics["confusion_matrix"], dtype=int)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(output_dir / "confusion_matrix.csv")

    row_totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(normalized, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            ax.text(col, row, f"{cm[row, col]}\n{normalized[row, col]:.1%}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Row-normalized percentage")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)
    return metrics
