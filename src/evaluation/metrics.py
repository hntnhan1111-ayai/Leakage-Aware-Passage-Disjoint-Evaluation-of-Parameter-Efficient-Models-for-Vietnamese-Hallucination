from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


LABELS = ["no", "intrinsic", "extrinsic"]


def validate_labels(values, name):
    bad = sorted(set(pd.Series(values).dropna().astype(str)) - set(LABELS))
    if bad:
        raise ValueError(f"{name} contains invalid labels: {bad}")
    return True


def compute_and_save(y_true, y_pred, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    validate_labels(y_true, "y_true")
    validate_labels(y_pred, "y_pred")
    report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out / "classification_report.csv")
    with (out / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_df.to_csv(out / "confusion_matrix.csv")
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=200)
    plt.close(fig)
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "labels": LABELS,
        "classification_report": report,
    }
    with (out / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    required = ["classification_report.csv", "classification_report.json", "confusion_matrix.csv", "confusion_matrix.png", "summary_metrics.json"]
    for name in required:
        p = out / name
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {p}")
    return summary
