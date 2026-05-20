import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = ["no", "intrinsic", "extrinsic"]
MODEL_SPECS = {
    "qwen35_4b_peft": {
        "display": "Qwen3.5 PEFT",
        "title": "Qwen3.5 PEFT Confusion Matrix.",
        "stem": "fig_qwen35_peft_confusion_matrix",
    },
    "gemma4_e2b_it_peft": {
        "display": "Gemma4 PEFT",
        "title": "Gemma4 PEFT Confusion Matrix.",
        "stem": "fig_gemma4_peft_confusion_matrix",
    },
}


def require_file(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty file: {p}")
    return p


def load_confusion_matrix(path):
    df = pd.read_csv(require_file(path), index_col=0)
    missing_rows = [label for label in LABELS if label not in df.index]
    missing_cols = [label for label in LABELS if label not in df.columns]
    if missing_rows or missing_cols:
        raise ValueError(f"{path} missing labels rows={missing_rows} columns={missing_cols}")
    return df.loc[LABELS, LABELS].astype(int)


def save_figure(fig, out_dir, stem):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{stem}.pdf", out / f"{stem}.png"]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=300, bbox_inches="tight")
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty figure: {path}")
        print(f"{path} {path.stat().st_size} bytes")
    return paths


def make_confusion_figure(cm, spec, out_dir):
    values = cm.values.astype(float)
    row_sums = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(5.9, 5.25))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title(spec["title"])
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            value = int(values[i, j])
            percent = normalized[i, j] * 100
            color = "white" if normalized[i, j] > 0.55 else "#202020"
            ax.text(j, i, f"{value}\n{percent:.1f}%", ha="center", va="center", color=color, fontsize=10)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Row-normalized percentage")
    fig.tight_layout()
    save_figure(fig, out_dir, spec["stem"])
    plt.close(fig)


def error_pairs(cm):
    rows = []
    for gold in LABELS:
        for pred in LABELS:
            if gold == pred:
                continue
            rows.append({"error_pair": f"{gold}\u2192{pred}", "count": int(cm.loc[gold, pred])})
    return rows


def make_error_comparison(qwen_cm, gemma_cm, out_dir):
    qwen_errors = {row["error_pair"]: row["count"] for row in error_pairs(qwen_cm)}
    gemma_errors = {row["error_pair"]: row["count"] for row in error_pairs(gemma_cm)}
    pairs = [f"{gold}\u2192{pred}" for gold in LABELS for pred in LABELS if gold != pred]
    data = pd.DataFrame({
        "error_pair": pairs,
        "Qwen3.5 PEFT": [qwen_errors[pair] for pair in pairs],
        "Gemma4 PEFT": [gemma_errors[pair] for pair in pairs],
    })
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "fig_qwen_gemma_error_comparison_data.csv"
    data.to_csv(data_path, index=False)
    if not data_path.exists() or data_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty data file: {data_path}")
    print(f"{data_path} {data_path.stat().st_size} bytes")
    x = np.arange(len(pairs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.bar(x - width / 2, data["Qwen3.5 PEFT"], width, label="Qwen3.5 PEFT", color="#2f6f9f")
    ax.bar(x + width / 2, data["Gemma4 PEFT"], width, label="Gemma4 PEFT", color="#8a9a5b")
    ax.set_ylabel("Off-diagonal error count")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=25, ha="right")
    ax.set_title("Qwen3.5 PEFT vs Gemma4 PEFT Error Comparison")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out, "fig_qwen_gemma_error_comparison")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/model_comparison")
    parser.add_argument("--out-dir", default="results/paper_figures")
    args = parser.parse_args()
    root = Path(args.root)
    matrices = {}
    for model_key, spec in MODEL_SPECS.items():
        cm = load_confusion_matrix(root / model_key / "confusion_matrix.csv")
        matrices[model_key] = cm
        make_confusion_figure(cm, spec, args.out_dir)
    make_error_comparison(matrices["qwen35_4b_peft"], matrices["gemma4_e2b_it_peft"], args.out_dir)


if __name__ == "__main__":
    main()
