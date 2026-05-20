import argparse
import json
from pathlib import Path


LABELS = ["no", "intrinsic", "extrinsic"]
MODEL_LABELS = {
    "vistral": "Vistral PEFT",
    "phobert": "PhoBERT FT",
    "xlmr": "XLM-R FT",
    "qwen35_4b": "Qwen3.5 Zero-shot",
    "qwen35_4b_peft": "Qwen3.5 PEFT",
    "gemma4_e2b_it": "Gemma Zero-shot",
    "gemma4_e2b_it_peft": "Gemma PEFT",
}


def load_summary(path):
    import pandas as pd

    df = pd.read_csv(path)
    df = df[df["status"].astype(str) == "completed"].copy()
    for col in ["accuracy", "macro_f1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["accuracy", "macro_f1"])
    if len(df) == 0:
        raise RuntimeError(f"No completed rows with metrics in {path}")
    order = ["vistral", "phobert", "xlmr", "qwen35_4b", "qwen35_4b_peft", "gemma4_e2b_it", "gemma4_e2b_it_peft"]
    df["order"] = df["model_key"].map({key: index for index, key in enumerate(order)}).fillna(999)
    return df.sort_values(["order", "model_key"]).drop(columns=["order"])


def display_name(model_key):
    return MODEL_LABELS.get(str(model_key), str(model_key))


def save_figure(fig, out_dir, stem):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / f"{stem}.pdf"
    png = out / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    for path in [pdf, png]:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty figure: {path}")


def fig_main_results_bar(summary, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    names = [display_name(key) for key in summary["model_key"]]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - width / 2, summary["macro_f1"], width, label="Macro-F1", color="#2f6f9f")
    ax.bar(x + width / 2, summary["accuracy"], width, label="Accuracy", color="#8a9a5b")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title("Overall performance on the ViHallu benchmark test split")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_main_results_bar")
    plt.close(fig)


def load_class_f1(summary):
    import pandas as pd

    rows = []
    for item in summary.to_dict("records"):
        report_path = Path(str(item["artifact_dir"])) / "classification_report.json"
        if not report_path.exists():
            report_csv = Path(str(item["artifact_dir"])) / "classification_report.csv"
            if not report_csv.exists():
                continue
            report_df = pd.read_csv(report_csv, index_col=0)
            values = {label: float(report_df.loc[label, "f1-score"]) for label in LABELS}
        else:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            values = {label: float(report[label]["f1-score"]) for label in LABELS}
        values["model"] = display_name(item["model_key"])
        rows.append(values)
    if not rows:
        raise RuntimeError("No per-model classification reports found")
    return pd.DataFrame(rows).set_index("model")[LABELS]


def fig_per_class_f1_heatmap(summary, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    data = load_class_f1(summary)
    fig_height = max(3.8, 0.55 * len(data) + 1.4)
    fig, ax = plt.subplots(figsize=(6.8, fig_height))
    im = ax.imshow(data.values, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "#202020")
    ax.set_title("Per-class F1 on the ViHallu benchmark test split")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F1")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_per_class_f1_heatmap")
    plt.close(fig)


def fig_vistral_confusion_matrix(summary, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    row = summary[summary["model_key"].astype(str) == "vistral"]
    if len(row) == 0:
        raise RuntimeError("Missing completed vistral row in model comparison summary")
    path = Path(str(row.iloc[0]["artifact_dir"])) / "confusion_matrix.csv"
    cm = pd.read_csv(path, index_col=0).loc[LABELS, LABELS]
    values = cm.values.astype(float)
    row_sums = values.sum(axis=1, keepdims=True)
    norm = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title("Vistral PEFT confusion matrix on the ViHallu benchmark test split")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{int(values[i, j])}\n{norm[i, j] * 100:.1f}%", ha="center", va="center", color="white" if norm[i, j] > 0.55 else "#202020")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized percentage")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_vistral_confusion_matrix")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/model_comparison/model_comparison_summary.csv")
    parser.add_argument("--out-dir", default="results/paper_figures")
    args = parser.parse_args()
    summary = load_summary(args.summary)
    fig_main_results_bar(summary, args.out_dir)
    fig_per_class_f1_heatmap(summary, args.out_dir)
    fig_vistral_confusion_matrix(summary, args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
