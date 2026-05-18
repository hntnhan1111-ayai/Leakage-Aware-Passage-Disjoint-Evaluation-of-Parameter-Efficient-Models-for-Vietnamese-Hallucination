import argparse
from pathlib import Path
import re
import zipfile


EXTENSIONS = {".py", ".md", ".tex", ".yaml", ".yml", ".json", ".sh", ".toml", ".rules", ".gitignore", ".txt"}
EXCLUDED_PARTS = {".git", ".venv", "models", "checkpoints", "results", "__pycache__", "adapters", "outputs", "wandb"}
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9_\-]{8,}")
LORA_R_RE = re.compile(r"['\"]r['\"]\s*:\s*\d+|\br\s*=\s*\d+|\$r\s*=\s*\d+\$")
SAMPLE_FRAC_RE = re.compile(r"sample\s*\([^\n]*frac\s*=\s*[^,)]+")
GEN_RE = re.compile(r"do_sample\s*=|temperature\s*=|top_p\s*=|top_k\s*=|max_new_tokens\s*=")
LABEL_SMOOTH_RE = re.compile(r"label_smoothing_factor")
SUBMISSION_RE = re.compile(r"final_submission_scratch\.csv|predict_label")
MACRO_F1_RE = re.compile(r"f1_score\s*\([^\n]*average\s*=\s*['\"]macro['\"]")


def is_included(path):
    if path.name == ".gitignore":
        return True
    if path.suffix not in EXTENSIONS:
        return False
    return not any(part in EXCLUDED_PARTS for part in path.parts)


def mask(line):
    return TOKEN_RE.sub("[MASKED_HF_TOKEN]", line.strip())


def collect():
    data = {
        "hardcoded_tokens": [],
        "lora_r_values": [],
        "sample_fractions": [],
        "macro_f1_without_full_reports": [],
        "generation_settings": [],
        "label_smoothing": [],
        "submission_outputs": [],
    }
    for path in sorted(Path(".").rglob("*")):
        if not path.is_file() or not is_included(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        has_report = "classification_report" in text
        has_confusion = "confusion_matrix" in text
        for lineno, line in enumerate(text.splitlines(), 1):
            item = f"{path}:{lineno}: {mask(line)}"
            if TOKEN_RE.search(line):
                data["hardcoded_tokens"].append(item)
            if LORA_R_RE.search(line):
                data["lora_r_values"].append(item)
            if SAMPLE_FRAC_RE.search(line):
                data["sample_fractions"].append(item)
            if MACRO_F1_RE.search(line) and not (has_report and has_confusion):
                data["macro_f1_without_full_reports"].append(item)
            if GEN_RE.search(line):
                data["generation_settings"].append(item)
            if LABEL_SMOOTH_RE.search(line):
                data["label_smoothing"].append(item)
            if SUBMISSION_RE.search(line):
                data["submission_outputs"].append(item)
    for path in sorted(Path(".").rglob("*.zip")):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        try:
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    content = z.read(info.filename)
                    if TOKEN_RE.search(content.decode("utf-8", errors="ignore")):
                        data["hardcoded_tokens"].append(f"{path}:{info.filename}: [MASKED_HF_TOKEN]")
        except zipfile.BadZipFile:
            continue
    return data


def write_section(lines, title, items):
    lines.append(f"## {title}")
    lines.append("")
    if items:
        for item in items:
            lines.append(f"* `{item}`")
    else:
        lines.append("* None found.")
    lines.append("")


def write_report(data, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Code-Paper Consistency Audit", ""]
    lines.append("All file references use exact repository paths and line numbers.")
    lines.append("")
    write_section(lines, "Hardcoded Hugging Face Token Patterns", data["hardcoded_tokens"])
    write_section(lines, "LoRA Rank Claims and Config Values", data["lora_r_values"])
    write_section(lines, "Evaluation Sampling Fractions", data["sample_fractions"])
    write_section(lines, "Macro-F1 Without Classification Report and Confusion Matrix", data["macro_f1_without_full_reports"])
    write_section(lines, "Generation Settings", data["generation_settings"])
    write_section(lines, "Label Smoothing Settings", data["label_smoothing"])
    write_section(lines, "Prediction Output Names and Columns", data["submission_outputs"])
    out.write_text("\n".join(lines), encoding="utf-8")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {out}")
    return out


def write_secret_report(data, out_path):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Secret Scan Report", ""]
    if data["hardcoded_tokens"]:
        lines.append("Result: FAIL")
        lines.append("")
        for item in data["hardcoded_tokens"]:
            lines.append(f"* `{item}`")
    else:
        lines.append("Result: PASS")
        lines.append("")
        lines.append("No hardcoded Hugging Face token patterns were found outside `.env`.")
    out.write_text("\n".join(lines), encoding="utf-8")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/paper_evidence/code_paper_consistency_audit.md")
    parser.add_argument("--secret-out", default="results/paper_evidence/secret_scan_report.md")
    parser.add_argument("--fail-on-secret", action="store_true")
    args = parser.parse_args()
    data = collect()
    out = write_report(data, args.out)
    secret_out = write_secret_report(data, args.secret_out)
    print(f"Wrote {out}")
    print(f"Wrote {secret_out}")
    if args.fail_on_secret and data["hardcoded_tokens"]:
        raise SystemExit("Hardcoded Hugging Face token pattern found outside .env. See masked audit report.")


if __name__ == "__main__":
    main()
