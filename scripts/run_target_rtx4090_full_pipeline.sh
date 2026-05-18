#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

mkdir -p results/paper_evidence docs models checkpoints adapters outputs

python3 - <<'PY'
from pathlib import Path
p = Path(".env")
if not p.exists():
    raise SystemExit("Missing .env. Create it with: printf 'HF_TOKEN=your_token_here\\n' > .env")
keys = []
for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and "=" in stripped:
        keys.append(stripped.split("=", 1)[0].strip())
if "HF_TOKEN" not in keys:
    raise SystemExit("Missing HF_TOKEN key in .env")
print("HF_TOKEN key exists")
PY

python3 - <<'PY'
import importlib.util
import os
import sys
if not (os.getenv("VIRTUAL_ENV") or os.getenv("CONDA_PREFIX")):
    raise SystemExit("No active environment. Run: uv venv --python 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt")
missing = []
for name in ["torch", "transformers", "datasets", "accelerate", "peft", "trl", "bitsandbytes", "sklearn", "pandas", "numpy", "matplotlib", "tqdm", "huggingface_hub", "dotenv", "yaml"]:
    if importlib.util.find_spec(name) is None:
        missing.append(name)
if missing:
    raise SystemExit("Missing Python packages. Run: uv pip install -r requirements.txt. Missing: " + ", ".join(missing))
print("Python environment validated")
PY

python3 -m compileall src scripts
python3 scripts/audit_code_paper_consistency.py --out results/paper_evidence/code_paper_consistency_audit.md --fail-on-secret

if [ "${RUN_DOWNLOAD_MODELS:-0}" = "1" ]; then
  python3 scripts/download_models.py
fi

PRED="${PRED_CSV:-}"
if [ -z "$PRED" ] && [ -f "final_submission_scratch.csv" ]; then
  PRED="final_submission_scratch.csv"
elif [ -z "$PRED" ] && [ -f "results/predictions.csv" ]; then
  PRED="results/predictions.csv"
elif [ -z "$PRED" ] && [ -f "results/paper_evidence/predictions.csv" ]; then
  PRED="results/paper_evidence/predictions.csv"
fi

if [ -z "$PRED" ] && [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ]; then
  python3 scripts/generate_predictions_current_best.py --gold_csv vihallu-test.csv --out_csv results/predictions.csv --config_json results/prediction_config.json --seed 42
  PRED="results/predictions.csv"
fi

if [ -z "$PRED" ]; then
  echo "No prediction CSV found and generation is disabled."
  echo "Next command options:"
  echo "  PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh"
  echo "  RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 bash scripts/run_target_rtx4090_full_pipeline.sh"
  exit 1
fi

python3 scripts/build_paper_evidence.py --gold_csv vihallu-test.csv --pred_csv "$PRED" --out_dir results/paper_evidence --label_col label --pred_col predict_label --seed 42

if [ "${RUN_QWEN3_BASELINE:-0}" = "1" ]; then
  python3 scripts/download_models.py --include-modern
  RUN_QWEN3_BASELINE=1 python3 scripts/run_optional_qwen3_prompt_baseline.py --gold_csv vihallu-test.csv --out_dir results/qwen3_prompt_baseline --seed 42
fi

python3 - <<'PY'
from src.utils.io import assert_nonempty_files
required = [
    "results/paper_evidence/predictions_merged.csv",
    "results/paper_evidence/classification_report.csv",
    "results/paper_evidence/classification_report.json",
    "results/paper_evidence/confusion_matrix.csv",
    "results/paper_evidence/confusion_matrix.png",
    "results/paper_evidence/wrong_predictions.csv",
    "results/paper_evidence/selected_error_cases.md",
    "results/paper_evidence/summary_metrics.json",
    "results/paper_evidence/latency_summary.csv",
    "results/paper_evidence/latency_summary.json",
    "results/paper_evidence/code_paper_consistency_audit.md",
    "results/paper_evidence/secret_scan_report.md",
    "results/paper_evidence/validation_report.md",
]
assert_nonempty_files(required)
print("All required target artifacts exist:")
for item in required:
    print(item)
PY

python3 - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
p = Path("docs/codex_memory_bank.md")
p.parent.mkdir(parents=True, exist_ok=True)
with p.open("a", encoding="utf-8") as f:
    f.write("\n## Target Pipeline Run\n")
    f.write(f"- UTC time: {datetime.now(timezone.utc).isoformat()}\n")
    f.write("- Command: bash scripts/run_target_rtx4090_full_pipeline.sh\n")
    f.write("- Outputs: results/paper_evidence\n")
PY
