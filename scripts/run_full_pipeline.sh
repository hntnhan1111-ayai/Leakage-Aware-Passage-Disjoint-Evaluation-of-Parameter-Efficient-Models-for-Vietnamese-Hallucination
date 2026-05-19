#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "Local-safe orchestration only."
echo "Do not run model download, inference, training, or GPU workloads on this local machine."
echo "For the RTX4090 run, use: bash scripts/run_all_e2e_rtx4090.sh"

python3 -m compileall src scripts
python3 scripts/audit_code_paper_consistency.py --out results/paper_evidence/code_paper_consistency_audit.md --fail-on-secret
python3 scripts/build_paper_evidence.py --help >/dev/null
python3 scripts/train_current_best.py --help >/dev/null
python3 scripts/generate_predictions_current_best.py --help >/dev/null
python3 scripts/run_baselines_e2e.py --help >/dev/null
python3 scripts/download_models.py --help >/dev/null
python3 scripts/update_paper_from_evidence.py --help >/dev/null
bash -n scripts/run_all_e2e_rtx4090.sh
bash -n scripts/run_target_rtx4090_full_pipeline.sh

echo "Local-safe validation complete."
