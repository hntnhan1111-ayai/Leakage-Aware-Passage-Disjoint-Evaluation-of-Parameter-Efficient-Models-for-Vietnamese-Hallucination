#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate
source scripts/env_rtx4090.sh

mkdir -p logs
MASTER_LOG="logs/resume_p1_p5_$(date +%Y%m%d_%H%M%S).log"
EXIT_FILE="${MASTER_LOG%.log}.exitcode"

exec > >(tee -a "$MASTER_LOG") 2>&1

on_exit() {
    RC=$?
    printf '%s\n' "$RC" > "$EXIT_FILE"

    echo
    echo "============================================================"
    echo "FINISHED_AT=$(date --iso-8601=seconds)"
    echo "EXIT_CODE=$RC"
    echo "MASTER_LOG=$MASTER_LOG"
    echo "============================================================"

    exit "$RC"
}
trap on_exit EXIT

run_stage() {
    local stage="$1"
    shift

    local stage_log="logs/${stage}_$(date +%Y%m%d_%H%M%S).log"

    echo
    echo "############################################################"
    echo "STAGE=$stage"
    echo "STARTED_AT=$(date --iso-8601=seconds)"
    printf 'COMMAND='
    printf ' %q' "$@"
    printf '\n'
    echo "STAGE_LOG=$stage_log"
    echo "############################################################"

    set +e
    "$@" 2>&1 | tee -a "$stage_log"
    local rc="${PIPESTATUS[0]}"
    set -e

    echo "STAGE_EXIT_CODE=$rc"

    if [ "$rc" -ne 0 ]; then
        echo "FAILED_STAGE=$stage"
        exit "$rc"
    fi
}

echo "ROOT=$ROOT"
echo "PYTHON=$(command -v python)"
echo "JAVA_HOME=$JAVA_HOME"
echo "JAVAC=$(command -v javac)"
echo "HF_HOME=$HF_HOME"

python --version
java -version
javac -version

python - <<'PY'
import torch
import bitsandbytes

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("bitsandbytes:", bitsandbytes.__version__)
print("GPU:", torch.cuda.get_device_name(0))
assert torch.cuda.is_available()
assert torch.version.cuda == "12.8"
PY

# P0/data already completed, but verify integrity and model availability.
run_stage preflight \
    python scripts/01_preflight.py \
        --require-cuda \
        --require-local-models

# P1: main results, seed 42.
run_stage p1_seed42 \
    bash scripts/run_p1_main_seed42.sh

# P2 context ablation, P3 statistics, automated P4 queue.
run_stage p2_p4 \
    bash scripts/run_p2_p4.sh

# P5: Qwen3.5 and Gemma4, seeds 42/43/44.
run_stage p5_multiseed \
    bash scripts/run_p5_multiseed.sh

# Generate paper tables when script exists.
if [ -f scripts/60_build_paper_tables.py ]; then
    run_stage paper_tables \
        python scripts/60_build_paper_tables.py
fi

echo
echo "ALL AUTOMATED P1–P5 STAGES COMPLETED"
