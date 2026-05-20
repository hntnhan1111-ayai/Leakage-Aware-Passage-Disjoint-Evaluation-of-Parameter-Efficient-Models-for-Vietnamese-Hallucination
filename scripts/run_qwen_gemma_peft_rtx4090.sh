#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

CONFIG="${BASELINE_CONFIG:-configs/baseline_models.yaml}"
TRAIN_CSV="${TRAIN_CSV:-vihallu-train.csv}"
GOLD_CSV="${GOLD_CSV:-vihallu-test.csv}"
OUT_ROOT="${MODEL_COMPARISON_ROOT:-results/model_comparison}"
SEED="${SEED:-42}"
EPOCHS="${PEFT_EPOCHS:-2}"

echo "Qwen/Gemma supervised PEFT execution contract"
echo "config=$CONFIG"
echo "train_csv=$TRAIN_CSV"
echo "gold_csv=$GOLD_CSV"
echo "out_root=$OUT_ROOT"
echo "seed=$SEED"
echo "epochs=$EPOCHS"
echo "debug_limit=${DEBUG_LIMIT:-}"
echo "force_rerun_model=${FORCE_RERUN_MODEL:-0}"

COMMON_ARGS=(
  --config "$CONFIG"
  --train-csv "$TRAIN_CSV"
  --gold-csv "$GOLD_CSV"
  --out-root "$OUT_ROOT"
  --seed "$SEED"
  --epochs "$EPOCHS"
)

if [ -n "${DEBUG_LIMIT:-}" ]; then
  COMMON_ARGS+=(--debug-limit "$DEBUG_LIMIT")
fi

if [ "${FORCE_RERUN_MODEL:-0}" = "1" ]; then
  COMMON_ARGS+=(--force-rerun-model)
fi

python3 scripts/train_eval_llm_peft_baseline.py --model-key qwen35_4b_peft "${COMMON_ARGS[@]}"
python3 scripts/train_eval_llm_peft_baseline.py --model-key gemma4_e2b_it_peft "${COMMON_ARGS[@]}"

if [ -n "${DEBUG_LIMIT:-}" ]; then
  echo "SKIP_GLOBAL_SUMMARY_REBUILD_FOR_DEBUG_LIMIT"
  exit 0
fi

REBUILD_ARGS=(
  --config "$CONFIG"
  --gold_csv "$GOLD_CSV"
  --train_csv "$TRAIN_CSV"
  --out_root "$OUT_ROOT"
  --seed "$SEED"
  --rebuild-summary-only
  --allow_known_public_split_leakage
)

python3 scripts/run_baselines_e2e.py "${REBUILD_ARGS[@]}"
