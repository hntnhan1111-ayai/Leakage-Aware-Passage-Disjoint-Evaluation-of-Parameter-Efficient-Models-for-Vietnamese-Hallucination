#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

SMOKE_TEST=0
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
for arg in "$@"; do
  case "$arg" in
    --smoke-test)
      SMOKE_TEST=1
      PRECHECK_ONLY=1
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 2
      ;;
  esac
done

mkdir -p results/paper_evidence docs models checkpoints adapters outputs

MANIFEST="${EXPERIMENT_MANIFEST:-configs/experiment_manifest.yaml}"
SEED="${SEED:-42}"
GOLD_CSV="${GOLD_CSV:-vihallu-test.csv}"
MODEL_DIR="${MODEL_DIR:-models/Vistral-7B-Chat}"
MODEL_KEY="${MODEL_KEY:-vistral}"
OUT_CSV="${OUT_CSV:-results/predictions.csv}"
CONFIG_JSON="${CONFIG_JSON:-results/prediction_config.json}"
MALFORMED_CSV="${MALFORMED_CSV:-results/paper_evidence/malformed_predictions.csv}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
PRED="${PRED_CSV:-}"

echo "Target execution contract"
echo "manifest=$MANIFEST"
echo "seed=$SEED"
echo "gold_csv=$GOLD_CSV"
echo "model_key=$MODEL_KEY"
echo "model_dir=$MODEL_DIR"
echo "out_csv=$OUT_CSV"
echo "config_json=$CONFIG_JSON"
echo "malformed_csv=$MALFORMED_CSV"
echo "max_new_tokens=$MAX_NEW_TOKENS"
echo "temperature=$TEMPERATURE"
echo "top_p=$TOP_P"
echo "precheck_only=$PRECHECK_ONLY"
echo "run_download_models=${RUN_DOWNLOAD_MODELS:-0}"
echo "run_generate_predictions=${RUN_GENERATE_PREDICTIONS:-0}"
echo "run_qwen3_baseline=${RUN_QWEN3_BASELINE:-0}"

python3 -m compileall src scripts

PREFLIGHT_ARGS=(
  --manifest "$MANIFEST"
  --model_key "$MODEL_KEY"
  --model_dir "$MODEL_DIR"
  --gold_csv "$GOLD_CSV"
  --out_dir results/paper_evidence
  --pred_out_csv "$OUT_CSV"
  --config_json "$CONFIG_JSON"
  --summary_json results/preflight_summary.json
  --require_active_env
  --require_hf_token
  --require_cuda
  --require_bf16
  --require_bitsandbytes
)
if [ -n "${ADAPTER_DIR:-}" ]; then
  PREFLIGHT_ARGS+=(--adapter_dir "$ADAPTER_DIR")
fi

run_preflight() {
  python3 scripts/preflight_target_run.py "${PREFLIGHT_ARGS[@]}" "$@"
}

GEN_ARGS=(
  --manifest "$MANIFEST"
  --model_key "$MODEL_KEY"
  --gold_csv "$GOLD_CSV"
  --out_csv "$OUT_CSV"
  --config_json "$CONFIG_JSON"
  --model_dir "$MODEL_DIR"
  --malformed_csv "$MALFORMED_CSV"
  --seed "$SEED"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
)
if [ -n "${ADAPTER_DIR:-}" ]; then
  GEN_ARGS+=(--adapter_dir "$ADAPTER_DIR")
fi
if [ "${NO_QUANT:-0}" = "1" ]; then
  GEN_ARGS+=(--no_quant)
fi

if [ -z "$PRED" ] && [ -f "final_submission_scratch.csv" ]; then
  PRED="final_submission_scratch.csv"
elif [ -z "$PRED" ] && [ -f "results/predictions.csv" ]; then
  PRED="results/predictions.csv"
elif [ -z "$PRED" ] && [ -f "results/paper_evidence/predictions.csv" ]; then
  PRED="results/paper_evidence/predictions.csv"
fi

if [ "$PRECHECK_ONLY" = "1" ]; then
  if [ -n "$PRED" ]; then
    run_preflight --precheck_only --pred_csv "$PRED"
    python3 scripts/build_paper_evidence.py --validate-only --gold_csv "$GOLD_CSV" --pred_csv "$PRED" --label_col label --pred_col predict_label --seed "$SEED"
  elif [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ]; then
    run_preflight --precheck_only --require_model_dir --require_adapter --malformed_csv "$MALFORMED_CSV"
    python3 scripts/generate_predictions_current_best.py --dry-run "${GEN_ARGS[@]}" --require-adapter --require-model-dir
  else
    echo "No prediction CSV found and generation is disabled."
    echo "Set PRED_CSV or RUN_GENERATE_PREDICTIONS=1 for precheck validation."
    exit 1
  fi
  echo "Target precheck complete. No models were loaded and no inference was run."
  exit 0
fi

if [ -n "$PRED" ]; then
  run_preflight --pred_csv "$PRED"
elif [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ]; then
  if [ "${RUN_DOWNLOAD_MODELS:-0}" = "1" ]; then
    run_preflight --require_adapter --malformed_csv "$MALFORMED_CSV"
  else
    run_preflight --require_model_dir --require_adapter --malformed_csv "$MALFORMED_CSV"
  fi
else
  echo "No prediction CSV found and generation is disabled."
  echo "Set PRED_CSV or RUN_GENERATE_PREDICTIONS=1."
  exit 1
fi

python3 scripts/audit_code_paper_consistency.py --out results/paper_evidence/code_paper_consistency_audit.md --fail-on-secret

if [ "${RUN_DOWNLOAD_MODELS:-0}" = "1" ]; then
  python3 scripts/download_models.py
fi

if [ -z "$PRED" ] && [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ]; then
  run_preflight --require_model_dir --require_adapter --malformed_csv "$MALFORMED_CSV"
  python3 scripts/generate_predictions_current_best.py "${GEN_ARGS[@]}"
  PRED="$OUT_CSV"
fi

if [ -z "$PRED" ]; then
  echo "No prediction CSV found and generation is disabled."
  echo "Next command options:"
  echo "  PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh"
  echo "  RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 ADAPTER_DIR=adapters/current_best PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh"
  echo "  RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 ADAPTER_DIR=adapters/current_best bash scripts/run_target_rtx4090_full_pipeline.sh"
  exit 1
fi

python3 scripts/build_paper_evidence.py --gold_csv "$GOLD_CSV" --pred_csv "$PRED" --out_dir results/paper_evidence --label_col label --pred_col predict_label --seed "$SEED" --malformed_csv "$MALFORMED_CSV"
python3 scripts/write_run_metadata.py --manifest "$MANIFEST" --model_key "$MODEL_KEY" --model_dir "$MODEL_DIR" --config_json "$CONFIG_JSON" --adapter_dir "${ADAPTER_DIR:-}" --seed "$SEED" --out results/paper_evidence/run_metadata.json

if [ "${RUN_QWEN3_BASELINE:-0}" = "1" ]; then
  python3 scripts/download_models.py --include-modern
  RUN_QWEN3_BASELINE=1 python3 scripts/run_optional_qwen3_prompt_baseline.py --gold_csv "$GOLD_CSV" --out_dir results/qwen3_prompt_baseline --seed "$SEED"
fi

python3 -c 'from src.utils.io import assert_nonempty_files; required=["results/paper_evidence/malformed_predictions.csv","results/paper_evidence/predictions_merged.csv","results/paper_evidence/classification_report.csv","results/paper_evidence/classification_report.json","results/paper_evidence/confusion_matrix.csv","results/paper_evidence/confusion_matrix.png","results/paper_evidence/wrong_predictions.csv","results/paper_evidence/selected_error_cases.md","results/paper_evidence/summary_metrics.json","results/paper_evidence/latency_summary.csv","results/paper_evidence/latency_summary.json","results/paper_evidence/run_metadata.json","results/paper_evidence/code_paper_consistency_audit.md","results/paper_evidence/secret_scan_report.md","results/paper_evidence/validation_report.md"]; assert_nonempty_files(required); print("All required target artifacts exist:"); [print(item) for item in required]'

UTC_NOW="$(python3 -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())')"
mkdir -p docs
printf '\n## Target Pipeline Run\n- UTC time: %s\n- Command: bash scripts/run_target_rtx4090_full_pipeline.sh\n- Outputs: results/paper_evidence\n' "$UTC_NOW" >> docs/codex_memory_bank.md
