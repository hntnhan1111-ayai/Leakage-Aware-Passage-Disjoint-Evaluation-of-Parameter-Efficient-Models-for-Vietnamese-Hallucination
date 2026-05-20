#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

MANIFEST="${EXPERIMENT_MANIFEST:-configs/experiment_manifest.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/baseline_models.yaml}"
SEED="${SEED:-42}"
MODEL_KEY="${MODEL_KEY:-vistral}"
MODEL_DIR="${MODEL_DIR:-models/Vistral-7B-Chat}"
ADAPTER_DIR="${ADAPTER_DIR:-adapters/current_best}"
TRAIN_CSV="${TRAIN_CSV:-vihallu-train.csv}"
GOLD_CSV="${GOLD_CSV:-vihallu-test.csv}"
OUT_CSV="${OUT_CSV:-results/predictions.csv}"
CONFIG_JSON="${CONFIG_JSON:-results/prediction_config.json}"
TRAINING_DIR="${TRAINING_DIR:-results/current_best_training}"
EVIDENCE_DIR="${EVIDENCE_DIR:-results/paper_evidence}"
BASELINE_ROOT="${BASELINE_ROOT:-results/baselines}"
MODEL_COMPARISON_ROOT="${MODEL_COMPARISON_ROOT:-results/model_comparison}"
MALFORMED_CSV="${MALFORMED_CSV:-results/paper_evidence/malformed_predictions.csv}"

LEAKAGE_ARGS=()
if [ "${ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE:-0}" = "1" ]; then
  LEAKAGE_ARGS+=(--allow_known_public_split_leakage)
fi

RESUME_ARGS=()
if [ "${RESUME_PREDICTIONS:-0}" = "1" ]; then
  RESUME_ARGS+=(--resume)
fi

echo "RTX4090 E2E execution contract"
echo "manifest=$MANIFEST"
echo "baseline_config=$BASELINE_CONFIG"
echo "seed=$SEED"
echo "model_key=$MODEL_KEY"
echo "model_dir=$MODEL_DIR"
echo "adapter_dir=$ADAPTER_DIR"
echo "train_csv=$TRAIN_CSV"
echo "gold_csv=$GOLD_CSV"
echo "out_csv=$OUT_CSV"
echo "config_json=$CONFIG_JSON"
echo "training_dir=$TRAINING_DIR"
echo "evidence_dir=$EVIDENCE_DIR"
echo "baseline_root=$BASELINE_ROOT"
echo "model_comparison_root=$MODEL_COMPARISON_ROOT"
echo "run_download_models=${RUN_DOWNLOAD_MODELS:-0}"
echo "run_train_current_best=${RUN_TRAIN_CURRENT_BEST:-0}"
echo "run_generate_predictions=${RUN_GENERATE_PREDICTIONS:-0}"
echo "run_baselines=${RUN_BASELINES:-0}"
echo "full_model_comparison=${FULL_MODEL_COMPARISON:-0}"
echo "strict_baselines=${STRICT_BASELINES:-0}"
echo "allow_known_public_split_leakage=${ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE:-0}"

mkdir -p "$TRAINING_DIR" "$EVIDENCE_DIR" "$BASELINE_ROOT" "$MODEL_COMPARISON_ROOT" adapters models checkpoints outputs docs

if [ -z "${HF_TOKEN:-}" ]; then
  if [ ! -f .env ] || ! grep -qE '^[[:space:]]*HF_TOKEN[[:space:]]*=' .env; then
    echo "HF_TOKEN key is required in .env or the process environment."
    exit 1
  fi
fi

python3 -m compileall src scripts
python3 scripts/verify_environment.py --target-check --require-active-env --require-hf-token --require-cuda --require-bf16 --require-bitsandbytes --dataset-check all --summary-json results/environment_summary.json

python3 scripts/preflight_target_run.py \
  --manifest "$MANIFEST" \
  --model_key "$MODEL_KEY" \
  --model_dir "$MODEL_DIR" \
  --gold_csv "$GOLD_CSV" \
  --out_dir "$EVIDENCE_DIR" \
  --pred_out_csv "$OUT_CSV" \
  --config_json "$CONFIG_JSON" \
  --malformed_csv "$MALFORMED_CSV" \
  --summary_json results/preflight_summary_initial.json \
  --require_active_env \
  --require_hf_token \
  --require_cuda \
  --require_bf16 \
  --require_bitsandbytes \
  "${LEAKAGE_ARGS[@]}"

if [ "${RUN_DOWNLOAD_MODELS:-0}" = "1" ]; then
  echo "RUN_DOWNLOAD_MODELS"
  python3 scripts/download_models.py --config "$BASELINE_CONFIG" --report-json "$MODEL_COMPARISON_ROOT/download_report.json"
else
  echo "SKIP_DOWNLOAD_MODELS"
fi

if [ "${RUN_TRAIN_CURRENT_BEST:-0}" = "1" ]; then
  echo "RUN_TRAIN"
  python3 scripts/train_current_best.py \
    --manifest "$MANIFEST" \
    --model_key "$MODEL_KEY" \
    --model_dir "$MODEL_DIR" \
    --train_csv "$TRAIN_CSV" \
    --output_dir "$TRAINING_DIR" \
    --adapter_dir "$ADAPTER_DIR" \
    --seed "$SEED"
else
  echo "SKIP_TRAIN"
fi

if [ "${RUN_TRAIN_CURRENT_BEST:-0}" = "1" ] || [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ]; then
  python3 -c 'from pathlib import Path; p=Path("'"$ADAPTER_DIR"'"); missing=[]; missing.append(str(p/"adapter_config.json")) if not (p/"adapter_config.json").is_file() else None; weights=[p/"adapter_model.safetensors",p/"adapter_model.bin"]; missing.append("adapter_model.safetensors or adapter_model.bin in "+str(p)) if not any(w.is_file() and w.stat().st_size>0 for w in weights) else None; raise SystemExit("Missing PEFT adapter files: "+", ".join(missing)+"\nRun: RUN_TRAIN_CURRENT_BEST=1 RUN_GENERATE_PREDICTIONS=0 RUN_BASELINES=0 bash scripts/run_all_e2e_rtx4090.sh" if missing else 0)'

  python3 scripts/preflight_target_run.py \
    --manifest "$MANIFEST" \
    --model_key "$MODEL_KEY" \
    --model_dir "$MODEL_DIR" \
    --adapter_dir "$ADAPTER_DIR" \
    --gold_csv "$GOLD_CSV" \
    --out_dir "$EVIDENCE_DIR" \
    --pred_out_csv "$OUT_CSV" \
    --config_json "$CONFIG_JSON" \
    --malformed_csv "$MALFORMED_CSV" \
    --summary_json results/preflight_summary_after_training.json \
    --require_model_dir \
    --require_adapter \
    --require_active_env \
    --require_hf_token \
    --require_cuda \
    --require_bf16 \
    --require_bitsandbytes \
    "${LEAKAGE_ARGS[@]}"
fi

if [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ]; then
  echo "RUN_INFERENCE"
  python3 scripts/generate_predictions_current_best.py \
    --manifest "$MANIFEST" \
    --model_key "$MODEL_KEY" \
    --model_dir "$MODEL_DIR" \
    --adapter_dir "$ADAPTER_DIR" \
    --gold_csv "$GOLD_CSV" \
    --out_csv "$OUT_CSV" \
    --config_json "$CONFIG_JSON" \
    --malformed_csv "$MALFORMED_CSV" \
    --latency_out_dir "$EVIDENCE_DIR" \
    --seed "$SEED" \
    "${RESUME_ARGS[@]}" \
    "${LEAKAGE_ARGS[@]}"
else
  echo "SKIP_MAIN_INFERENCE"
fi

if [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ]; then
  if [ ! -s "$OUT_CSV" ]; then
    echo "Missing or empty prediction CSV: $OUT_CSV"
    exit 1
  fi

  python3 scripts/audit_code_paper_consistency.py --out "$EVIDENCE_DIR/code_paper_consistency_audit.md" --fail-on-secret

  python3 scripts/build_paper_evidence.py \
    --gold_csv "$GOLD_CSV" \
    --pred_csv "$OUT_CSV" \
    --out_dir "$EVIDENCE_DIR" \
    --label_col label \
    --pred_col predict_label \
    --seed "$SEED" \
    --malformed_csv "$MALFORMED_CSV" \
    "${LEAKAGE_ARGS[@]}"

  python3 scripts/write_run_metadata.py \
    --manifest "$MANIFEST" \
    --model_key "$MODEL_KEY" \
    --model_dir "$MODEL_DIR" \
    --config_json "$CONFIG_JSON" \
    --adapter_dir "$ADAPTER_DIR" \
    --seed "$SEED" \
    --out "$EVIDENCE_DIR/run_metadata.json" \
    "${LEAKAGE_ARGS[@]}"
fi

if [ "${RUN_BASELINES:-0}" = "1" ]; then
  echo "RUN_MODEL_COMPARISON"
  BASELINE_ARGS=(
    --config "$BASELINE_CONFIG"
    --gold_csv "$GOLD_CSV"
    --train_csv "$TRAIN_CSV"
    --out_root "$MODEL_COMPARISON_ROOT"
    --seed "$SEED"
  )
  if [ -n "${DEBUG_LIMIT:-}" ]; then
    BASELINE_ARGS+=(--debug-limit "$DEBUG_LIMIT")
  fi
  if [ -n "${ONLY_MODEL_KEY:-}" ]; then
    BASELINE_ARGS+=(--only-model-key "$ONLY_MODEL_KEY")
  fi
  if [ "${FORCE_RERUN_MODEL:-0}" = "1" ]; then
    BASELINE_ARGS+=(--force-rerun-model)
  fi
  if [ "${RUN_DOWNLOAD_MODELS:-0}" = "1" ]; then
    BASELINE_ARGS+=(--download-missing)
  fi
  if [ "${STRICT_BASELINES:-0}" = "1" ]; then
    BASELINE_ARGS+=(--strict)
  fi
  python3 scripts/run_baselines_e2e.py "${BASELINE_ARGS[@]}" "${LEAKAGE_ARGS[@]}" || {
    if [ "${STRICT_BASELINES:-0}" = "1" ]; then
      exit 1
    fi
    echo "One or more optional baselines failed. Main method artifacts will still be validated."
  }
else
  echo "SKIP_BASELINES"
fi

REQUIRED_ARTIFACTS=()
if [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ]; then
  REQUIRED_ARTIFACTS+=(
    "$ADAPTER_DIR/adapter_config.json"
    "$OUT_CSV"
    "$CONFIG_JSON"
    "$EVIDENCE_DIR/predictions_merged.csv"
    "$EVIDENCE_DIR/classification_report.csv"
    "$EVIDENCE_DIR/classification_report.json"
    "$EVIDENCE_DIR/summary_metrics.json"
    "$EVIDENCE_DIR/confusion_matrix.csv"
    "$EVIDENCE_DIR/confusion_matrix.png"
    "$EVIDENCE_DIR/wrong_predictions.csv"
    "$EVIDENCE_DIR/selected_error_cases.md"
    "$EVIDENCE_DIR/malformed_predictions.csv"
    "$EVIDENCE_DIR/latency_summary.csv"
    "$EVIDENCE_DIR/latency_summary.json"
    "$EVIDENCE_DIR/validation_report.md"
    "$EVIDENCE_DIR/run_metadata.json"
  )
fi
if [ "${RUN_TRAIN_CURRENT_BEST:-0}" = "1" ]; then
  REQUIRED_ARTIFACTS+=("$ADAPTER_DIR/adapter_config.json" "$TRAINING_DIR/training_config_resolved.json" "$TRAINING_DIR/train_runtime.json")
fi
if [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ] && [ "${ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE:-0}" = "1" ]; then
  REQUIRED_ARTIFACTS+=("$EVIDENCE_DIR/leakage_report.md")
fi
if [ "${RUN_BASELINES:-0}" = "1" ]; then
  REQUIRED_ARTIFACTS+=(
    "$MODEL_COMPARISON_ROOT/model_comparison_summary.csv"
    "$MODEL_COMPARISON_ROOT/model_comparison_summary.md"
    "$MODEL_COMPARISON_ROOT/model_comparison_status.json"
    "$MODEL_COMPARISON_ROOT/skipped_models.csv"
  )
fi

missing=()
for item in "${REQUIRED_ARTIFACTS[@]}"; do
  if [ ! -s "$item" ]; then
    missing+=("$item")
  fi
done
if [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ] || [ "${RUN_TRAIN_CURRENT_BEST:-0}" = "1" ]; then
if [ ! -s "$ADAPTER_DIR/adapter_model.safetensors" ] && [ ! -s "$ADAPTER_DIR/adapter_model.bin" ]; then
  missing+=("$ADAPTER_DIR/adapter_model.safetensors or $ADAPTER_DIR/adapter_model.bin")
fi
fi
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing required main-method artifacts:"
  printf '%s\n' "${missing[@]}"
  exit 1
fi

echo "Final paper artifact table"
for item in "${REQUIRED_ARTIFACTS[@]}"; do
  size="$(stat -c%s "$item")"
  printf '%-80s %12s bytes\n' "$item" "$size"
done
if [ "${RUN_GENERATE_PREDICTIONS:-0}" = "1" ] || [ "${RUN_TRAIN_CURRENT_BEST:-0}" = "1" ]; then
if [ -s "$ADAPTER_DIR/adapter_model.safetensors" ]; then
  size="$(stat -c%s "$ADAPTER_DIR/adapter_model.safetensors")"
  printf '%-80s %12s bytes\n' "$ADAPTER_DIR/adapter_model.safetensors" "$size"
else
  size="$(stat -c%s "$ADAPTER_DIR/adapter_model.bin")"
  printf '%-80s %12s bytes\n' "$ADAPTER_DIR/adapter_model.bin" "$size"
fi
fi

if [ -s "$MODEL_COMPARISON_ROOT/model_comparison_summary.csv" ]; then
  echo "Model comparison table"
  python3 - <<PY
import pandas as pd
df = pd.read_csv("$MODEL_COMPARISON_ROOT/model_comparison_summary.csv")
cols = ["model_key", "model_id", "method_type", "rows", "accuracy", "macro_f1", "status", "skip_reason"]
print(df[cols].to_string(index=False))
PY
fi

echo "E2E_RTX4090_DONE"
