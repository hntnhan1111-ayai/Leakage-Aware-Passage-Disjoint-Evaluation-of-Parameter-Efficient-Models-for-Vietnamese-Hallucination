# Final Execution Checklist

Use this checklist on the RTX4090 machine before the expensive benchmark run.

## Execution Order

1. Pull `submit/icit2026-evidence-revision`.
2. Activate the uv environment.
3. Confirm `HF_TOKEN` is present in `.env` or the shell environment.
4. Run the one-command E2E pipeline.
5. Confirm all required paper artifacts exist after the run.
6. If the public split leakage override was used, cite `leakage_report.md` as a limitation.

## Preflight Command

The new one-command runner performs preflight internally. A standalone legacy preflight is still available when model files and adapter files already exist:

```bash
RUN_GENERATE_PREDICTIONS=1 ADAPTER_DIR=adapters/current_best PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

If evaluating an existing prediction CSV:

```bash
PRED_CSV=final_submission_scratch.csv PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

## Benchmark Command

Canonical one-command E2E command:

```bash
ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1 RUN_DOWNLOAD_MODELS=1 RUN_TRAIN_CURRENT_BEST=1 RUN_GENERATE_PREDICTIONS=1 RUN_BASELINES=1 bash scripts/run_all_e2e_rtx4090.sh
```

## Expected Outputs

Generation outputs:

* `results/predictions.csv`
* `results/prediction_config.json`
* `results/paper_evidence/malformed_predictions.csv`
* `adapters/current_best/adapter_config.json`
* `adapters/current_best/adapter_model.safetensors` or `adapters/current_best/adapter_model.bin`
* `results/current_best_training/training_config_resolved.json`
* `results/current_best_training/train_runtime.json`

Evidence outputs:

* `results/paper_evidence/classification_report.csv`
* `results/paper_evidence/summary_metrics.json`
* `results/paper_evidence/confusion_matrix.png`
* `results/paper_evidence/wrong_predictions.csv`
* `results/paper_evidence/validation_report.md`
* `results/paper_evidence/latency_summary.csv`
* `results/paper_evidence/run_metadata.json`
* `results/paper_evidence/leakage_report.md` when `ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1`

## Final Paper-Writing Inputs

* Results section: `classification_report.csv`, `summary_metrics.json`, `confusion_matrix.png`
* Error Analysis section: `wrong_predictions.csv`, `selected_error_cases.md`
* Runtime section: `latency_summary.csv`, `latency_summary.json`
* Reproducibility appendix: `run_metadata.json`, `prediction_config.json`, `validation_report.md`, `code_paper_consistency_audit.md`
