# Final Execution Checklist

Use this checklist on the RTX4090 machine before the expensive benchmark run.

## Execution Order

1. Pull `submit/icit2026-evidence-revision`.
2. Activate the uv environment.
3. Confirm `HF_TOKEN` is present in `.env` or the shell environment.
4. Run the no-inference precheck.
5. If precheck passes and the dataset semantics are clean, run the benchmark command.
6. Confirm all required paper artifacts exist after the run.

## Preflight Command

When model files and adapter files already exist:

```bash
RUN_GENERATE_PREDICTIONS=1 ADAPTER_DIR=adapters/current_best PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

If evaluating an existing prediction CSV:

```bash
PRED_CSV=final_submission_scratch.csv PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

## Benchmark Command

When model download is required:

```bash
RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 ADAPTER_DIR=adapters/current_best bash scripts/run_target_rtx4090_full_pipeline.sh
```

When model files already exist:

```bash
RUN_GENERATE_PREDICTIONS=1 ADAPTER_DIR=adapters/current_best bash scripts/run_target_rtx4090_full_pipeline.sh
```

## Expected Outputs

Generation outputs:

* `results/predictions.csv`
* `results/prediction_config.json`
* `results/paper_evidence/malformed_predictions.csv`

Evidence outputs:

* `results/paper_evidence/classification_report.csv`
* `results/paper_evidence/summary_metrics.json`
* `results/paper_evidence/confusion_matrix.png`
* `results/paper_evidence/wrong_predictions.csv`
* `results/paper_evidence/validation_report.md`
* `results/paper_evidence/latency_summary.csv`
* `results/paper_evidence/run_metadata.json`

## Final Paper-Writing Inputs

* Results section: `classification_report.csv`, `summary_metrics.json`, `confusion_matrix.png`
* Error Analysis section: `wrong_predictions.csv`, `selected_error_cases.md`
* Runtime section: `latency_summary.csv`, `latency_summary.json`
* Reproducibility appendix: `run_metadata.json`, `prediction_config.json`, `validation_report.md`, `code_paper_consistency_audit.md`
