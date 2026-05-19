# Execution Flow

This document defines the exact target RTX4090 execution order for the canonical evidence path.

## End-To-End Order

1. Activate the target uv environment and confirm `HF_TOKEN` is available in `.env` or the process environment.
2. Run `PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh` when model files and adapter files are already present.
3. Run `bash scripts/run_target_rtx4090_full_pipeline.sh` for the real target execution.
4. If `RUN_DOWNLOAD_MODELS=1`, download model files before generation.
5. Run strict preflight again after model download and before any generation step.
6. Generate predictions with `scripts/generate_predictions_current_best.py`.
7. Abort immediately if malformed generations are detected.
8. Build paper evidence with `scripts/build_paper_evidence.py`.
9. Verify required evidence artifacts exist and are non-empty.

## Validation Stages

* `python3 -m compileall src scripts`
* `scripts/preflight_target_run.py` for manifest, dataset, output, adapter, tokenizer, model, and existing-prediction checks
* `scripts/audit_code_paper_consistency.py` for masked code-paper and secret audit
* `scripts/generate_predictions_current_best.py --dry-run` for local contract checks without inference
* `scripts/build_paper_evidence.py --validate-only` when an existing prediction CSV is supplied

## Fail-Fast Stages

* Environment dependency failure stops before any model work.
* Missing or incompatible adapter files stop before any model load.
* Missing model config or tokenizer files stop before inference.
* Missing dataset columns, invalid labels, duplicate-row contract mismatches, or unwritable output paths stop before inference.
* Empty or unparsable generations write `results/paper_evidence/malformed_predictions.csv` and stop before evidence metrics.
* Non-zero malformed prediction counts stop `scripts/build_paper_evidence.py` before metric computation.
* Missing required artifacts stop the shell pipeline after evidence generation.

## Expected Outputs

Generation stage:

* `results/predictions.csv`
* `results/prediction_config.json`
* `results/paper_evidence/malformed_predictions.csv`

Evidence stage:

* `results/paper_evidence/predictions_merged.csv`
* `results/paper_evidence/classification_report.csv`
* `results/paper_evidence/classification_report.json`
* `results/paper_evidence/confusion_matrix.csv`
* `results/paper_evidence/confusion_matrix.png`
* `results/paper_evidence/wrong_predictions.csv`
* `results/paper_evidence/selected_error_cases.md`
* `results/paper_evidence/summary_metrics.json`
* `results/paper_evidence/latency_summary.csv`
* `results/paper_evidence/latency_summary.json`
* `results/paper_evidence/run_metadata.json`
* `results/paper_evidence/validation_report.md`
* `results/paper_evidence/code_paper_consistency_audit.md`
* `results/paper_evidence/secret_scan_report.md`
