# Benchmark To Paper Workflow

This document describes the full workflow from target benchmark execution to ICIT paper writing.

## Workflow

1. Activate the target uv environment.
2. Run the one-command E2E target workflow.
3. Confirm all main-method artifacts exist.
4. Review `leakage_report.md` when the public split override is used.
5. Use the generated artifacts directly to write the paper sections.

Canonical target command:

```bash
ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1 RUN_DOWNLOAD_MODELS=1 RUN_TRAIN_CURRENT_BEST=1 RUN_GENERATE_PREDICTIONS=1 RUN_BASELINES=1 bash scripts/run_all_e2e_rtx4090.sh
```

## Artifact Dependencies

* `adapters/current_best/adapter_config.json` and adapter weights depend on `scripts/train_current_best.py`.
* `results/current_best_training/training_config_resolved.json` and `train_runtime.json` depend on successful current-best training.
* `results/predictions.csv` depends on model files, adapter files, dataset files, and deterministic generation settings.
* `results/prediction_config.json` depends on the generation run and records the decoding contract.
* `results/paper_evidence/malformed_predictions.csv` depends on the generation run and records any parsing failures.
* `results/paper_evidence/classification_report.csv`, `summary_metrics.json`, and `confusion_matrix.png` depend on `predictions.csv` plus the gold evaluation CSV.
* `results/paper_evidence/wrong_predictions.csv` and `selected_error_cases.md` depend on merged gold and prediction rows.
* `results/paper_evidence/latency_summary.csv` depends on the prediction generation timing path.
* `results/paper_evidence/run_metadata.json` depends on the target runtime environment, Git state, manifest, and prediction config.
* `results/paper_evidence/leakage_report.md` is required when `ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1`.
* `results/baselines/*` depends on enabled baseline settings in `configs/baseline_models.yaml`.

## Paper Section Inputs

* Results section:
  `classification_report.csv`, `summary_metrics.json`, `confusion_matrix.png`
* Error Analysis section:
  `wrong_predictions.csv`, `selected_error_cases.md`
* Runtime section:
  `latency_summary.csv`, `latency_summary.json`
* Reproducibility appendix:
  `run_metadata.json`, `prediction_config.json`, `validation_report.md`, `code_paper_consistency_audit.md`, `leakage_report.md` when override is used

## Readiness Conditions

The benchmark-to-paper workflow is valid only when:

* dataset semantics are explicit,
* train/test leakage is absent or explicitly overridden as challenge-style evaluation,
* the precheck passes,
* malformed predictions are zero,
* the evidence validation report passes, and
* the required outputs are non-empty.
