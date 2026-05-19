# Benchmark To Paper Workflow

This document describes the full workflow from target benchmark execution to ICIT paper writing.

## Workflow

1. Run target precheck with `PRECHECK_ONLY=1`.
2. Resolve any precheck failure before benchmark execution.
3. Run the target benchmark pipeline.
4. Confirm all benchmark and evidence artifacts exist.
5. Use the generated artifacts directly to write the paper sections.

## Artifact Dependencies

* `results/predictions.csv` depends on model files, adapter files, dataset files, and deterministic generation settings.
* `results/prediction_config.json` depends on the generation run and records the decoding contract.
* `results/paper_evidence/malformed_predictions.csv` depends on the generation run and records any parsing failures.
* `results/paper_evidence/classification_report.csv`, `summary_metrics.json`, and `confusion_matrix.png` depend on `predictions.csv` plus the gold evaluation CSV.
* `results/paper_evidence/wrong_predictions.csv` and `selected_error_cases.md` depend on merged gold and prediction rows.
* `results/paper_evidence/latency_summary.csv` depends on the prediction generation timing path.
* `results/paper_evidence/run_metadata.json` depends on the target runtime environment, Git state, manifest, and prediction config.

## Paper Section Inputs

* Results section:
  `classification_report.csv`, `summary_metrics.json`, `confusion_matrix.png`
* Error Analysis section:
  `wrong_predictions.csv`, `selected_error_cases.md`
* Runtime section:
  `latency_summary.csv`, `latency_summary.json`
* Reproducibility appendix:
  `run_metadata.json`, `prediction_config.json`, `validation_report.md`, `code_paper_consistency_audit.md`

## Readiness Conditions

The benchmark-to-paper workflow is valid only when:

* dataset semantics are correct,
* there is no train/test leakage,
* the precheck passes,
* malformed predictions are zero,
* the evidence validation report passes, and
* the required outputs are non-empty.
