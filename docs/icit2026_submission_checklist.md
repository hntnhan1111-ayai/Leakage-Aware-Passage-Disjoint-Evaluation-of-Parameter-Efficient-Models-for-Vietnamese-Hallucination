# ICIT 2026 Submission Checklist

## Evidence

* Run `bash scripts/run_full_pipeline.sh`.
* Confirm `results/paper_evidence/classification_report.csv` exists and is non-empty.
* Confirm `results/paper_evidence/classification_report.json` exists and is non-empty.
* Confirm `results/paper_evidence/confusion_matrix.csv` exists and is non-empty.
* Confirm `results/paper_evidence/confusion_matrix.png` exists and is non-empty.
* Confirm `results/paper_evidence/wrong_predictions.csv` exists and is non-empty or intentionally empty only when there are no errors.
* Confirm `results/paper_evidence/selected_error_cases.md` exists and is non-empty.
* Confirm `results/paper_evidence/summary_metrics.json` exists and is non-empty.
* Confirm `results/paper_evidence/latency_summary.csv` exists and is non-empty.
* Confirm `results/paper_evidence/code_paper_consistency_audit.md` exists and is non-empty.

## Paper

* Update `materials/paper.tex` only from generated evidence artifacts.
* Avoid unsupported state-of-the-art claims.
* Use the contribution framing: lightweight and reproducible empirical recipe for Vietnamese hallucination detection.
* Confirm LoRA rank claims match audited code and generated evidence.
* Confirm the test split is the full `vihallu-test.csv` unless a configured sample is explicitly reported.

## Security

* Confirm `.env` is ignored.
* Confirm no Hugging Face token value appears outside `.env`.
* Do not stage model weights, adapters, checkpoints, datasets, caches, or generated large result files.

## Git

* Use branch `submit/icit2026-evidence-revision`.
* Run `git status --short`.
* Stage only source, configs, docs, README, paper edits, and small reproducibility metadata.
* Do not force push.
