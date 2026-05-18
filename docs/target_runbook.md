# Target RTX4090 Runbook

## Local Policy

The local Windows/WSL machine is only for source edits, lightweight validation, commit, and push. Do not run model downloads, inference, training, or optional baselines locally.

## Target Full Run

On the RTX4090 Ubuntu machine:

```bash
cd ~/Hallu-Paper
git fetch origin
git checkout submit/icit2026-evidence-revision
git pull
source .venv/bin/activate
RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

## Existing Prediction CSV

If a prediction CSV already exists and contains `id,predict_label`:

```bash
PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh
```

## Expected Artifacts

* `results/predictions.csv`
* `results/prediction_config.json`
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
* `results/paper_evidence/validation_report.md`
* `results/paper_evidence/code_paper_consistency_audit.md`
* `results/paper_evidence/secret_scan_report.md`

## Paper Update

After the evidence artifacts exist:

```bash
python3 scripts/update_paper_from_evidence.py --paper materials/paper.tex --evidence_dir results/paper_evidence
```
