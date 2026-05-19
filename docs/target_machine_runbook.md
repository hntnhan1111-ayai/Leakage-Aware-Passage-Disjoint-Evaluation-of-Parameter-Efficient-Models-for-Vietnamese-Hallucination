# Target Machine Runbook

## Purpose

Use this runbook on the RTX4090 Ubuntu machine after the branch is pulled.

## Existing Prediction CSV

If a prediction CSV already exists, point the target pipeline at it:

```bash
PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh
```

## Full One-Command Target Run

```bash
cd ~/Hallu-Paper
git fetch origin
git checkout submit/icit2026-evidence-revision
git pull

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

uv venv --python 3.12
source .venv/bin/activate
uv pip install -U pip
uv pip install -r requirements.txt

printf 'HF_TOKEN=your_token_here\n' > .env

ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1 RUN_DOWNLOAD_MODELS=1 RUN_TRAIN_CURRENT_BEST=1 RUN_GENERATE_PREDICTIONS=1 RUN_BASELINES=1 bash scripts/run_all_e2e_rtx4090.sh
```

The public ViHallu train/test files currently contain known overlap. Without `ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1`, the pipeline fails before expensive work. With the override, the run is marked as challenge-style public split evaluation and writes `results/paper_evidence/leakage_report.md`.

The one-command runner downloads models when requested, trains the current-best Vistral QLoRA adapter into `adapters/current_best`, runs full-test inference into `results/predictions.csv`, generates paper evidence, writes runtime metadata, and runs enabled baselines. Optional baselines do not fail the main run unless `STRICT_BASELINES=1`.

## Expected Artifact Paths

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
* `results/paper_evidence/run_metadata.json`
* `results/paper_evidence/leakage_report.md` when `ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1`
* `results/paper_evidence/code_paper_consistency_audit.md`
* `results/paper_evidence/secret_scan_report.md`

## Legacy Target Runner

`scripts/run_target_rtx4090_full_pipeline.sh` remains available for legacy adapter/prediction workflows. New full train to inference to evidence execution should use `scripts/run_all_e2e_rtx4090.sh`.

## After Evidence Exists

```bash
python3 scripts/update_paper_from_evidence.py --paper materials/paper.tex --evidence_dir results/paper_evidence
```
