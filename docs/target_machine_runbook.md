# Target Machine Runbook

## Purpose

Use this runbook on the RTX4090 Ubuntu machine after the branch is pulled.

## Existing Prediction CSV

If a prediction CSV already exists, point the target pipeline at it:

```bash
PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh
```

## Full Target Run

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

RUN_GENERATE_PREDICTIONS=1 ADAPTER_DIR=adapters/current_best PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh
RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 ADAPTER_DIR=adapters/current_best bash scripts/run_target_rtx4090_full_pipeline.sh
```

Replace `adapters/current_best` with the real adapter directory if a different adapter path is used. The precheck validates paths, imports, configs, CUDA, bf16, adapter files, tokenizer files, writable outputs, and adapter-model compatibility without loading full models or running inference.

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
* `results/paper_evidence/code_paper_consistency_audit.md`
* `results/paper_evidence/secret_scan_report.md`

## After Evidence Exists

```bash
python3 scripts/update_paper_from_evidence.py --paper materials/paper.tex --evidence_dir results/paper_evidence
```
