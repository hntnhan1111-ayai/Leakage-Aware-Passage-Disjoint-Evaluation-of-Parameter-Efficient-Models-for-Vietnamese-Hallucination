# Hallu-Paper ICIT 2026 Evidence Pipeline

This branch prepares a target-machine evidence pipeline for Vietnamese hallucination detection on ViHallu. Local Windows/WSL work is limited to source edits, documentation, lightweight validation, commit, and push.

Do not run model downloads, inference, training, or optional baselines on the local machine.

## Local Validation

Use the local-safe wrapper only:

```bash
bash scripts/run_full_pipeline.sh
```

This command compiles Python, runs the masked code/secret audit, validates CLI help paths, and checks target Bash syntax. It does not download models or run inference.

## Target RTX4090 Run

On the Linux Ubuntu RTX4090 machine:

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

RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

To evaluate an existing prediction CSV instead:

```bash
PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh
```

## Environment

Preferred setup uses `uv`. A fallback `python3 -m venv` workflow is documented in `docs/setup_rtx4090.md`.

Keep `requirements.txt` for compatibility. `pyproject.toml` mirrors the dependency list for uv-aware tooling but does not require model files or datasets to install.

## Inputs

Gold CSV files must contain:

* `id`
* `context`
* `prompt`
* `response`
* `label`

Prediction CSV files must contain:

* `id`
* `predict_label`

Valid labels are exactly:

* `no`
* `intrinsic`
* `extrinsic`

Default dataset fallback paths:

* `vihallu-train.csv`, `data/vihallu-train.csv`, `materials/vihallu-train.csv`
* `vihallu-test.csv`, `data/vihallu-test.csv`, `materials/vihallu-test.csv`
* `vihallu-private-test.csv`, `data/vihallu-private-test.csv`, `materials/vihallu-private-test.csv`

## Target Evidence Artifacts

The target evidence output directory is `results/paper_evidence`.

Required artifacts:

* `predictions_merged.csv`
* `classification_report.csv`
* `classification_report.json`
* `confusion_matrix.csv`
* `confusion_matrix.png`
* `wrong_predictions.csv`
* `selected_error_cases.md`
* `summary_metrics.json`
* `latency_summary.csv`
* `latency_summary.json`
* `validation_report.md`
* `code_paper_consistency_audit.md`
* `secret_scan_report.md`

Generated CSV, JSON, PNG, model, checkpoint, adapter, cache, `.env`, dataset, and zip files are ignored by Git.

## Paper Update

Do not update `materials/paper.tex` with metrics until target-machine evidence exists. After the target run generates artifacts:

```bash
python3 scripts/update_paper_from_evidence.py --paper materials/paper.tex --evidence_dir results/paper_evidence
```
