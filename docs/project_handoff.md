# Project Summary

Hallu-Paper is a Vietnamese hallucination-detection project for ICIT 2026. The rescue branch is `submit/icit2026-evidence-revision`.

The repository is prepared for reproducible evidence generation on a target RTX4090 machine. Local Windows/WSL work is limited to editing, lightweight validation, and Git operations.

## Current Architecture

* `src/data/vihallu.py` handles split discovery, CSV parsing, label validation, and fallback dataset paths.
* `src/data/vihallu.py` also provides duplicate-ID-aware alignment keys for the full test split.
* `src/models/download.py` handles Hugging Face snapshot downloads and skip-if-present behavior.
* `src/models/loader.py` centralizes causal LLM, image-text, and encoder loading with bfloat16 and NF4 assumptions.
* `src/evaluation/metrics.py` writes classification report, confusion matrix, and summary metrics.
* `src/evaluation/latency.py` records runtime and CUDA memory metrics.
* `scripts/build_paper_evidence.py` merges predictions with gold labels and writes evidence outputs.
* `scripts/generate_predictions_current_best.py` is the target-side prediction entrypoint with dry-run support.
* `scripts/verify_environment.py` validates dependencies, manifests, dataset fallback paths, token key presence, CUDA, bf16, and package versions.
* `scripts/preflight_target_run.py` validates adapter files, tokenizer files, model compatibility, writable outputs, and existing prediction contracts before generation or evidence building.
* `scripts/run_target_rtx4090_full_pipeline.sh` orchestrates the target execution path.
* `scripts/run_full_pipeline.sh` is the local-safe validation wrapper.
* `scripts/update_paper_from_evidence.py` only updates the paper when generated evidence exists.

## Implemented Pipelines

* Local-safe validation: compile, help text checks, dry-run checks, and Bash syntax checks.
* Target preflight: `PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh`.
* Target RTX4090 pipeline: dependency verification, optional model download, prediction generation or existing `PRED_CSV` consumption, evidence generation, artifact validation, and optional baseline execution.

## Important Scripts

* `scripts/generate_predictions_current_best.py`
* `scripts/verify_environment.py`
* `scripts/preflight_target_run.py`
* `scripts/build_paper_evidence.py`
* `scripts/audit_code_paper_consistency.py`
* `scripts/update_paper_from_evidence.py`
* `scripts/run_full_pipeline.sh`
* `scripts/run_target_rtx4090_full_pipeline.sh`
* `scripts/download_models.py`
* `scripts/run_optional_qwen3_prompt_baseline.py`

## Expected Outputs

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

## Research Contributions

* Lightweight, reproducible empirical recipe for Vietnamese hallucination detection.
* Evidence-first workflow with deterministic defaults and explicit validation.
* Conservative paper positioning that avoids unsupported novelty claims.

## Current Weaknesses

* Target-machine evidence has not been generated yet in this workspace.
* Local Python does not have the complete target dependency stack, so `scripts/verify_environment.py` fails locally until a uv environment is installed.
* `vihallu-test.csv` contains paired duplicate IDs; current code avoids id-only cartesian merges by using `row_index` or `id` occurrence alignment.
* The current public `vihallu-train.csv` and non-augmented `vihallu-test.csv` rows overlap exactly, so target preflight now blocks benchmark execution until the evaluation split is corrected.
* GitHub push was blocked earlier by non-interactive HTTPS authentication.
* Legacy root-level script copies remain in the workspace and should not be staged accidentally.
* `uv` is not installed locally; only setup docs are present.

## Priority Next Steps

* Authenticate GitHub and push the branch.
* Run target precheck.
* Run the target RTX4090 pipeline.
* Update the paper from generated evidence.
* Decide whether legacy scripts should be archived after target validation.

## Target Machine Execution Plan

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
