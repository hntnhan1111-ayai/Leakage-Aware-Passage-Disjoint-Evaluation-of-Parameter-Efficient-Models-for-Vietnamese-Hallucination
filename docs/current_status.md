# Current Status

Last updated: 2026-05-18.

## Repository State

* Branch: `submit/icit2026-evidence-revision`
* Local Git identity: `hntnhan1111-ayai <hntnhan1111@gmail.com>`
* HEAD commit before this handoff update: `9c22e85`
* Working tree contains new persistent handoff docs that are not yet committed.

## What Is Done

* Local-safe validation scripts exist.
* Target RTX4090 orchestration script exists.
* Target RTX4090 orchestration now has `PRECHECK_ONLY=1` preflight mode plus a strict second preflight before generation.
* `configs/experiment_manifest.yaml`, `docs/canonical_files.md`, `docs/target_preflight.md`, and `docs/runtime_validation.md` exist.
* `scripts/verify_environment.py` exists and fails clearly on missing dependencies, CUDA, bf16, token key, manifest errors, and dataset path errors.
* `scripts/preflight_target_run.py` exists for adapter, tokenizer, model, output-directory, and existing-prediction preflight checks.
* Prediction generation entrypoint exists with `--help` and `--dry-run`.
* Evidence builder and paper-update guard scripts exist.
* Evidence merging handles duplicate full-test IDs with `row_index` or deterministic `id` occurrence keys.
* Generation config now records malformed-count accounting and deterministic decoding parameters.
* `run_metadata.json` generation exists for reproducibility appendix inputs.
* Persistent handoff docs and memory bank files exist.
* Lightweight validation passed for compile, help text, dry run under Windows `python3`, and Bash syntax.

## What Is Not Done

* No target-machine model download or inference has been run in this workspace.
* No new prediction CSV has been generated in this workspace.
* No target-machine evidence artifacts have been generated in this workspace.
* `materials/paper.tex` has not been updated from generated evidence.
* Git push still depends on interactive GitHub authentication.
* Local Python dependency verification fails until the target uv environment is installed.
* The current public ViHallu train/test files fail the no-leakage requirement and are now blocked by target preflight.

## Current Blocker

* Non-interactive HTTPS push to GitHub cannot prompt for credentials in this session.
* Target dependency verification cannot pass on the local Python environment because required ML packages are missing.
* Target benchmark execution should not proceed until the public train/test leakage is resolved or the evaluation dataset is replaced with a valid non-leaking split.

## Immediate Next Action

* Resolve the public train/test leakage first, then run `PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh` on the RTX4090 machine.
