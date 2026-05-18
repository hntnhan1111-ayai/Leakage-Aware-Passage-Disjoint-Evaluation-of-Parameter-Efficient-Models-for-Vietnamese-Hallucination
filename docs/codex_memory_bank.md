# Codex Memory Bank

## Project

Hallu-Paper ICIT 2026 rescue implementation.

## Current Objective

Prepare source, docs, scripts, uv workflow, and target RTX4090 pipeline for GitHub push without running local model download, inference, training, or GPU workloads.

## Stable Rules

* Use seed 42.
* Use full `vihallu-test.csv` by default on the target machine.
* Never print or commit secrets.
* Never hardcode `HF_TOKEN`.
* Do not update paper metrics without generated artifacts.
* Do not claim `r=64` unless reproduced.
* Local Windows/WSL is for code, docs, lightweight validation, commit, and push only.
* Target RTX4090 machine is for model download, inference, training, optional baselines, and full evidence generation.

## Current Context

* Repository path: `/mnt/d/uit-paper`.
* Branch: `submit/icit2026-evidence-revision`.
* Remote: `https://github.com/hntnhan1111-ayai/Hallu-Paper.git`.
* `uv` is not installed locally; docs include install commands.
* `.codex/rules` initially failed under normal sandbox access, then succeeded with elevated filesystem access.
* `.git/config` initially failed under normal sandbox access, then local Git identity was configured with elevated filesystem access.

## Commands Run

* `pwd`
* `git rev-parse --is-inside-work-tree`
* `git branch --show-current`
* `git status --short`
* `git remote -v`
* `git config user.name`
* `git config user.email`
* `python3 --version`
* `uv --version`
* `find . -maxdepth 4 -type f | sort`
* `sed -n '1,260p' AGENTS.md`
* `sed -n '1,260p' docs/codex_memory_bank.md`
* `cat requirements.txt`
* `sed -n '1,260p' scripts/run_full_pipeline.sh`
* `sed -n '1,260p' scripts/build_paper_evidence.py`
* `sed -n '1,220p' scripts/audit_code_paper_consistency.py`
* `sed -n '1,220p' .gitignore`
* `sed -n '1,280p' uit_our_method.py`
* `sed -n '1,280p' uit_r64.py`
* `sed -n '1,220p' src/models/loader.py`
* `sed -n '1,220p' src/data/vihallu.py`
* `sed -n '1,220p' src/evaluation/metrics.py`
* `sed -n '1,220p' src/evaluation/latency.py`
* `sed -n '1,220p' scripts/download_models.py`
* `sed -n '1,220p' scripts/run_optional_qwen3_prompt_baseline.py`
* `mkdir -p .codex/rules docs results/paper_evidence`
* `mkdir -p docs results/paper_evidence`
* `chmod +x scripts/run_full_pipeline.sh scripts/run_target_rtx4090_full_pipeline.sh`
* `git config user.name "hntnhan1111-ayai"`
* `git config user.email "hntnhan1111@gmail.com"`
* `ls -la .git`
* `ls -la .git/config`
* `touch .git/config.locktest`
* `python3 -m compileall src scripts`
* `python3 scripts/audit_code_paper_consistency.py --out results/paper_evidence/code_paper_consistency_audit.md --fail-on-secret`
* `python3 scripts/build_paper_evidence.py --help`
* `python3 scripts/generate_predictions_current_best.py --help`
* `python3 scripts/generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv`
* `python3 scripts/download_models.py --help`
* `python3 scripts/update_paper_from_evidence.py --help`
* `bash -n scripts/run_full_pipeline.sh`
* `bash -n scripts/run_target_rtx4090_full_pipeline.sh`
* `find scripts src -type f -name '*.pyc' -delete`
* `find scripts src -type d -name '__pycache__' -empty -delete`

## Bugs Found

* `.codex/rules` creation failed under normal sandbox access with `Read-only file system`.
* `git config user.name` and `git config user.email` initially failed because `.git/config` could not be locked under normal sandbox access.
* Local `uv` command is missing.
* Previous `scripts/run_full_pipeline.sh` would call model download; it has been converted to local-safe validation only.

## Fixes Applied

* Added local-vs-target machine rules, uv rules, Git identity rules, and completion rules to `AGENTS.md`.
* Added `pyproject.toml`.
* Added uv and target execution docs.
* Added Codex fallback docs for rules/config.
* Added target-only `scripts/run_target_rtx4090_full_pipeline.sh`.
* Reworked `scripts/run_full_pipeline.sh` as local-safe validation.
* Added `scripts/generate_predictions_current_best.py` with `--help` and `--dry-run` support.
* Added `scripts/update_paper_from_evidence.py` with evidence guards.
* Updated `scripts/build_paper_evidence.py` to support help without heavy imports and write `validation_report.md`.
* Updated `scripts/audit_code_paper_consistency.py` to write `secret_scan_report.md`.
* Updated `.gitignore` for generated results, caches, datasets, model artifacts, and zip archives.
* Created `.codex/config.toml` and `.codex/rules/default.rules`.
* Configured local Git identity as `hntnhan1111-ayai <hntnhan1111@gmail.com>`.
* Removed generated local `__pycache__` and `*.pyc` files after validation.

## Lightweight Validation Status

* PASS: `git rev-parse --is-inside-work-tree`
* PASS: `python3 -m compileall src scripts`
* PASS: `python3 scripts/audit_code_paper_consistency.py --out results/paper_evidence/code_paper_consistency_audit.md --fail-on-secret`
* PASS: `python3 scripts/build_paper_evidence.py --help`
* PASS: `python3 scripts/generate_predictions_current_best.py --help`
* PASS: `python3 scripts/generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv`
* PASS: `python3 scripts/download_models.py --help`
* PASS: `python3 scripts/update_paper_from_evidence.py --help`
* PASS: `bash -n scripts/run_full_pipeline.sh`
* PASS: `bash -n scripts/run_target_rtx4090_full_pipeline.sh`

## Evidence Artifacts

* Local source-only validation may create:
  * `results/paper_evidence/code_paper_consistency_audit.md`
  * `results/paper_evidence/secret_scan_report.md`
* Full evidence artifacts must be generated on the target RTX4090 machine.

## Paper Update Status

* `materials/paper.tex` was not updated locally because target-machine evidence metrics have not been generated.
* `scripts/update_paper_from_evidence.py` is ready to update the paper after target artifacts exist.

## Git Status

* Repository is a valid Git work tree.
* Local Git identity is configured.
* Local commit created: `957dbb6` with message `Prepare target RTX4090 ICIT evidence pipeline`.
* Push attempted with `git push -u origin submit/icit2026-evidence-revision`.
* Push over HTTPS is blocked in this non-interactive session: `fatal: could not read Username for 'https://github.com': No such device or address`.

## Next Steps

* Run lightweight validation commands.
* Authenticate GitHub with Git Credential Manager, `gh auth login`, SSH remote, or a configured credential helper.
* Then run `git push -u origin submit/icit2026-evidence-revision`.
* On target RTX4090, run `RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 bash scripts/run_target_rtx4090_full_pipeline.sh`.
