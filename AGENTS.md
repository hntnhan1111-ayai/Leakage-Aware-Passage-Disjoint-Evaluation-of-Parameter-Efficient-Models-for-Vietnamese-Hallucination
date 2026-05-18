# AGENTS.md

## Project

Hallu-Paper is a Vietnamese hallucination detection paper/code repository targeting ICIT 2026.

## Deadline

Internal freeze: 2026-05-20.

## Current priority

Generate reproducible paper evidence before adding new methods.

## Required behavior

Inspect real files before editing.
Use exact file paths in reports.
Do not invent results.
Do not hardcode secrets.
Do not print .env values.
Read HF_TOKEN from .env or environment only.
Do not commit .env, models, checkpoints, adapters, datasets, caches, or large result files.
Do not force push.
Write generated Python code without explanatory comments.
Write generated Bash scripts without explanatory comments except shebang.
Set seed 42 by default.
Fail fast on missing files, missing columns, invalid labels, missing models, broken imports, or missing output artifacts.
Create missing output directories before writing files.
Use dataset path fallback discovery.
Record all bugs, fixes, decisions, and commands in docs/codex_session_memory.md.

## Paper positioning

Use: lightweight and reproducible empirical recipe for Vietnamese hallucination detection.
Avoid: novel state-of-the-art framework.

## Main labels

no
intrinsic
extrinsic

## Required paper artifacts

classification_report.csv
classification_report.json
confusion_matrix.csv
confusion_matrix.png
wrong_predictions.csv
selected_error_cases.md
summary_metrics.json
latency_summary.csv
code_paper_consistency_audit.md

## Final command target

bash scripts/run_full_pipeline.sh

## Git policy

Use branch submit/icit2026-evidence-revision.
Before push, run validation and inspect git status.
Push only source, configs, docs, README, paper edits, and small non-sensitive reproducibility metadata.

## Local vs Target Machine Rules

* The local Windows/WSL machine is for code editing, lightweight validation, documentation, Git commit, and Git push only.
* Do not download Hugging Face models on the local machine.
* Do not run model inference on the local machine.
* Do not run training on the local machine.
* Do not run GPU workloads on the local machine.
* The target machine is Linux Ubuntu with RTX4090 24GB accessed through AnyDesk.
* Full pipeline, model downloads, inference, training, and optional baselines must be run only on the target RTX4090 machine after pulling the pushed branch.

## uv Environment Rules

* Prefer uv for project environment setup.
* Keep requirements.txt for compatibility.
* Document target setup with:
  * `uv venv --python 3.12`
  * `source .venv/bin/activate`
  * `uv pip install -r requirements.txt`
* Do not install packages into global system Python.

## Git Identity Rules

* Use local repository Git identity:
  * `git config user.name "hntnhan1111-ayai"`
  * `git config user.email "hntnhan1111@gmail.com"`
* Do not store GitHub passwords or personal access tokens in repository files.
* If GitHub authentication is requested during push, use browser/device authentication or Git Credential Manager.

## Completion Rules

* Before ending, run only lightweight local validations:
  * `python3 -m compileall src scripts`
  * `python3 scripts/audit_code_paper_consistency.py --out results/paper_evidence/code_paper_consistency_audit.md --fail-on-secret`
  * `python3 scripts/build_paper_evidence.py --help`
  * `python3 scripts/generate_predictions_current_best.py --help`
  * `bash -n scripts/run_full_pipeline.sh`
  * `bash -n scripts/run_target_rtx4090_full_pipeline.sh`
* Do not claim evidence metrics exist unless target-machine pipeline has generated them.
* Keep `docs/codex_memory_bank.md` updated.
* If context grows too large, compact current state into `docs/codex_memory_bank.md` before continuing.
