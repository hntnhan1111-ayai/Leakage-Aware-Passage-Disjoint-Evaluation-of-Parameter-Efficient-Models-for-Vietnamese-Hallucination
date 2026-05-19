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

## Project Overview

* Hallu-Paper is the Vietnamese hallucination-detection repository for ICIT 2026.
* The rescue branch is `submit/icit2026-evidence-revision`.
* The current workflow focuses on reproducible evidence, target-machine execution, and conservative paper claims.

## Current Goal

* Finish the evidence-generation path on the RTX4090 target machine.
* Keep local work limited to code, docs, validation, and Git operations.
* Preserve a concise handoff trail for Codex CLI, Codex Desktop, and the Codex IDE extension.

## Research Constraints

* Avoid overclaiming novelty or state-of-the-art status.
* Use the full `vihallu-test.csv` by default on the target machine.
* Keep legacy scripts until the target pipeline is validated and any archival decision is explicit.

## Local Machine Restrictions

* Do not download Hugging Face models locally.
* Do not run inference, training, or optional baselines locally.
* Do not run GPU workloads locally.
* Do not stage `.env`, datasets, model weights, checkpoints, adapters, caches, or other large artifacts.

## Target RTX4090 Machine Rules

* The Linux Ubuntu RTX4090 machine accessed through AnyDesk is the authoritative execution environment.
* Use the target machine for model download, inference, training, optional baselines, and paper-evidence generation.
* Use `uv` on the target machine when available.

## Reproducibility Rules

* Default seed is `42`.
* Generation paths should be deterministic by default.
* Read `HF_TOKEN` only from `.env` or process environment.
* Never print or hardcode token values.
* Validate required files, columns, labels, and outputs before treating a run as valid.

## Paper Evidence Requirements

* Required artifacts include classification report, confusion matrix, wrong predictions, selected error cases, summary metrics, latency summary, and code-paper audit outputs.
* Do not update paper metrics unless the generated evidence exists.
* Do not claim `r=64` unless the audited evidence actually shows `r=64`.

## Required Validation Commands

* `python3 -m compileall src scripts`
* `python3 scripts/audit_code_paper_consistency.py --out results/paper_evidence/code_paper_consistency_audit.md --fail-on-secret`
* `python3 scripts/build_paper_evidence.py --help`
* `python3 scripts/generate_predictions_current_best.py --help`
* `python3 scripts/generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv`
* `bash -n scripts/run_full_pipeline.sh`
* `bash -n scripts/run_target_rtx4090_full_pipeline.sh`

## Git Safety Rules

* Configure local repo identity with `git config user.name "hntnhan1111-ayai"` and `git config user.email "hntnhan1111@gmail.com"`.
* Use explicit `git add` paths only.
* Do not use `git add .`.
* Do not force push.
* If GitHub authentication is needed, use browser/device auth, Git Credential Manager, or SSH.

## Known Research Risks

* Target-machine evidence has not been generated yet in this workspace.
* The current branch still contains unarchived legacy root-level script copies.
* Some runs may require a valid adapter or existing prediction CSV on the target machine.

## Handoff Files

* `docs/codex_memory_bank.md` is the rolling persistent state file.
* `docs/project_handoff.md` is the concise cross-session overview.
* `docs/current_status.md` is the quick status snapshot.
* `docs/open_tasks.md` tracks prioritized TODOs.
* `docs/known_bugs.md` tracks reproducible failures only.
* `docs/research_direction.md` captures realistic ICIT 2026 research upgrades.
* `docs/runtime_constraints.md` documents local versus target execution rules.
* `docs/target_machine_runbook.md` is the target RTX4090 execution guide.
* `docs/paper_submission_status.md` tracks paper readiness and missing evidence.
