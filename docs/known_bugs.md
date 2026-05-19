# Known Bugs

## GitHub Push Authentication

* Failing command: `git push -u origin submit/icit2026-evidence-revision`
* Observed error: `fatal: could not read Username for 'https://github.com': No such device or address`
* Likely root cause: non-interactive HTTPS authentication is not configured in this session.
* Suggested fix: run `gh auth login --web`, switch the remote to SSH, or configure a credential helper, then push again.

## uv Not Installed Locally

* Failing command: `uv --version`
* Observed error: `/bin/bash: line 1: uv: command not found`
* Likely root cause: `uv` is not installed on the local Windows/WSL machine.
* Suggested fix: install `uv` only where needed using the Astral installer, or use the documented `python3 -m venv` fallback for local-only checks.

## Local Push Writes Need Git Metadata Access

* Failing command under normal sandbox access: `git config user.name "hntnhan1111-ayai"`
* Observed error: `could not lock config file .git/config: Read-only file system`
* Likely root cause: the local sandbox blocks writes to `.git` metadata unless elevated filesystem access is used.
* Suggested fix: use an environment where `.git` is writable, or repeat the command with the approved filesystem access mode.

## Local Python Environment Missing Target Dependencies

* Failing commands: `python3 scripts/verify_environment.py` and `python3 scripts/verify_environment.py --target-check`
* Observed missing imports: `torch`, `datasets`, `accelerate`, `peft`, `trl`, `sklearn`, `matplotlib`
* Likely root cause: the local Python environment is not the target uv environment and does not have the ML stack installed.
* Suggested fix: on the RTX4090 machine, run `uv venv --python 3.12`, activate it, and run `uv pip install -r requirements.txt` before preflight.

## WSL Python Missing pandas For Dry Run

* Failing command: `bash -lc "python3 scripts/generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv"`
* Observed error: `ModuleNotFoundError: No module named 'pandas'`
* Likely root cause: the WSL Python environment differs from the Windows `python3` used by PowerShell validation.
* Suggested fix: run validation inside one activated uv environment on the target machine.

## Duplicate IDs In Full Test CSV

* Observed file: `vihallu-test.csv`
* Observed pattern: 14,000 rows, 7,000 unique `id` values, and paired original/augmented duplicate rows.
* Risk: merging gold and prediction CSVs on `id` alone creates a cartesian expansion.
* Fix in current source: generated prediction CSVs include `row_index`; evidence building falls back to deterministic `id` occurrence alignment when `row_index` is unavailable.

## Train-Test Leakage In Public ViHallu Splits

* Observed files: `vihallu-train.csv` and `vihallu-test.csv`
* Observed pattern: all 7,000 train IDs appear in the non-augmented test rows, and the overlapping rows are identical on `id`, `context`, `prompt`, `response`, and `label`.
* Risk: any benchmark run on the current public test split would produce invalid evaluation claims due to direct train/test leakage.
* Fix in current source: `scripts/preflight_target_run.py` now fails fast when this leakage pattern is present.
