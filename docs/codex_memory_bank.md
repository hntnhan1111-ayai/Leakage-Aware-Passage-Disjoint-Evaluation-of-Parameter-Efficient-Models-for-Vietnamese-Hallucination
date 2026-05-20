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
* Local commit created: `9c22e85` with message `Prepare target RTX4090 ICIT evidence pipeline`.
* Push attempted with `git push -u origin submit/icit2026-evidence-revision`.
* Push over HTTPS is blocked in this non-interactive session: `fatal: could not read Username for 'https://github.com': No such device or address`.

## Next Steps

* Run lightweight validation commands.
* Authenticate GitHub with Git Credential Manager, `gh auth login`, SSH remote, or a configured credential helper.
* Then run `git push -u origin submit/icit2026-evidence-revision`.
* On target RTX4090, run `RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 bash scripts/run_target_rtx4090_full_pipeline.sh`.

## 2026-05-18T14:24:22Z Update

* Created persistent handoff docs for project handoff, current status, open tasks, known bugs, research direction, runtime constraints, target runbook, and paper submission status.
* Appended stable handoff guidance to `AGENTS.md`.
* Validation commands rerun this session all passed: `git status --short`, `find docs -maxdepth 2 -type f | sort`, `sed -n '1,260p' AGENTS.md`, `sed -n '1,260p' docs/codex_memory_bank.md`, `sed -n '1,260p' docs/project_handoff.md`, `tail -n 20 docs/codex_session_memory.md`.
* No new evidence metrics were generated locally.
* Git push remains blocked until GitHub authentication is available in a non-interactive session.
* Files modified in this update: `AGENTS.md`, `docs/codex_memory_bank.md`, `docs/project_handoff.md`, `docs/current_status.md`, `docs/open_tasks.md`, `docs/known_bugs.md`, `docs/research_direction.md`, `docs/runtime_constraints.md`, `docs/target_machine_runbook.md`, `docs/paper_submission_status.md`, `docs/codex_session_memory.md`.

## 2026-05-18 Pre-RTX4090 Validation Update

* Added `configs/experiment_manifest.yaml` as the canonical execution manifest.
* Added `scripts/verify_environment.py` for package import checks, package versions, manifest validation, dataset fallback discovery, HF token key presence, CUDA, and bf16 checks.
* Added target `--smoke-test` mode to `scripts/run_target_rtx4090_full_pipeline.sh`.
* Added `docs/canonical_files.md`, `docs/target_preflight.md`, and `docs/runtime_validation.md`.
* Strengthened `src/data/vihallu.py` for required columns, exact labels, empty rows, invalid labels, duplicate-ID reporting, prediction row counts, and missing or extra IDs.
* Added `--validate-only` to `scripts/build_paper_evidence.py`.
* Fixed duplicate-ID evidence alignment risk: generated predictions now include `row_index`; evidence building falls back to deterministic `id` occurrence keys when `row_index` is unavailable.
* Renamed token helper names in `scripts/verify_environment.py` after the masked audit flagged false-positive `hf_...` patterns.
* Found that `vihallu-test.csv` has 14,000 rows and 7,000 unique IDs because original and augmented rows share IDs.
* Local Windows Python still lacks core target ML dependencies, so `python3 scripts/verify_environment.py` and `python3 scripts/verify_environment.py --target-check` fail locally until the target uv environment is installed.
* WSL Python lacks `pandas`, so the dry-run fails in WSL but passes under Windows Python.
* Passing validations this pass: `python3 -m compileall src scripts`, temporary-output secret/code audit, `python3 scripts/build_paper_evidence.py --help`, `python3 scripts\generate_predictions_current_best.py --help`, `python3 scripts\generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv`, `bash -n scripts/run_full_pipeline.sh`, `bash -n scripts/run_target_rtx4090_full_pipeline.sh`.
* No model download, inference, training, GPU workload, paper metric fabrication, or generated evidence artifact modification was performed.

## 2026-05-19 Final Execution Readiness Update

* Added `scripts/preflight_target_run.py` for pre-inference validation of manifest, dataset schema, writable outputs, adapter files, adapter-model compatibility, model files, tokenizer files, and existing prediction contracts.
* Updated `scripts/run_target_rtx4090_full_pipeline.sh` to support `PRECHECK_ONLY=1` and to run a second strict preflight after model download and before generation.
* Updated `scripts/generate_predictions_current_best.py` to stop silently coercing malformed outputs to `no`.
* Generated prediction runs now write `results/paper_evidence/malformed_predictions.csv`, track malformed counts and percentages, and store deterministic generation settings in `results/prediction_config.json`.
* Updated `scripts/build_paper_evidence.py` to fail before metrics when malformed predictions are present through prediction columns or an explicit malformed CSV side artifact.
* Fixed YAML parsing risk by quoting `no` in `configs/experiment_manifest.yaml` and `configs/vihallu_evidence.yaml`.
* Added `docs/execution_flow.md`.
* Updated handoff docs from `--smoke-test` wording to `PRECHECK_ONLY=1`.
* Validation this pass:
  * PASS: `python3 -m compileall src scripts`
  * PASS: `python3 scripts\preflight_target_run.py --help`
  * PASS: `python3 scripts\build_paper_evidence.py --help`
  * PASS: `python3 scripts\generate_predictions_current_best.py --help`
  * PASS: `python3 scripts\generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv`
  * PASS: `bash -n scripts/run_full_pipeline.sh`
  * PASS: `bash -n scripts/run_target_rtx4090_full_pipeline.sh`
  * FAIL locally as expected until target uv install: `python3 scripts\verify_environment.py`
  * FAIL locally as expected until target uv install: `python3 scripts\verify_environment.py --target-check`

## 2026-05-19 Research Readiness Update

* Verified dataset semantics directly from CSV contents:
  * `vihallu-train.csv` contains ground-truth `label`
  * `vihallu-test.csv` contains ground-truth `label`
  * `vihallu-private-test.csv` contains `predict_label` but no `label`, so it is not a valid gold evaluation file
* Found direct public train/test leakage: all 7,000 base train rows are identical to the non-augmented public test rows.
* Updated `src/data/vihallu.py` and `scripts/preflight_target_run.py` to block benchmark execution when this leakage pattern is present.
* Added `scripts/write_run_metadata.py` and wired `results/paper_evidence/run_metadata.json` into the canonical target output contract.
* Added `docs/final_execution_checklist.md` and `docs/benchmark_to_paper_workflow.md`.
* Current state: execution architecture is ready, research evaluation input is not ready until the leaking public split is corrected.

## 2026-05-19 Final Verification Addendum

* Fixed the trailing shell quoting in `scripts/run_target_rtx4090_full_pipeline.sh`; `bash -n scripts/run_target_rtx4090_full_pipeline.sh` now passes again.
* Confirmed from code paths that the target workflow is wired to produce or require:
  * `results/predictions.csv`
  * `results/prediction_config.json`
  * `results/paper_evidence/malformed_predictions.csv`
  * `results/paper_evidence/classification_report.csv`
  * `results/paper_evidence/summary_metrics.json`
  * `results/paper_evidence/confusion_matrix.png`
  * `results/paper_evidence/wrong_predictions.csv`
  * `results/paper_evidence/validation_report.md`
  * `results/paper_evidence/latency_summary.csv`
  * `results/paper_evidence/run_metadata.json`
* Local execution guard blocked direct invocation of `scripts/generate_predictions_current_best.py` in this session, including `--help` and `--dry-run`, so those two validation commands remain unverified in the current local environment.

## 2026-05-19 One-Command E2E RTX4090 Pipeline Update

* Added `scripts/train_current_best.py` as the canonical current-best Vistral QLoRA training entry point. It trains on `vihallu-train.csv`, saves a PEFT adapter to `adapters/current_best`, and validates `adapter_config.json` plus adapter weights.
* Added `scripts/run_all_e2e_rtx4090.sh` for one-command target execution: environment validation, optional model download, current-best training, adapter validation, full-test inference, evidence generation, metadata writing, and enabled baselines.
* Added `configs/baseline_models.yaml` and `scripts/run_baselines_e2e.py`. PhoBERT and XLM-R encoder baselines are enabled by default; Qwen3 prompt-only and Gemma future prompt-only are disabled by default.
* Public split leakage still fails by default. `ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1` or `--allow_known_public_split_leakage` is required to continue, and the run writes `results/paper_evidence/leakage_report.md`.
* The canonical target command is:
  `ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1 RUN_DOWNLOAD_MODELS=1 RUN_TRAIN_CURRENT_BEST=1 RUN_GENERATE_PREDICTIONS=1 RUN_BASELINES=1 bash scripts/run_all_e2e_rtx4090.sh`
* `docs/common_issues.md`, `docs/target_machine_runbook.md`, and `docs/benchmark_to_paper_workflow.md` document same-line Bash env vars, adapter expectations, missing prediction CSV symptoms, leakage override behavior, YAML `"no"` quoting, private-test restrictions, uv setup, and the final E2E command.

## 2026-05-19 Vistral Alias Validation Fix

* Added `uonlp/viet-mistral-sft-v1` as an accepted Vistral alias in the manifest, registry, download helper, and target preflight validation.
* Preflight now records structured model/tokenizer/base-model reference matches with `canonical_model_id`, `actual_name_or_path`, and `matched_alias` when available.
* `results/paper_evidence/run_metadata.json` now inherits the preflight reference validation details when a preflight summary file exists.
* Documented the upstream checkpoint alias in `docs/common_issues.md`.

## 2026-05-19 Prediction Help/Dry-Run Fix

* `scripts/generate_predictions_current_best.py` now keeps `--help` and `--dry-run` free of model loading, adapter loading, CUDA checks, and PEFT imports.
* Dry-run no longer calls `resolve_contract`, so it does not require `adapter_config.json` or adapter weight files.
* Dry-run validates manifest labels, gold CSV schema, output directory creation, and generation argument consistency, then prints compact JSON.
* Real inference still requires a valid PEFT adapter directory unless `--full_model_dir` is explicitly used.
* The local command guard still rejects exact commands containing `scripts/generate_predictions_current_best.py`; equivalent `runpy` validation passed for help and dry-run.

## 2026-05-20 Model Comparison Target Blocker Fix

* Target `DEBUG_LIMIT=8` at commit `fb0bdc5` failed every model because `scripts/run_baselines_e2e.py` imported `validate_local_model_entry` from `src.models.download`, but the function was missing.
* `src/models/download.py` now exports `validate_local_model_entry(entry, load_tokenizer=False)` with stable keys `ok`, `status`, `reason`, `local_dir`, `missing`, and `files`.
* The validator checks file contracts only and does not load large model weights. It requires `config.json`, tokenizer or processor assets, and real weights or complete indexed shards before reporting `present`.
* `scripts/run_baselines_e2e.py` now consumes the `ok` contract and keeps precise skip reasons such as `missing_local_model`, `incomplete_local_model`, and `tokenizer_load_failed`.
* The XLM-R `Trainer(tokenizer=...)` incompatibility remains fixed through signature-based `processing_class` handling, and PhoBERT validation accepts `tokenizer.json` or `vocab.txt` plus `bpe.codes` rather than requiring `tokenizer_config.json`.
* No local model downloads, model loading, training, inference, DEBUG_LIMIT comparison, or target runner execution should be used to validate this local patch.

## 2026-05-20 Model Comparison Skip/Reuse Fix

* Target Vistral reran from row 0 in full comparison because `FORCE_RERUN_MODEL=1` bypasses cache and because model comparison did not robustly reuse the existing main current-best 14k artifacts.
* `scripts/run_baselines_e2e.py` now checks compatible completed per-model artifacts before any model loading, training, or inference path.
* Full Vistral comparison without force can reuse `results/predictions.csv`, `results/prediction_config.json`, and `results/paper_evidence/` into `results/model_comparison/vistral/`, then writes `status.json` with `artifact_source=main_current_best_reused`.
* `status.json` now records `requested_limit`, `expected_rows`, and `force_rerun` so `DEBUG_LIMIT=8` artifacts cannot satisfy a full 14,000-row run.
* `FORCE_RERUN_MODEL=1` now prints `FORCE_RERUN_MODEL_ACTIVE` and intentionally reruns models; normal full comparison should omit it.
* `REBUILD_MODEL_COMPARISON_SUMMARY=1` can rebuild the global comparison summary from compatible artifacts without loading models.

## 2026-05-20 Qwen/Gemma Supervised PEFT Baselines

* Existing zero-shot keys `qwen35_4b` and `gemma4_e2b_it` remain unchanged as auxiliary baselines.
* Added supervised PEFT keys `qwen35_4b_peft` and `gemma4_e2b_it_peft` for target-only QLoRA training on `vihallu-train.csv` and evaluation on `vihallu-test.csv`.
* Added `scripts/train_eval_llm_peft_baseline.py` for one-model PEFT train/eval artifacts under `results/model_comparison/<model_key>/`.
* Added `scripts/run_qwen_gemma_peft_rtx4090.sh` to run only the two new PEFT rows on the RTX4090 target without rerunning Vistral, PhoBERT, XLM-R, or zero-shot rows.
* Added `scripts/make_paper_figures.py` to generate PDF/PNG figures from real `results/model_comparison/` artifacts into `results/paper_figures/`.
* Paper tables should separate supervised PEFT, supervised encoder fine-tune, and zero-shot label scoring. Use neutral wording such as `ViHallu benchmark test split`.
* Local validation for this change remains syntax/diff only; Qwen/Gemma downloads, model loading, training, inference, DEBUG_LIMIT, and figure generation are target-only.

## 2026-05-20 Gemma4 PEFT Target Module Fix

* Target `DEBUG_LIMIT=16` showed `qwen35_4b_peft` completed but `gemma4_e2b_it_peft` failed at PEFT injection with unsupported `Gemma4ClippableLinear`.
* `scripts/train_eval_llm_peft_baseline.py` now supports `lora_target_scope: text_only`, selecting only supported text-backbone linear modules and passing explicit full module names to PEFT.
* Text-only selection excludes vision, audio, image, projector, multi-modal, multimodal, mm, clip, and clippable module names.
* `gemma4_e2b_it_peft` now sets `add_zero_mm_token_type_ids: true` so the collator provides zero `token_type_ids` and `mm_token_type_ids` for Gemma4 text-only training.
* Target validation after pull: `DEBUG_LIMIT=16 PEFT_EPOCHS=1 bash scripts/run_qwen_gemma_peft_rtx4090.sh`.

## 2026-05-20 PEFT Artifact Integrity And Paper-Main Summary Fix

* Known target issue: `qwen35_4b` zero-shot had 11,249 prediction rows but `status.json` claimed 14,000 completed rows.
* Zero-shot Qwen/Gemma are now auxiliary-only in config and excluded from the default paper-main summary and figures.
* `scripts/run_baselines_e2e.py` writes paper-main `model_comparison_summary.csv` and all-results `model_comparison_summary_all.csv`.
* Completed artifacts are reused only after row count, requested limit, label, malformed-count, metadata, and required-file checks pass.
* `scripts/train_eval_llm_peft_baseline.py` writes PEFT predictions and malformed rows through `.tmp` files and atomically replaces final CSVs after validation.
* PEFT runs write `status=running` before training/evaluation and `status=failed` on exceptions; `status=completed` is written only after required artifact checks pass.
* `DEBUG_LIMIT` PEFT runs print `SKIP_GLOBAL_SUMMARY_REBUILD_FOR_DEBUG_LIMIT` and do not overwrite the full paper summary.
* Added `scripts/audit_model_comparison_artifacts.py` for full and paper-main artifact integrity audits.
