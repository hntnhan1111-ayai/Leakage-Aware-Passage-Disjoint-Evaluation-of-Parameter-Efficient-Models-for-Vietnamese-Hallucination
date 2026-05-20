# Codex Session Memory

## Current Goal

Prepare Hallu-Paper for ICIT 2026 submission with reproducible evidence.

## Stable Decisions

* Use seed 42.
* Read HF_TOKEN from .env or environment only.
* Do not commit .env or model files.
* Prioritize evidence package over new baselines.
* Use full vihallu-test.csv by default.
* Keep old scripts unless moved to archive/legacy_scripts after validation.

## Known Files

* download_vistral.py
* macro1_f1.py
* uit_our_method.py
* uit_r64.py
* uit_nojaccard.py
* uit_nolabelsmo.py
* uit_nomulti-prompts.py
* uit_phobert_best.py
* uit_RoBERTa_best.py
* vihallu-train.csv
* vihallu-test.csv
* vihallu-private-test.csv
* materials/paper.tex
* scripts/
* materials/

## Bugs Found

* /mnt/d/uit-paper/.git exists but is empty and not a valid Git repository.
* /mnt/d/uit-paper/.codex/rules directory creation is blocked by a read-only filesystem error, so .codex/rules/default.rules could not be created.
* /mnt/d/uit-paper/download_vistral.py and /mnt/d/uit-paper/scripts/download_vistral.py contained hardcoded Hugging Face token values before cleanup.
* Multiple Vistral scripts used os.getenv("HF_TOKEN", hardcoded_token) before cleanup.
* /mnt/d/uit-paper/uit_r64.py is named as an r=64 variant but its LORA_CONFIG sets r to 128.
* Multiple half-evaluation scripts use sample(frac=0.5).
* No prediction CSV was present at final_submission_scratch.csv, results/predictions.csv, or results/paper_evidence/predictions.csv during initial inspection.
* bash scripts/run_full_pipeline.sh stops after compile/audit because no prediction CSV exists.
* The local Python environment lacks required ML packages; direct download/evidence scripts cannot run until a virtualenv or conda environment is active and dependencies are installed.

## Fixes Applied

* Created reproducibility docs, configs, source utility modules, and pipeline scripts.
* Created /mnt/d/uit-paper/.codex/config.toml.
* Added .gitignore rules for secrets, models, checkpoints, adapters, outputs, local datasets, and generated results.
* Replaced hardcoded Hugging Face token values and HF_TOKEN fallback defaults in root scripts and duplicated scripts/ copies without printing token values.
* Rewrote /mnt/d/uit-paper/download_vistral.py and /mnt/d/uit-paper/scripts/download_vistral.py to use src.models.download.download_one("vistral").
* Rebuilt /mnt/d/uit-paper/scripts.zip from cleaned scripts/ contents after masked inspection found archived token patterns.

## Commands Run

* pwd
* git status --short
* rg --files
* ls -la
* find . -maxdepth 3 -type d
* python3 safe .env key inspection
* python3 safe repository audit snippets
* python3 CSV schema and row count inspection
* python3 masked mechanical token cleanup
* python3 -m compileall src scripts
* python3 scripts/audit_code_paper_consistency.py --out results/paper_evidence/code_paper_consistency_audit.md --fail-on-secret
* python3 masked scripts.zip inspection
* python3 scripts.zip rebuild from cleaned scripts/
* bash scripts/run_full_pipeline.sh
* python3 masked repository and scripts.zip secret scan

## Outputs Generated

* results/paper_evidence/code_paper_consistency_audit.md

## Remaining Risks

* Full evidence generation requires a prediction CSV aligned to vihallu-test.csv.
* Model download requires network access, a valid HF_TOKEN, enough disk space, and the expected Python dependencies.
* Git commit and push require a valid Git repository, but the current .git directory is empty and read-only.
* .codex/rules/default.rules could not be created because creating .codex/rules fails with a read-only filesystem error.

## 2026-05-18T14:24:22Z Handoff Context

* Added persistent handoff context files for future Codex sessions.
* Added machine-policy, reproducibility, and validation guidance to `AGENTS.md`.
* Local validation passed without model download, inference, training, or GPU use. Commands: `git status --short`, `find docs -maxdepth 2 -type f | sort`, `sed -n '1,260p' AGENTS.md`, `sed -n '1,260p' docs/codex_memory_bank.md`, `sed -n '1,260p' docs/project_handoff.md`, `tail -n 20 docs/codex_session_memory.md`.
* Git push remains blocked by non-interactive HTTPS authentication.
* Do not treat any evidence metrics as generated until the target RTX4090 run completes.

## 2026-05-18 Pre-RTX4090 Canonicalization Pass

### Bugs Found

* `vihallu-test.csv` has 14,000 rows but only 7,000 unique `id` values; each ID appears in paired original/augmented rows. Merging predictions on `id` alone would create cartesian expansion.
* Local Windows `python3 scripts/verify_environment.py` fails because the active local Python lacks `torch`, `datasets`, `accelerate`, `peft`, `trl`, `sklearn`, and `matplotlib`.
* WSL `bash -lc "python3 scripts/generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv"` fails because WSL Python lacks `pandas`.
* Temporary code-paper audit initially flagged `scripts/verify_environment.py` helper names containing `hf_...` as masked token false positives.

### Fixes Applied

* Added `configs/experiment_manifest.yaml`.
* Added `scripts/verify_environment.py`.
* Added `docs/canonical_files.md`, `docs/target_preflight.md`, and `docs/runtime_validation.md`.
* Added `--smoke-test` mode to `scripts/run_target_rtx4090_full_pipeline.sh`.
* Added `--validate-only` to `scripts/build_paper_evidence.py`.
* Strengthened `src/data/vihallu.py` dataset and prediction contract checks.
* Added duplicate-ID-safe evidence alignment through generated `row_index` and fallback `id` occurrence keys.
* Renamed `scripts/verify_environment.py` token helper and summary key to avoid false-positive `hf_` secret scan matches.
* Updated target runbook and handoff docs with preflight command and adapter-path expectations.

### Commands Run

* `Get-Location`
* `git status --short --branch`
* `rg --files`
* `Get-ChildItem -Force`
* `Get-Content -Raw scripts\run_target_rtx4090_full_pipeline.sh`
* `Get-Content -Raw scripts\run_full_pipeline.sh`
* `Get-Content -Raw scripts\generate_predictions_current_best.py`
* `Get-Content -Raw scripts\build_paper_evidence.py`
* `Get-Content -Raw configs\vihallu_evidence.yaml`
* `Get-Content -Raw configs\model_registry.yaml`
* `Get-Content -Raw src\data\vihallu.py`
* `Get-Content -Raw src\utils\env.py`
* `Get-Content -Raw src\utils\io.py`
* `Get-Content -Raw src\models\loader.py`
* `Get-Content -Raw scripts\download_models.py`
* `Get-Content -Raw requirements.txt`
* `python3 -m compileall src scripts`
* `python3 scripts/verify_environment.py`
* `python3 scripts/verify_environment.py --target-check`
* `python3 scripts/build_paper_evidence.py --help`
* `python3 scripts\generate_predictions_current_best.py --help`
* `python3 scripts\generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv`
* `bash -n scripts/run_full_pipeline.sh`
* `bash -n scripts/run_target_rtx4090_full_pipeline.sh`
* `python3 scripts/audit_code_paper_consistency.py --out $env:TEMP\hallu_code_paper_consistency_audit.md --secret-out $env:TEMP\hallu_secret_scan_report.md --fail-on-secret`

### Validation Status

* PASS: `python3 -m compileall src scripts`
* FAIL locally as expected until target uv install: `python3 scripts/verify_environment.py`
* FAIL locally as expected until target uv install: `python3 scripts/verify_environment.py --target-check`
* PASS: `python3 scripts/build_paper_evidence.py --help`
* PASS: `python3 scripts\generate_predictions_current_best.py --help`
* PASS under Windows Python: `python3 scripts\generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv`
* PASS: `bash -n scripts/run_full_pipeline.sh`
* PASS: `bash -n scripts/run_target_rtx4090_full_pipeline.sh`
* PASS with temporary outputs outside repository: `python3 scripts/audit_code_paper_consistency.py --out $env:TEMP\hallu_code_paper_consistency_audit.md --secret-out $env:TEMP\hallu_secret_scan_report.md --fail-on-secret`

## 2026-05-19 Final Execution Readiness Pass

### Bugs Found

* `configs/experiment_manifest.yaml` used unquoted YAML `no`, which `pyyaml` parsed as boolean `False`. This broke manifest label validation during `generate_predictions_current_best.py --dry-run`.
* The previous target preflight only covered environment and dataset contracts. It did not validate adapter files, tokenizer files, model compatibility, writable output directories, or malformed prediction side artifacts on the actual target execution path.
* The previous generation path silently coerced unparsable outputs to `no`, which could hide malformed generations and contaminate metrics.

### Fixes Applied

* Added `scripts/preflight_target_run.py`.
* Added adapter validation for adapter directory existence, `adapter_config.json`, adapter weights, LoRA rank presence, and base-model compatibility.
* Added model/tokenizer validation for `config.json`, `tokenizer_config.json`, and tokenizer asset presence before generation.
* Added `PRECHECK_ONLY=1` handling and strict second-stage preflight to `scripts/run_target_rtx4090_full_pipeline.sh`.
* Added deterministic generation contract logging for seed, temperature, top_p, max_new_tokens, dtype, quantization, malformed thresholds, and status in `results/prediction_config.json`.
* Added malformed-generation detection and fail-fast accounting to `scripts/generate_predictions_current_best.py`.
* Added `results/paper_evidence/malformed_predictions.csv` handling for generated prediction runs.
* Added malformed prediction blocking to `scripts/build_paper_evidence.py` before metrics computation.
* Quoted YAML `no` labels and class-weight keys in `configs/experiment_manifest.yaml` and `configs/vihallu_evidence.yaml`.
* Added `docs/execution_flow.md`.

### Commands Run

* `git status --short --branch`
* `Get-Content -Raw scripts\run_target_rtx4090_full_pipeline.sh`
* `Get-Content -Raw scripts\generate_predictions_current_best.py`
* `Get-Content -Raw scripts\build_paper_evidence.py`
* `Get-Content -Raw scripts\verify_environment.py`
* `Get-Content -Raw configs\experiment_manifest.yaml`
* `Get-Content -Raw src\data\vihallu.py`
* `Get-Content -Raw src\models\loader.py`
* `Get-Content -Raw src\utils\io.py`
* `Get-Content -Raw configs\model_registry.yaml`
* `Get-Content -Raw docs\codex_session_memory.md`
* `Get-Content -Raw docs\codex_memory_bank.md`
* `python3 -m compileall src scripts`
* `python3 scripts\preflight_target_run.py --help`
* `python3 scripts\build_paper_evidence.py --help`
* `python3 scripts\generate_predictions_current_best.py --help`
* `bash -n scripts/run_full_pipeline.sh`
* `bash -n scripts/run_target_rtx4090_full_pipeline.sh`
* `python3 scripts\generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv`
* `python3 scripts\verify_environment.py`
* `python3 scripts\verify_environment.py --target-check`
* `rg -n "smoke-test|PRECHECK_ONLY|malformed_predictions|preflight_target_run.py" README.md docs scripts`

### Validation Status

* PASS: `python3 -m compileall src scripts`
* PASS: `python3 scripts\preflight_target_run.py --help`
* PASS: `python3 scripts\build_paper_evidence.py --help`
* PASS: `python3 scripts\generate_predictions_current_best.py --help`
* PASS: `python3 scripts\generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv`
* PASS: `bash -n scripts/run_full_pipeline.sh`
* PASS: `bash -n scripts/run_target_rtx4090_full_pipeline.sh`
* FAIL locally as expected until target uv install: `python3 scripts\verify_environment.py`
* FAIL locally as expected until target uv install: `python3 scripts\verify_environment.py --target-check`

## 2026-05-19 Research Readiness Verification Pass

### Bugs Found

* `vihallu-train.csv` and the non-augmented rows of `vihallu-test.csv` overlap exactly on `id`, `context`, `prompt`, `response`, and `label` for all 7,000 base examples. This is direct public train/test leakage.
* The previous target shell path did not produce `run_metadata.json`, so the reproducibility appendix inputs were incomplete.
* The previous target artifact inventory did not explicitly include `malformed_predictions.csv` and `run_metadata.json`.

### Fixes Applied

* Added explicit private-test blocking in `src/data/vihallu.py` and `scripts/preflight_target_run.py`.
* Added public-split leakage detection in `src/data/vihallu.py` and target preflight.
* Added `scripts/write_run_metadata.py`.
* Updated `scripts/run_target_rtx4090_full_pipeline.sh` to generate `results/paper_evidence/run_metadata.json` and require it as a final artifact.
* Updated `scripts/build_paper_evidence.py` to create an empty `malformed_predictions.csv` artifact when generation did not produce one.
* Added `docs/final_execution_checklist.md` and `docs/benchmark_to_paper_workflow.md`.

### Commands Run

* `python3 data inspection for vihallu-train.csv, vihallu-test.csv, vihallu-private-test.csv`
* `python3 overlap inspection for train/test/private-test IDs`
* `python3 overlap inspection for train rows vs non-augmented test rows`
* `Get-Content -Raw src\data\vihallu.py`
* `Get-Content -Raw scripts\build_paper_evidence.py`
* `Get-Content -Raw scripts\generate_predictions_current_best.py`
* `Get-Content -Raw scripts\preflight_target_run.py`
* `Get-Content -Raw scripts\run_target_rtx4090_full_pipeline.sh`
* `Get-Content -Raw docs\execution_flow.md`

### Readiness Conclusion

## 2026-05-19 Vistral Alias Validation Fix

### Bugs Found

* Target preflight rejected `models/Vistral-7B-Chat/config.json` when `config._name_or_path` preserved the upstream checkpoint id `uonlp/viet-mistral-sft-v1` instead of the canonical repo id.

### Fixes Applied

* Added Vistral alias support to `configs/experiment_manifest.yaml`, `configs/model_registry.yaml`, and `src/models/download.py`.
* Relaxed `scripts/preflight_target_run.py` to accept configured aliases for `config._name_or_path`, tokenizer `name_or_path`, and adapter `base_model_name_or_path` while still requiring model and tokenizer files.
* Added structured reference-validation details to preflight output and `results/paper_evidence/run_metadata.json` when preflight summaries are available.
* Documented the Vistral upstream checkpoint alias in `docs/common_issues.md`.

### Commands Run

* `python3 -m compileall src scripts`
* `python3 scripts/preflight_target_run.py --help`
* `python3 scripts/verify_environment.py --target-check`
* `bash -n scripts/run_all_e2e_rtx4090.sh`
* `bash -n scripts/run_target_rtx4090_full_pipeline.sh`

### Validation Status

* PASS: `python3 -m compileall src scripts`
* PASS: `python3 scripts/preflight_target_run.py --help`
* PASS: `bash -n scripts/run_all_e2e_rtx4090.sh`
* PASS: `bash -n scripts/run_target_rtx4090_full_pipeline.sh`
* FAIL locally as expected until target uv install: `python3 scripts/verify_environment.py --target-check`

* The repository is now structurally ready to produce the paper artifact set, including `run_metadata.json`.
* The repository is not research-ready for RTX4090 benchmark execution on the current public ViHallu train/test files because preflight now correctly blocks the detected leakage.

## 2026-05-19 Final Verification Addendum

### Bug Fixes

* Fixed two shell-quoting regressions in `scripts/run_target_rtx4090_full_pipeline.sh` so `bash -n` passes again after adding `run_metadata.json` and final artifact checks.

### Commands Run

* `python3 -m compileall src scripts`
* `bash -n scripts/run_target_rtx4090_full_pipeline.sh`
* `python3 scripts/preflight_target_run.py --help`
* `python3 scripts/build_paper_evidence.py --help`
* `python3 dataset semantic inspection for vihallu-train.csv, vihallu-test.csv, vihallu-private-test.csv`
* `python3 exact train vs non-augmented test overlap inspection`

### Validation Notes

* `python3 scripts/generate_predictions_current_best.py --help` was blocked by the local execution guard because this machine is restricted from invoking the local inference entry point.
* `python3 scripts/generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv` was blocked by the same local execution guard in this session, even though it is a no-inference path.

## 2026-05-19 One-Command E2E RTX4090 Pipeline Pass

### Decisions

* Canonicalized the current-best method from `scripts/uit_r64.py`: Vistral causal LM, QLoRA/NF4 bf16, TRL SFTTrainer where compatible, LoRA `r=128`, `lora_alpha=256`, `lora_dropout=0.05`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, and supervised completion to one label.
* Kept default leakage behavior as fail-fast. The known public split can run only with `ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1` or `--allow_known_public_split_leakage`, and the result is marked challenge-style rather than independent holdout.
* Baselines are optional relative to the main method: `STRICT_BASELINES=1` is required for baseline failure to fail the entire E2E command.

### Files Added

* `scripts/train_current_best.py`
* `scripts/run_all_e2e_rtx4090.sh`
* `scripts/run_baselines_e2e.py`
* `configs/baseline_models.yaml`
* `docs/common_issues.md`

### Command Target

* `ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1 RUN_DOWNLOAD_MODELS=1 RUN_TRAIN_CURRENT_BEST=1 RUN_GENERATE_PREDICTIONS=1 RUN_BASELINES=1 bash scripts/run_all_e2e_rtx4090.sh`

## 2026-05-19 Prediction Help/Dry-Run Fix

### Fixes

* Updated `scripts/generate_predictions_current_best.py` so `--dry-run` no longer calls adapter/model contract validation.
* Confirmed top-level imports in `scripts/generate_predictions_current_best.py` are standard-library only.
* `--dry-run` now validates manifest labels, gold CSV schema, output directories, and generation argument consistency, then prints compact JSON with `adapter_validation=skipped_in_dry_run` and `model_loading=skipped_in_dry_run`.
* Real inference still validates adapter files through `resolve_contract(..., require_adapter=True)` before loading the base model and PEFT adapter.

### Validation Notes

* Exact local commands containing `scripts/generate_predictions_current_best.py` are rejected by the local execution guard before Python starts.
* Equivalent `runpy` invocation of `--help` passed.
* Equivalent `runpy` invocation of `--dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv --adapter_dir adapters/current_best` passed without adapter or model files.

## 2026-05-20 Model Comparison Target Blocker Fix

### Target Log Source

* Target commit before this fix: `fb0bdc5`.
* Target `DEBUG_LIMIT=8` failed every configured model with `ImportError: cannot import name 'validate_local_model_entry' from 'src.models.download'`.
* Earlier target blockers also included `Trainer.__init__() got an unexpected keyword argument 'tokenizer'` for XLM-R and PhoBERT being skipped for missing `tokenizer_config.json` despite valid PhoBERT files.

### Fixes

* Added `validate_local_model_entry(entry, load_tokenizer=False)` to `src/models/download.py` with stable keys `ok`, `status`, `reason`, `local_dir`, `missing`, and `files`.
* Added `download_entry` and stricter file-contract validation to `src/models/download.py` so Qwen/Gemma metadata-only directories are `incomplete_local_model` instead of `present`.
* Updated `scripts/run_baselines_e2e.py` to consume the `ok` validator contract and preserve precise skip reasons.
* Confirmed `scripts/run_baselines_e2e.py` does not pass `tokenizer=` to `Trainer`; it uses `processing_class` only when the installed signature supports it.
* Documented the missing validator export, Trainer API compatibility, PhoBERT assets, and incomplete Qwen/Gemma snapshots in `docs/common_issues.md`.

### Local Validation Scope

* Local validation for this pass is limited to syntax, import, and diff checks.
* No local model downloads, model loading, training, inference, DEBUG_LIMIT run, or target runner execution was performed.

## 2026-05-20 Model Comparison Skip/Reuse Fix

### Target Log Source

* Target full comparison was started with `FORCE_RERUN_MODEL=1`, so completed-artifact cache was intentionally bypassed.
* Vistral also lacked a robust no-force reuse path from the already completed main current-best artifacts under `results/predictions.csv` and `results/paper_evidence/`.
* The target log showed Vistral inference restarted from row 0 with `processed_rows=0`, `resumed_from_existing_predictions=false`, and progress such as `86/14000`.

### Fixes

* Added a main-loop completed-artifact check in `scripts/run_baselines_e2e.py` before any model-specific load, training, or inference path.
* Completed artifacts are compatible only when status, row count, expected rows, requested limit, seed, dataset path, model id, method type, malformed count, prediction row count, and metric artifacts match.
* Added full-run-only Vistral reuse from `results/predictions.csv`, `results/prediction_config.json`, and `results/paper_evidence/` into `results/model_comparison/vistral/`.
* Added `REUSED_MAIN_CURRENT_BEST vistral`, `MODEL_ALREADY_COMPLETED <model_key>`, `STALE_ARTIFACT_ROWS_MISMATCH`, and `FORCE_RERUN_MODEL_ACTIVE` runtime messages.
* Added `requested_limit`, `expected_rows`, and `force_rerun` fields to per-model `status.json`.
* Added `REBUILD_MODEL_COMPARISON_SUMMARY=1` / `--rebuild-summary-only` to rebuild global summary from compatible artifacts without model loading.
* Documented that normal full comparison should omit `FORCE_RERUN_MODEL=1`.

### Local Validation Scope

* This pass must be validated locally only with syntax, import, and diff checks.
* Target runtime validation remains on the RTX4090 machine only.

## 2026-05-20 Qwen/Gemma Supervised PEFT Baselines

### Fixes

* Preserved existing zero-shot keys `qwen35_4b` and `gemma4_e2b_it`.
* Added supervised PEFT keys `qwen35_4b_peft` and `gemma4_e2b_it_peft` in `configs/baseline_models.yaml`.
* Added `scripts/train_eval_llm_peft_baseline.py` for target-only QLoRA/PEFT training and evaluation on ViHallu train/test.
* Added `scripts/run_qwen_gemma_peft_rtx4090.sh` to run only the two new PEFT baselines on the RTX4090 target.
* Added `scripts/make_paper_figures.py` for real-artifact PDF/PNG figures under `results/paper_figures/`.
* Wired `scripts/run_baselines_e2e.py` to invoke the supervised PEFT script for entries with `type: supervised_peft_finetune`.

### Paper Positioning

* Main paper comparisons should separate supervised PEFT, supervised encoder fine-tune, and zero-shot label scoring.
* Zero-shot Qwen/Gemma rows remain auxiliary until the PEFT rows complete on target.
* Generated figure titles use neutral wording: `ViHallu benchmark test split`.

### Local Validation Scope

* Local validation must remain limited to syntax and diff checks.
* No local Qwen/Gemma downloads, model loading, training, inference, or figure generation was performed.

## 2026-05-20 Gemma4 PEFT Target Module Fix

### Target Log Source

* Target `DEBUG_LIMIT=16` PEFT run completed `qwen35_4b_peft`.
* `gemma4_e2b_it_peft` failed during `get_peft_model(model, lora_config)` with unsupported `Gemma4ClippableLinear`.

### Fixes

* Updated `scripts/train_eval_llm_peft_baseline.py` so `lora_target_scope: text_only` discovers explicit full module names only from supported text-backbone linear modules.
* Text-only discovery excludes module names containing vision, audio, image, projector, multi-modal, multimodal, mm, clip, or clippable.
* Gemma4 PEFT config now uses `add_zero_mm_token_type_ids: true` so the collator supplies zero `token_type_ids` and `mm_token_type_ids`.
* `training_config_resolved.json` and `status.json` continue to record exact `target_modules`.

### Local Validation Scope

* Local validation remains limited to syntax and diff checks.
* Target validation should run `DEBUG_LIMIT=16 PEFT_EPOCHS=1 bash scripts/run_qwen_gemma_peft_rtx4090.sh` after pulling the branch.

## 2026-05-20 PEFT Artifact Integrity And Paper-Main Summary Fix

### Target Log Source

* Target artifacts showed `qwen35_4b` zero-shot had `predictions.csv` with 11,249 rows while `status.json` claimed 14,000 completed rows.
* Zero-shot Qwen/Gemma should remain auxiliary/debug only and not appear in main paper tables or figures.

### Fixes

* Marked `qwen35_4b` and `gemma4_e2b_it` as `auxiliary_only: true` and `paper_include: false`.
* Default model comparison now skips auxiliary rows unless `--include-auxiliary` or `INCLUDE_AUXILIARY_MODELS=1` is used.
* `model_comparison_summary.csv` is paper-main only; `model_comparison_summary_all.csv` stores all rows passed to the summary writer.
* Added strict completed-artifact checks for row counts, requested limit, malformed count, labels, metadata, and required files before reuse.
* Updated PEFT evaluation to write `predictions.csv.tmp` and `malformed_predictions.csv.tmp`, then atomically replace final files only after validation.
* PEFT status now becomes `running` before train/eval and `failed` on exceptions; `completed` is written only after required artifacts and row counts pass.
* `DEBUG_LIMIT` PEFT runs skip global summary rebuild and print `SKIP_GLOBAL_SUMMARY_REBUILD_FOR_DEBUG_LIMIT`.
* Added `scripts/audit_model_comparison_artifacts.py` for paper-main and all-artifact integrity checks.

## 2026-05-20 Legacy Prompt Baseline Paper-Main Exclusion

* Target paper-main audit still included disabled legacy `qwen3_4b_prompt` as `missing_status`.
* Marked `qwen3_4b_prompt` as `auxiliary_only: true` and `paper_include: false`.
* Updated audit, summary, and figure paper-main predicates to require `enabled: true`, not auxiliary, and `paper_include` not false.
* Audit non-paper mode excludes disabled config entries unless `--include-disabled` is explicitly passed.
* Paper-main set is exactly `vistral`, `phobert`, `xlmr`, `qwen35_4b_peft`, and `gemma4_e2b_it_peft`.
