# Canonical Files

This file defines the repository files that are authoritative for the ICIT 2026 evidence pipeline. Files not listed as `CANONICAL` must not be treated as the current validated execution path.

## CANONICAL Target Execution

* `scripts/run_all_e2e_rtx4090.sh` - canonical one-command RTX4090 train, inference, baseline, evidence, and metadata runner.
* `scripts/run_target_rtx4090_full_pipeline.sh` - legacy target RTX4090 adapter/prediction orchestration and `PRECHECK_ONLY=1` preflight mode.
* `scripts/run_full_pipeline.sh` - local-safe validation wrapper.
* `scripts/verify_environment.py` - dependency, manifest, dataset fallback, CUDA, bf16, and token-presence checks.
* `scripts/preflight_target_run.py` - end-to-end target preflight for environment, dataset, adapter, model, tokenizer, writable outputs, and existing prediction contracts.
* `scripts/train_current_best.py` - current-best Vistral QLoRA training entry point that saves `adapters/current_best`.
* `scripts/generate_predictions_current_best.py` - current prediction generation entry point for the Vistral adapter recipe.
* `scripts/build_paper_evidence.py` - evidence artifact builder and prediction CSV contract validator.
* `scripts/write_run_metadata.py` - target-side runtime metadata writer for reproducibility appendix inputs.
* `scripts/run_baselines_e2e.py` - baseline orchestration with non-strict optional failure handling.
* `scripts/audit_code_paper_consistency.py` - code-paper consistency and masked secret audit.
* `scripts/download_models.py` - target-only model download entry point.
* `scripts/update_paper_from_evidence.py` - guarded paper update after evidence exists.

## CANONICAL Configs And Source Modules

* `configs/experiment_manifest.yaml` - execution manifest for model IDs, dataset paths, generation defaults, seed, dtype, and LoRA assumptions.
* `configs/baseline_models.yaml` - enabled/disabled baseline registry and baseline training defaults.
* `configs/model_registry.yaml` - model registry used by download and audit workflows.
* `configs/vihallu_evidence.yaml` - evidence pipeline configuration reference.
* `src/data/vihallu.py` - ViHallu dataset and prediction schema validation.
* `src/models/download.py` - Hugging Face snapshot download helpers.
* `src/models/loader.py` - model loader helpers.
* `src/evaluation/metrics.py` - report and confusion-matrix generation.
* `src/evaluation/latency.py` - latency summary helpers.
* `src/utils/env.py` - `.env` and `HF_TOKEN` key handling without printing secrets.
* `src/utils/io.py` - output and artifact assertions.
* `src/utils/seed.py` - seed setup.

## EXPERIMENTAL

* `scripts/run_optional_qwen3_prompt_baseline.py` - optional prompt-only baseline implementation used only when enabled by baseline orchestration.
* `scripts/uit_our_method.py`
* `scripts/uit_r64.py`
* `scripts/uit_nojaccard.py`
* `scripts/uit_nolabelsmo.py`
* `scripts/uit_nomulti-prompts.py`
* `scripts/uit_phobert_best.py`
* `scripts/uit_phobert_best_dotproduct.py`
* `scripts/uit_phobert_best_half.py`
* `scripts/uit_phobert_best_notopk.py`
* `scripts/uit_phobert_best_top1.py`
* `scripts/uit_RoBERTa_best.py`
* `scripts/uit_RoBERTa_best_half.py`
* `scripts/uit_RoBERTa_best_half_aggressive.py`
* `scripts/uit_RoBERTa_best_half_base.py`
* `scripts/uit_RoBERTa_best_half_nostop.py`
* `scripts/uit_RoBERTa_best_half_partial.py`
* `scripts/download_vistral.py`
* `scripts/macro1_f1.py`

## LEGACY Root-Level Copies

The following root-level files are retained for auditability only. Do not use them as the execution path.

* `download_vistral.py`
* `macro1_f1.py`
* `uit_our_method.py`
* `uit_r64.py`
* `uit_nojaccard.py`
* `uit_nolabelsmo.py`
* `uit_nomulti-prompts.py`
* `uit_phobert_best.py`
* `uit_phobert_best_dotproduct.py`
* `uit_phobert_best_half.py`
* `uit_phobert_best_notopk.py`
* `uit_phobert_best_top1.py`
* `uit_RoBERTa_best.py`
* `uit_RoBERTa_best_half.py`
* `uit_RoBERTa_best_half_aggressive.py`
* `uit_RoBERTa_best_half_base.py`
* `uit_RoBERTa_best_half_nostop.py`
* `uit_RoBERTa_best_half_partial.py`
