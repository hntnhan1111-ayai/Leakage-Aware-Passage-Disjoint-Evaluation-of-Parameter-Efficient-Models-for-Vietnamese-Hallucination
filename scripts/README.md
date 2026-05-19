# Scripts

Use `bash scripts/run_all_e2e_rtx4090.sh` on the RTX4090 target for the full ICIT 2026 train to inference to evidence pipeline.

Legacy experiment scripts remain in place for auditability. The validated evidence path is:

* `scripts/verify_environment.py`
* `scripts/preflight_target_run.py`
* `scripts/audit_code_paper_consistency.py`
* `scripts/download_models.py`
* `scripts/train_current_best.py`
* `scripts/generate_predictions_current_best.py`
* `scripts/build_paper_evidence.py`
* `scripts/write_run_metadata.py`
* `scripts/run_baselines_e2e.py`
* `scripts/run_all_e2e_rtx4090.sh`
* `scripts/run_target_rtx4090_full_pipeline.sh`
* `scripts/run_optional_qwen3_prompt_baseline.py`
* `scripts/run_full_pipeline.sh`

Run the canonical target E2E command with:

```bash
ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1 RUN_DOWNLOAD_MODELS=1 RUN_TRAIN_CURRENT_BEST=1 RUN_GENERATE_PREDICTIONS=1 RUN_BASELINES=1 bash scripts/run_all_e2e_rtx4090.sh
```

`scripts/run_target_rtx4090_full_pipeline.sh` remains as the legacy adapter/prediction runner.
