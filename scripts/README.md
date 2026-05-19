# Scripts

Use `bash scripts/run_full_pipeline.sh` from the repository root for the ICIT 2026 evidence pipeline.

Legacy experiment scripts remain in place for auditability. The validated evidence path is:

* `scripts/verify_environment.py`
* `scripts/preflight_target_run.py`
* `scripts/audit_code_paper_consistency.py`
* `scripts/download_models.py`
* `scripts/generate_predictions_current_best.py`
* `scripts/build_paper_evidence.py`
* `scripts/write_run_metadata.py`
* `scripts/run_target_rtx4090_full_pipeline.sh`
* `scripts/run_optional_qwen3_prompt_baseline.py`
* `scripts/run_full_pipeline.sh`

Run target preflight with:

```bash
PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```
