# Lightweight Local Validation Commands

Run only these on the local Windows/WSL editing machine:

```bash
git rev-parse --is-inside-work-tree
git status --short
python3 -m compileall src scripts
python3 scripts/preflight_target_run.py --help
python3 scripts/audit_code_paper_consistency.py --out results/paper_evidence/code_paper_consistency_audit.md --fail-on-secret
python3 scripts/verify_environment.py
python3 scripts/verify_environment.py --target-check
python3 scripts/build_paper_evidence.py --help
python3 scripts/generate_predictions_current_best.py --help
python3 scripts/generate_predictions_current_best.py --dry-run --gold_csv vihallu-test.csv --out_csv results/predictions.csv
python3 scripts/download_models.py --help
python3 scripts/update_paper_from_evidence.py --help
bash -n scripts/run_full_pipeline.sh
bash -n scripts/run_target_rtx4090_full_pipeline.sh
git status --short
git diff --stat
```

Do not run model downloads, inference, training, optional Qwen baseline, or the target full pipeline on the local machine.
