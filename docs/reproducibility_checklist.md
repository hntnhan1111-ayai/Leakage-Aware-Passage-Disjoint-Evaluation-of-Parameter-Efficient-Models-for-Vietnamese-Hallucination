# Reproducibility Checklist

This checklist verifies that the camera-ready repository can be independently audited.

## Data and Split

- [x] Frozen data configuration in \configs/data.yaml- [x] Frozen model configuration in \configs/models.yaml- [x] Processed train/dev/test CSV files in \data/processed/- [x] Dataset audit report in eports/dataset/AUDIT_SUMMARY.md- [x] Split metadata in eports/dataset/clean_split_metadata.json- [x] Split validation in eports/dataset/clean_split_validation.json- [x] Quarantined legacy 14,000-row file documented and excluded from evaluation
- [x] Exact overlap across partitions verified as zero in \clean_split_validation.json
## Model Training and Evaluation

- [x] Model configurations frozen in \configs/models.yaml\ with exact revisions
- [x] Main seed-42 results in esults/main_results/main_results_seed_42.csv- [x] Multi-seed results for Qwen3.5 and Gemma 4 in esults/multiseed/- [x] Context ablation results in esults/context_ablation/- [x] Statistical validation in esults/statistics/seed_42/- [x] Error analysis in esults/error_analysis/- [x] Runtime metadata per run in \logs/
## Paper Artifacts

- [x] Camera-ready LaTeX source: \paper/camera_ready/main.tex- [x] References: \paper/camera_ready/references.bib- [x] Compiled PDF: \paper/camera_ready/main.pdf- [x] Page count recorded in \paper/camera_ready/PAGE_COUNT.txt- [x] Supporting tables in \paper/camera_ready/artifacts/
## Code

- [x] Source code in \src/vihallu_repro/- [x] Scripts in \scripts/- [x] Tests in \	ests/- [x] Import and file-reference checks pass

## Security

- [x] No secrets scanned and removed
- [x] No model weights committed
- [x] No Hugging Face cache committed
- [x] No virtual environments committed
- [x] No absolute local paths in public documentation

## Repository

- [x] Clean git status
- [x] \git diff --check\ passes
- [x] .gitignore updated
- [x] README.md rewritten from scratch
- [x] CITATION.cff created
- [x] All model HF links verified
