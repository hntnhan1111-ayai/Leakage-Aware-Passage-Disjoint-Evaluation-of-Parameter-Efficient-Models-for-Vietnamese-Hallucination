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
