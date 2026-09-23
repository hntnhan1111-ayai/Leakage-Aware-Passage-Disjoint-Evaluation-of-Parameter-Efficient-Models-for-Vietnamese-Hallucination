# Repository Structure

This document explains each major directory and which files are canonical.

## Top-Level

| Path | Description |
|---|---|
| README.md | Main academic introduction and reproducibility guide |
| CITATION.cff | Citation metadata (provisional, no DOI assigned) |
| .gitignore | Excludes model weights, caches, virtual environments, LaTeX artifacts |
| pyproject.toml | Python package configuration and core dependencies |
| requirements.txt | Core (data-only) Python dependencies |
| requirements-gpu.txt | GPU/model training dependencies |
| BUNDLE_MANIFEST_SHA256.txt | SHA-256 checksums for artifact integrity |
| BUNDLE_README.txt | Summary of the original experiment bundle contents |

## configs/

| File | Description |
|---|---|
| data.yaml | Data paths, split proportions, seeds, near-duplicate detection settings |
| models.yaml | Frozen model IDs, revisions, adaptation settings, and training defaults |

## data/

| Path | Description |
|---|---|
| processed/train.csv | Passage-disjoint training split (4,900 rows) |
| processed/dev.csv | Passage-disjoint development split (1,050 rows) |
| processed/test.csv | Passage-disjoint held-out test split (1,050 rows) |
| raw/ | Acquisition instructions for original ViHallu data (not redistributed) |
| legacy/ | Quarantined 14,000-row artifact and documentation |
| README.md | Data provenance, licensing, and redistribution notes |

## docs/

| File | Description |
|---|---|
| CAMERA_READY_CONSISTENCY_AUDIT.md | Audit of the 0.8251 vs 0.8280 result discrepancy |
| reviewer_to_artifact_map.md | Maps reviewer concerns to existing repository artifacts |
| reproducibility_checklist.md | Verification checklist for reproducibility |
| repository_structure.md | This document |

## reports/

| Path | Description |
|---|---|
| dataset/AUDIT_SUMMARY.md | High-level dataset audit summary |
| dataset/clean_split_metadata.json | Frozen split parameters and group counts |
| dataset/clean_split_validation.json | Verified zero overlap across partitions |
| dataset/original_dataset_audit.json | Audit of original released data |
| dataset/quarantined_duplicate_test_diagnosis.json | Diagnosis of the legacy 14,000-row file |
| dataset/near_duplicate_context_audit.json | Near-duplicate context analysis |
| environment/cuda_stack.json | GPU environment snapshot |
| environment/public_model_snapshots.json | Model snapshot records |
| environment/uv-pip-freeze.txt | Python package versions |
| final_run_review/ | Final execution log review |

## results/

| Path | Description |
|---|---|
| main_results/main_results_seed_42.csv | Primary seed-42 results for all four models |
| main_results/main_results_seed_42.md | Human-readable summary of main results |
| multiseed/mean_std_summary.csv | Three-seed mean and std for PEFT models |
| multiseed/seed_level_metrics.csv | Per-seed metrics for PEFT models |
| context_ablation/qwen35_peft/seed_42/ | Full, no-prompt, response-only, and shuffled-context results |
| statistics/seed_42/statistical_validation.json | Bootstrap CI and McNemar test results |
| statistics/seed_42/aligned_predictions.csv | Aligned predictions for statistical testing |
| error_analysis/qwen35_peft/seed_42/ | Error review queue and suggested categories |

## scripts/

| Script | Description |
|---|---|
| 00_audit_and_split.py | Data audit and passage-disjoint split construction |
| 01_preflight.py | Environment preflight check |
| 02_setup_vncorenlp.py | VnCoreNLP setup |
| 03_download_models.py | Download frozen model revisions |
| 10_train_encoder.py | Encoder fine-tuning (PhoBERT, XLM-R) |
| 11_train_peft.py | PEFT fine-tuning (Qwen3.5, Gemma 4) |
| 12_run_main_models.py | Orchestrate main model runs |
| 13_collect_main_results.py | Collect main results into CSV |
| 20_context_ablation.py | Context-sensitivity ablation |
| 30_statistical_validation.py | Bootstrap CI and McNemar test |
| 40_error_analysis.py | Error review queue creation |
| 41_finalize_error_analysis.py | Finalize manually reviewed errors |
| 50_multiseed_summary.py | Compute multi-seed mean and std |
| 60_build_paper_tables.py | Generate LaTeX tables from saved results |
| run_p0_data.sh | Execute P0 (audit and split) |
| run_p1_main_seed42.sh | Execute P1 (main seed-42 runs) |
| run_p2_p4.sh | Execute P2-P4 (analysis) |
| run_p5_multiseed.sh | Execute P5 (multi-seed) |

## src/vihallu_repro/

| Module | Description |
|---|---|
| audit.py | Dataset audit functions |
| data.py | Data loading and preprocessing |
| environment.py | Environment metadata recording |
| error_analysis.py | Error analysis utilities |
| metrics.py | Metric computation |
| peft_runtime.py | PEFT model runtime (candidate-label NLL scoring) |
| prompts.py | Prompt construction |
| split.py | Passage-disjoint split construction |
| stats.py | Statistical validation (bootstrap, McNemar) |

## tests/

| File | Description |
|---|---|
| test_data_pipeline.py | Data pipeline unit tests |
| test_statistics.py | Statistical function unit tests |

## legacy/

Historical experiment files and Vistral-related code that are not part of the final four-model experiment matrix. Retained for reproducibility history only.
