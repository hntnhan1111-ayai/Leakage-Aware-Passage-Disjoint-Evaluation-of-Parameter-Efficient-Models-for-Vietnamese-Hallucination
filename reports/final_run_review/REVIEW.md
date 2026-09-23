# P1–P5 Final Run Review

- Generated: `2026-07-19T10:22:13.197032`
- Master log: `logs/resume_p1_p5_20260718_224141.log`
- Exit code: `0`
- Overall review: `PASS`

## Stage status

- `preflight`: exit `0`
- `p1_seed42`: exit `0`
- `p2_p4`: exit `0`
- `p5_multiseed`: exit `0`
- `paper_tables`: exit `0`

## Completed models

| Model | Seed | Accuracy | Macro-F1 | Weighted-F1 | Train seconds |
|---|---:|---:|---:|---:|---:|
| gemma4_peft | 42 | N/A | N/A | N/A | 4019.788597 |
| gemma4_peft | 43 | N/A | N/A | N/A | 4041.769666 |
| gemma4_peft | 44 | N/A | N/A | N/A | 4034.930967 |
| phobert | 42 | 0.743810 | 0.745213 | 0.743796 | 107.116076 |
| qwen35_peft | 42 | N/A | N/A | N/A | 3502.903172 |
| qwen35_peft | 43 | N/A | N/A | N/A | 10509.870737 |
| qwen35_peft | 44 | N/A | N/A | N/A | 10451.842794 |
| xlmr | 42 | 0.729524 | 0.729464 | 0.727705 | 216.307685 |

## Required artifacts

- [x] P1 main results
- [x] P2 context ablation
- [x] P3 statistical validation
- [x] P4 error analysis
- [x] P5 multi-seed

## Log diagnostics

- Fatal-marker lines: `0`
- Warning lines: `0`

## CSV integrity

| File | Rows | Columns | Empty cells | Status |
|---|---:|---:|---:|---|
| `results/context_ablation/qwen35_peft/seed_42/context_ablation_summary.csv` | 4 | 8 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/full/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/full/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/full/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/no_prompt/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/no_prompt/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/no_prompt/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/response_only/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/response_only/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/response_only/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/shuffled_context/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/shuffled_context/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/shuffled_context/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/context_ablation/qwen35_peft/seed_42/shuffled_context/shuffle_mapping.csv` | 1050 | 4 | 0 | ok |
| `results/error_analysis/qwen35_peft/seed_42/error_review_queue.csv` | 50 | 20 | 150 | ok |
| `results/error_analysis/qwen35_peft/seed_42/suggested_error_category_counts.csv` | 17 | 3 | 0 | ok |
| `results/main_results/main_results_seed_42.csv` | 4 | 18 | 0 | ok |
| `results/models/gemma4_peft/seed_42/dev/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/gemma4_peft/seed_42/dev/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/gemma4_peft/seed_42/dev/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/gemma4_peft/seed_42/test/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/gemma4_peft/seed_42/test/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/gemma4_peft/seed_42/test/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/gemma4_peft/seed_43/dev/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/gemma4_peft/seed_43/dev/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/gemma4_peft/seed_43/dev/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/gemma4_peft/seed_43/test/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/gemma4_peft/seed_43/test/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/gemma4_peft/seed_43/test/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/gemma4_peft/seed_44/dev/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/gemma4_peft/seed_44/dev/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/gemma4_peft/seed_44/dev/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/gemma4_peft/seed_44/test/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/gemma4_peft/seed_44/test/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/gemma4_peft/seed_44/test/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/phobert/seed_42/dev/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/phobert/seed_42/dev/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/phobert/seed_42/dev/predictions.csv` | 1050 | 6 | 0 | ok |
| `results/models/phobert/seed_42/test/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/phobert/seed_42/test/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/phobert/seed_42/test/predictions.csv` | 1050 | 6 | 0 | ok |
| `results/models/qwen35_peft/seed_42/dev/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/qwen35_peft/seed_42/dev/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/qwen35_peft/seed_42/dev/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/qwen35_peft/seed_42/test/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/qwen35_peft/seed_42/test/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/qwen35_peft/seed_42/test/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/qwen35_peft/seed_43/dev/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/qwen35_peft/seed_43/dev/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/qwen35_peft/seed_43/dev/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/qwen35_peft/seed_43/test/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/qwen35_peft/seed_43/test/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/qwen35_peft/seed_43/test/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/qwen35_peft/seed_44/dev/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/qwen35_peft/seed_44/dev/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/qwen35_peft/seed_44/dev/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/qwen35_peft/seed_44/test/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/qwen35_peft/seed_44/test/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/qwen35_peft/seed_44/test/predictions.csv` | 1050 | 15 | 0 | ok |
| `results/models/xlmr/seed_42/dev/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/xlmr/seed_42/dev/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/xlmr/seed_42/dev/predictions.csv` | 1050 | 6 | 0 | ok |
| `results/models/xlmr/seed_42/test/classification_report.csv` | 6 | 5 | 0 | ok |
| `results/models/xlmr/seed_42/test/confusion_matrix.csv` | 3 | 4 | 0 | ok |
| `results/models/xlmr/seed_42/test/predictions.csv` | 1050 | 6 | 0 | ok |
| `results/multiseed/mean_std_summary.csv` | 2 | 14 | 0 | ok |
| `results/multiseed/seed_level_metrics.csv` | 6 | 8 | 0 | ok |
| `results/statistics/seed_42/aligned_predictions.csv` | 1050 | 4 | 0 | ok |
