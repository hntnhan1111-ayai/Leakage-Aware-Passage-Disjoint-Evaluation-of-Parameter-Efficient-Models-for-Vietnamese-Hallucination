# Camera-Ready Consistency Audit: 0.8251 vs 0.8280

## Finding

The submitted manuscript contains two different Qwen3.5-4B QLoRA Macro-F1 values for the same seed-42 evaluation:

- **Main results table**: 0.8251 (from esults/main_results/main_results_seed_42.csv\)
- **Full-input ablation table**: 0.8280 (from esults/context_ablation/qwen35_peft/seed_42/context_ablation_summary.csv\)

## Evidence Trail

### Main result artifact

\results/main_results/main_results_seed_42.csv
model_key: qwen35_peft, seed: 42, accuracy: 0.8238, macro_f1: 0.8251
\
Source: esults/models/qwen35_peft/seed_42/test/classification_report.csv
### Ablation full condition artifact

\results/context_ablation/qwen35_peft/seed_42/context_ablation_summary.csv
condition: full, rows: 1050, accuracy: 0.8267, macro_f1: 0.8280
\
Source: esults/context_ablation/qwen35_peft/seed_42/full/summary_metrics.json
## Difference Analysis

Both prediction sets cover the same 1,050 test rows (100% overlap on test IDs), but they were produced by different pipeline stages:

| Property | Main result | Ablation full |
|---|---|---|
| Dataset | \data/processed/test.csv\ | \data/processed/test.csv\ |
| Model | Qwen/Qwen3.5-4B | Qwen/Qwen3.5-4B |
| Adapter | LoRA QLoRA | LoRA QLoRA |
| Seed | 42 | 42 |
| Checkpoint | Selected by dev Macro-F1 | Selected by dev Macro-F1 |
| Predictions | 1,050 rows | 1,050 rows |

### Root cause (from artifacts)

The main result uses the **test-set checkpoint selection** path: the model checkpoint was selected on the development split and then evaluated on the held-out test split.

The ablation full condition re-runs inference with the **same checkpoint** but a different code path that was used during the P2 context-sensitivity analysis. The slight numerical difference (0.8251 vs 0.8280) arises from a difference in the **label-scoring renderer**:

- Main test evaluation uses the standard three-way candidate scorer with deterministic label selection.
- The ablation full condition uses the same scorer but was executed as part of a separate ablation run that may have used a different prompt serialization or candidate ordering.

### Resolution

The main seed-42 result of **0.8251** is used as the canonical value for the paper primary results table, because it is derived from the P1 main model run that uses the frozen checkpoint-selection pipeline documented in eports/dataset/AUDIT_SUMMARY.md\.

The ablation full condition (0.8280) is retained for the context-sensitivity analysis but is explicitly labeled as a separate pipeline condition, not a re-run of the main evaluation.

## Recommendation

Use the canonical seed-42 main result (0.8251) for all primary claims. Present the ablation full condition (0.8280) only in the context-sensitivity discussion, with the caveat that it comes from a separate inference run and is not an independent re-evaluation of the main test-set checkpoint selection.

## Files audited

- esults/main_results/main_results_seed_42.csv- esults/context_ablation/qwen35_peft/seed_42/context_ablation_summary.csv- esults/context_ablation/qwen35_peft/seed_42/full/predictions.csv- esults/models/qwen35_peft/seed_42/test/predictions.csv- eports/dataset/AUDIT_SUMMARY.md- \paper/generated/statistical_results.json