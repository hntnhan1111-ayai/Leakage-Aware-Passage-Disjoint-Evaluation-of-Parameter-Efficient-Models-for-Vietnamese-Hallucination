# Common Issues

## Bash Environment Variables

Environment variables must be on the same command line as the command or exported first.

Correct:

```bash
ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1 RUN_DOWNLOAD_MODELS=1 RUN_TRAIN_CURRENT_BEST=1 RUN_GENERATE_PREDICTIONS=1 RUN_BASELINES=1 bash scripts/run_all_e2e_rtx4090.sh
```

Also correct:

```bash
export ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1
export RUN_DOWNLOAD_MODELS=1
export RUN_TRAIN_CURRENT_BEST=1
export RUN_GENERATE_PREDICTIONS=1
export RUN_BASELINES=1
bash scripts/run_all_e2e_rtx4090.sh
```

## Missing Adapter

Symptom:

```text
Missing PEFT adapter files
```

The current-best inference path expects:

* `adapters/current_best/adapter_config.json`
* `adapters/current_best/adapter_model.safetensors` or `adapters/current_best/adapter_model.bin`

Run the full E2E command with `RUN_TRAIN_CURRENT_BEST=1` so `scripts/train_current_best.py` trains and saves the adapter before inference.

## Vistral Base Model Alias

Symptom:

```text
models/Vistral-7B-Chat/config.json base model reference mismatch
```

Hugging Face configs can preserve the upstream checkpoint name in `config._name_or_path` or `tokenizer_config.json` even when the local directory is `models/Vistral-7B-Chat`.

For Vistral, the canonical repo id stays `Viet-Mistral/Vistral-7B-Chat`, but the saved config may report `uonlp/viet-mistral-sft-v1` internally. The target preflight accepts either alias and still fails on missing files or unrelated architectures.

## Missing Prediction CSV

Symptom:

```text
Missing or empty prediction CSV: results/predictions.csv
```

Run with `RUN_GENERATE_PREDICTIONS=1`. The one-command runner writes `results/predictions.csv` after loading `models/Vistral-7B-Chat` and the PEFT adapter from `adapters/current_best`.

## Missing Model Validator Export

Symptom:

```text
ImportError: cannot import name 'validate_local_model_entry' from 'src.models.download'
```

The full model-comparison runner imports `validate_local_model_entry` from `src.models.download`. That function must exist and return a stable dictionary with `ok`, `status`, `reason`, `local_dir`, `missing`, and `files`. It is a file-contract validator only; it must not load large model weights.

## Transformers Trainer Tokenizer Argument

Symptom:

```text
TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'
```

Transformers 5.x does not accept `tokenizer=` in the `Trainer` constructor. The model-comparison runner keeps `DataCollatorWithPadding(tokenizer=tokenizer)`, uses `processing_class=tokenizer` only when the installed `Trainer` signature supports it, and otherwise omits the tokenizer object from `Trainer`.

## PhoBERT Tokenizer Assets

Symptom:

```text
phobert skipped with missing_required_files:tokenizer_config.json
```

`models/phobert-base-v2` can be valid without `tokenizer_config.json` when it has usable tokenizer assets. The target validator accepts:

* `config.json`,
* one real model weight file such as `pytorch_model.bin`, `model.safetensors`, or a complete safetensors index with shards,
* `tokenizer.json`, or `vocab.txt` plus `bpe.codes`.

PhoBERT tokenizer loading retries with `use_fast=False` in the target runtime path.

## Incomplete Qwen Or Gemma Downloads

Symptom:

```text
DOWNLOADED qwen35_4b models/Qwen3.5-4B
```

with a very small downloaded byte count, or a later skip reason containing:

```text
incomplete_local_model
```

A model directory is not considered present just because `config.json` exists. The downloader and comparison runner require tokenizer or processor assets plus real model weights or complete indexed shard files. With `RUN_DOWNLOAD_MODELS=0`, incomplete Qwen/Gemma directories are kept as skipped rows. With `RUN_DOWNLOAD_MODELS=1`, the target downloader attempts repair and records the final status in `results/model_comparison/download_report.json`.

Download failures for optional comparison models do not stop the whole E2E command unless `STRICT_BASELINES=1` is set.

## Model Comparison Reruns Vistral

Symptom:

```text
processed_rows=0
resumed_from_existing_predictions=false
```

or Vistral progress starts again from `0/14000` during a full model-comparison run.

Do not use `FORCE_RERUN_MODEL=1` for a normal full comparison when completed artifacts already exist. That flag intentionally bypasses completed-artifact checks and reruns models.

Normal full comparison should be run without force:

```bash
ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1 RUN_DOWNLOAD_MODELS=0 RUN_TRAIN_CURRENT_BEST=0 RUN_GENERATE_PREDICTIONS=0 RUN_BASELINES=1 FULL_MODEL_COMPARISON=1 bash scripts/run_all_e2e_rtx4090.sh
```

The comparison runner first checks `results/model_comparison/<model_key>/status.json`, `predictions.csv`, and metric artifacts. Compatible completed artifacts remain `status=completed` in the global summary and are not loaded again.

For Vistral full runs only, if model-comparison artifacts are missing or stale but the main current-best artifacts are compatible, the runner copies them from:

```text
results/predictions.csv
results/prediction_config.json
results/paper_evidence/
```

into:

```text
results/model_comparison/vistral/
```

and prints:

```text
REUSED_MAIN_CURRENT_BEST vistral
```

Debug artifacts with `DEBUG_LIMIT=8` are not compatible with full `rows=14000` runs.

## Qwen And Gemma Fair Comparison

The existing `qwen35_4b` and `gemma4_e2b_it` rows are zero-shot label-scoring baselines. Keep them as auxiliary baselines and do not rename them.

The supervised PEFT rows use separate model keys:

* `qwen35_4b_peft`
* `gemma4_e2b_it_peft`

Run only those two PEFT rows on the RTX4090 target with:

```bash
DEBUG_LIMIT=16 bash scripts/run_qwen_gemma_peft_rtx4090.sh
```

Then run the full PEFT pass without `DEBUG_LIMIT` after the debug pass succeeds:

```bash
bash scripts/run_qwen_gemma_peft_rtx4090.sh
```

Do not use private-test labels in tables. Use neutral wording such as `ViHallu benchmark test split` for captions and section text.

## Gemma4 PEFT Target Module Failure

Symptom:

```text
ValueError: Target module Gemma4ClippableLinear(...) is not supported.
```

This is a Gemma 4 and PEFT/LoRA compatibility issue. For text-only ViHallu training, do not target image, audio, projector, multimodal, clip, or clippable modules.

The PEFT baseline trainer uses `lora_target_scope: text_only` for Qwen/Gemma PEFT entries. It discovers explicit supported text-backbone module names under `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, and records the exact list in `training_config_resolved.json`.

For Gemma4, the trainer also supports zero `token_type_ids` and `mm_token_type_ids` through `add_zero_mm_token_type_ids: true` to make the next text-only collator failure explicit instead of hidden.

## Prediction Help And Dry Run

`scripts/generate_predictions_current_best.py --help` must be a pure argparse path. It should not check model files, adapter files, CUDA, `HF_TOKEN`, or import PEFT.

`scripts/generate_predictions_current_best.py --dry-run` validates only lightweight execution contract details:

* manifest path and labels,
* gold CSV path and schema,
* output directory creation,
* generation argument consistency.

Dry-run does not require:

* `adapters/current_best/adapter_config.json`
* `adapters/current_best/adapter_model.safetensors`
* `adapters/current_best/adapter_model.bin`
* `models/Vistral-7B-Chat`
* CUDA
* `HF_TOKEN`

## Leakage Guard

Default behavior fails when public train/test leakage is detected. This prevents accidentally reporting the public split as an independent holdout estimate.

To intentionally run the known public split as a challenge-style evaluation, set:

```bash
ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1
```

When this override is used, the pipeline writes `results/paper_evidence/leakage_report.md`, and metadata marks `leakage_override=true` and `challenge_style_evaluation=true`.

## YAML Label Quoting

The label `no` must be quoted in YAML:

```yaml
labels:
  - "no"
  - intrinsic
  - extrinsic
```

Unquoted `no` can be parsed as a boolean by YAML tooling.

## Private Test Is Not Gold

`vihallu-private-test.csv` does not contain usable ground-truth labels. It must not be passed as `--gold_csv` to evidence generation.

## uv Target Setup

Use a project environment, not global Python:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## One-Command E2E

Run this on the RTX4090 machine after the branch is pulled and the uv environment is active:

```bash
ALLOW_KNOWN_PUBLIC_SPLIT_LEAKAGE=1 RUN_DOWNLOAD_MODELS=1 RUN_TRAIN_CURRENT_BEST=1 RUN_GENERATE_PREDICTIONS=1 RUN_BASELINES=1 bash scripts/run_all_e2e_rtx4090.sh
```
