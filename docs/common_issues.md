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

## Malformed Vistral Generations

Symptom:

```text
Malformed generation detected ... raw_output contains copied context text instead of no/intrinsic/extrinsic
```

The fix should not map copied context to `no`. The current-best generator uses strict label-only prompts, decodes only newly generated tokens, retries once with an ultra-strict prompt, and then uses deterministic label scoring over `no`, `intrinsic`, and `extrinsic` if generation remains unparsable.

Final paper runs must still finish with `malformed_count=0` in `results/prediction_config.json` and zero rows in `results/paper_evidence/malformed_predictions.csv`.

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
