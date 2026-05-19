# Target Preflight

Run preflight on the Ubuntu RTX4090 machine after pulling the branch and installing dependencies. It does not download models, load full models, run inference, train, or create paper evidence metrics.

## Environment Setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
printf 'HF_TOKEN=your_token_here\n' > .env
```

## Single Preflight Command

Use `PRECHECK_ONLY=1` for the final no-inference readiness pass.

For an existing prediction CSV:

```bash
PRED_CSV=final_submission_scratch.csv PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

For target generation when model files and adapter files are already present:

```bash
RUN_GENERATE_PREDICTIONS=1 ADAPTER_DIR=adapters/current_best PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

If model download will happen in the full run, use `PRECHECK_ONLY=1` only after the model directory exists. The normal full pipeline still runs a second strict preflight after download and before generation.

## Standalone Target Preflight

```bash
python3 scripts/preflight_target_run.py \
  --manifest configs/experiment_manifest.yaml \
  --model_key vistral \
  --model_dir models/Vistral-7B-Chat \
  --gold_csv vihallu-test.csv \
  --adapter_dir adapters/current_best \
  --out_dir results/paper_evidence \
  --pred_out_csv results/predictions.csv \
  --config_json results/prediction_config.json \
  --malformed_csv results/paper_evidence/malformed_predictions.csv \
  --require_model_dir \
  --require_adapter \
  --require_active_env \
  --require_hf_token \
  --require_cuda \
  --require_bf16 \
  --require_bitsandbytes \
  --precheck_only
```

## Standalone Environment Check

```bash
python3 scripts/verify_environment.py --target-check --require-active-env --require-hf-token --require-cuda --require-bf16 --require-bitsandbytes
```

## What Preflight Validates

* Python imports and package versions for the required runtime stack.
* Active environment presence when invoked through the target script.
* `HF_TOKEN` key presence in `.env` or the process environment without printing the value.
* CUDA availability and bf16 support on the target GPU.
* `configs/experiment_manifest.yaml` structure, seed, labels, generation defaults, dtype assumptions, and LoRA assumptions.
* Dataset fallback discovery for `vihallu-test.csv`.
* Dataset schema and required gold CSV columns.
* Adapter directory existence, `adapter_config.json`, adapter weight file, base-model compatibility, and LoRA rank presence.
* Model directory, `config.json`, tokenizer config, and tokenizer asset presence when generation is part of the run.
* Writable output directories for predictions, config JSON, malformed CSV, and evidence outputs.
* Prediction CSV schema and label contract when `PRED_CSV` is provided.
* Malformed prediction accounting before evidence generation when a malformed CSV is part of the run.

## Expected Outcome

A passing precheck prints `Target precheck complete. No models were loaded and no inference was run.` A failure should identify the exact missing package, file, directory, label, column, CUDA capability, token key, adapter-model mismatch, tokenizer mismatch, or malformed prediction condition before expensive execution starts.
