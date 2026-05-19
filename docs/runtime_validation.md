# Runtime Validation

The pipeline fails before expensive target work when inputs, dependencies, or execution assumptions do not match the contract.

## Environment Contract

Use:

```bash
python3 scripts/verify_environment.py
python3 scripts/verify_environment.py --target-check
```

Target execution uses stricter flags through `scripts/run_target_rtx4090_full_pipeline.sh`:

```bash
python3 scripts/verify_environment.py --target-check --require-active-env --require-hf-token --require-cuda --require-bf16 --require-bitsandbytes
```

The verifier imports required packages, records versions, validates `configs/experiment_manifest.yaml`, checks dataset fallback paths, reports CUDA/device assumptions, and never prints `HF_TOKEN`.

## Target Preflight Contract

Use:

```bash
python3 scripts/preflight_target_run.py --help
PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

`scripts/preflight_target_run.py` validates the execution path beyond package imports. It checks manifest selection, dataset schema, writable output directories, adapter files, adapter-model compatibility, tokenizer presence, and existing prediction contracts before expensive model execution starts.

## Dataset Schema Contract

Gold CSV files must contain exactly these required columns:

* `id`
* `context`
* `prompt`
* `response`
* `label`

Valid labels are exactly:

* `no`
* `intrinsic`
* `extrinsic`

`src/data/vihallu.py` rejects missing columns, empty required values, invalid labels, and gold files whose observed label set is not exactly `no`, `intrinsic`, `extrinsic`. Duplicate IDs are detected with exact CSV rows. The full `vihallu-test.csv` currently contains paired original and augmented rows with duplicate `id` values, so the evidence path uses `row_index` when present and otherwise uses an `id` occurrence key to avoid cartesian merges.

## Prediction CSV Contract

Prediction CSV files must contain:

* `id`
* `predict_label`

`scripts/build_paper_evidence.py` validates prediction files before writing artifacts. It rejects missing columns, empty prediction values, invalid labels, row-count mismatches, missing gold rows, extra prediction IDs, and mismatched duplicate-ID occurrence counts.

If `is_malformed` is present in the prediction CSV or a non-empty malformed CSV is passed through `--malformed_csv`, evidence generation fails before metrics computation.

For a no-write contract check:

```bash
python3 scripts/build_paper_evidence.py --validate-only --gold_csv vihallu-test.csv --pred_csv results/predictions.csv
```

## Malformed Generation Contract

`scripts/generate_predictions_current_best.py` writes malformed rows to `results/paper_evidence/malformed_predictions.csv`, records malformed counts and percentages in `results/prediction_config.json`, and fails on the first malformed output by default through `--max_malformed_count 0` and `--max_malformed_rate 0.0`.

Malformed outputs include:

* Empty generations.
* Unparsable generations that do not map to `no`, `intrinsic`, or `extrinsic`.
* Any row where one or more template votes fail parsing.

## Target Precheck Contract

Use:

```bash
PRECHECK_ONLY=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

Precheck validates imports, configs, dataset paths, prediction contracts or adapter discovery, CUDA, bf16, package versions, writable outputs, and adapter/tokenizer compatibility. It does not download models, load full models, run inference, train, or write evidence metrics.

## Runtime Logging

Target execution logs the manifest path, seed, gold CSV, model key, model directory, output paths, generation flags, and package/device summary before model work. `scripts/generate_predictions_current_best.py` writes `results/prediction_config.json` with seed, temperature, top_p, max_new_tokens, dtype, malformed policy, malformed counts, and status for every real generation attempt.
