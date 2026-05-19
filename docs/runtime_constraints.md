# Runtime Constraints

## Local Machine

* Local Windows/WSL is for editing, lightweight validation, and Git operations only.
* Do not run model downloads locally.
* Do not run inference locally.
* Do not run training locally.
* Do not run GPU workloads locally.

## Target RTX4090 Machine

* The Linux Ubuntu RTX4090 machine is the authoritative execution environment.
* The target machine can run model downloads, prediction generation, evaluation, optional baselines, and paper-evidence generation.
* Use the target machine for any command that touches CUDA or Hugging Face model artifacts.

## CUDA Assumptions

* RTX4090-class CUDA execution is expected on the target machine.
* `bfloat16` is the default dtype for causal LLM loading and generation.
* NF4 4-bit loading is the default for large causal LLM inference paths when configured.
* `device_map="auto"` is used for model placement where applicable.

## uv Workflow

* Prefer `uv venv --python 3.12`.
* Activate with `source .venv/bin/activate`.
* Install with `uv pip install -r requirements.txt`.
* Keep `requirements.txt` available as the compatibility path.
* Do not install packages into global system Python.
