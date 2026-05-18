# Target RTX4090 uv Setup

Run these commands on the Linux Ubuntu RTX4090 machine after pulling this branch.

```bash
cd ~/Hallu-Paper
git fetch origin
git checkout submit/icit2026-evidence-revision
git pull

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

uv venv --python 3.12
source .venv/bin/activate
uv pip install -U pip
uv pip install -r requirements.txt

printf 'HF_TOKEN=your_token_here\n' > .env

RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

To evaluate an existing prediction file instead of generating predictions:

```bash
PRED_CSV=final_submission_scratch.csv bash scripts/run_target_rtx4090_full_pipeline.sh
```

Do not run model download, inference, training, or optional baselines on the local Windows/WSL editing machine.
