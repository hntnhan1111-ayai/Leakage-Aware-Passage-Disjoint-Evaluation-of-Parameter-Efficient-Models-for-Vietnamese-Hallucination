# RTX4090 Setup

Preferred setup uses uv.

```bash
cd ~/Hallu-Paper
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv venv --python 3.12
source .venv/bin/activate
uv pip install -U pip
uv pip install -r requirements.txt
printf 'HF_TOKEN=your_token_here\n' > .env
RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```

Fallback setup without uv:

```bash
cd ~/Hallu-Paper
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
printf 'HF_TOKEN=your_token_here\n' > .env
RUN_DOWNLOAD_MODELS=1 RUN_GENERATE_PREDICTIONS=1 bash scripts/run_target_rtx4090_full_pipeline.sh
```
