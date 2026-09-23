#!/usr/bin/env bash
set -euo pipefail
python scripts/01_preflight.py --require-cuda --require-local-models
python scripts/12_run_main_models.py --seed 42 --models phobert xlmr qwen35_peft gemma4_peft
