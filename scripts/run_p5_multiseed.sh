#!/usr/bin/env bash
set -euo pipefail
for seed in 42 43 44; do
  python scripts/11_train_peft.py --model-key qwen35_peft --seed "$seed"
  python scripts/11_train_peft.py --model-key gemma4_peft --seed "$seed"
done
python scripts/50_multiseed_summary.py --models qwen35_peft gemma4_peft --seeds 42 43 44
