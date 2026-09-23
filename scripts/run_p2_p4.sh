#!/usr/bin/env bash
set -euo pipefail
python scripts/20_context_ablation.py --model-key qwen35_peft --seed 42
python scripts/30_statistical_validation.py --models qwen35_peft gemma4_peft --seed 42 --iterations 10000
python scripts/40_error_analysis.py --model-key qwen35_peft --seed 42 --count 50
python scripts/60_build_paper_tables.py --seed 42
