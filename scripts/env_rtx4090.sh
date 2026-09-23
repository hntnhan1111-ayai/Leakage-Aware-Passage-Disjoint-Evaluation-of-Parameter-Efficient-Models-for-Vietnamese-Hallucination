#!/usr/bin/env bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
export PATH="$JAVA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$JAVA_HOME/lib:${LD_LIBRARY_PATH:-}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export HF_HOME="$ROOT/.hf"
export HF_HUB_CACHE="$ROOT/.hf/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
