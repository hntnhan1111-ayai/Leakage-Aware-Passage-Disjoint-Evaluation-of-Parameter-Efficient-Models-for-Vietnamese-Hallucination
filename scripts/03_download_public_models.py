#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/models.yaml'
MODEL_KEYS = ['phobert', 'xlmr', 'qwen35_peft', 'gemma4_peft']

cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
api = HfApi()
records = []
cache_dir = Path(os.environ['HF_HUB_CACHE'])
cache_dir.mkdir(parents=True, exist_ok=True)

for key in MODEL_KEYS:
    entry = cfg['models'][key]
    repo_id = entry['hf_id']
    print(f'=== {key}: {repo_id} ===', flush=True)
    info = api.model_info(repo_id=repo_id, token=False)
    revision = info.sha
    names = [s.rfilename for s in (info.siblings or [])]
    has_safetensors = any(name.endswith('.safetensors') for name in names)
    ignore = ['*.h5', '*.msgpack', '*.ot', '*.onnx', '*.gguf']
    if has_safetensors:
        ignore.append('pytorch_model*.bin')
    snapshot = Path(snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir),
        token=False,
        force_download=False,
        max_workers=4,
        ignore_patterns=ignore,
    )).resolve()
    entry['revision'] = revision
    entry['local_dir'] = str(snapshot)
    records.append({
        'model_key': key,
        'hf_id': repo_id,
        'revision': revision,
        'snapshot': str(snapshot),
    })
    print(f'READY {key}: {snapshot}', flush=True)

CONFIG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
out = ROOT / 'reports/environment/public_model_snapshots.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
