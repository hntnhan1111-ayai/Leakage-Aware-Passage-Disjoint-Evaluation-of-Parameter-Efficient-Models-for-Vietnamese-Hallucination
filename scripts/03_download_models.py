#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    selected = args.models or list(config["models"])
    records = []
    api = HfApi()
    for key in selected:
        entry = config["models"][key]
        local_dir = ROOT / entry["local_dir"]
        local_dir.mkdir(parents=True, exist_ok=True)
        requested_revision = entry.get("revision") or None
        info = api.model_info(entry["hf_id"], revision=requested_revision)
        resolved_revision = info.sha
        resolved = snapshot_download(
            repo_id=entry["hf_id"],
            revision=resolved_revision,
            local_dir=str(local_dir),
        )
        (local_dir / ".resolved_revision").write_text(resolved_revision + "\n", encoding="utf-8")
        records.append({"model_key": key, "hf_id": entry["hf_id"], "requested_revision": requested_revision, "resolved_revision": resolved_revision, "local_dir": str(local_dir), "resolved_path": resolved})
        print(f"DOWNLOADED {key} revision={resolved_revision} -> {resolved}")
    output = ROOT / "reports/model_downloads.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
