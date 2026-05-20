import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_yaml(path):
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model comparison config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a YAML mapping")
    return data


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {p}")
    return p


def enabled_entries(config, only_model_key=None):
    entries = []
    for key, item in config.get("baselines", {}).items():
        if only_model_key and key != only_model_key:
            continue
        if not item.get("enabled", False):
            continue
        entries.append((key, item))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_models.yaml")
    parser.add_argument("--report-json", default="results/model_comparison/download_report.json")
    parser.add_argument("--only-model-key", default=None)
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    from src.models.download import download_entry
    from src.utils.env import load_hf_token

    token = load_hf_token(required=True)
    config = load_yaml(args.config)
    items = []
    for key, item in config.get("baselines", {}).items():
        if args.only_model_key and key != args.only_model_key:
            continue
        if not args.include_disabled and not item.get("enabled", False):
            continue
        items.append((key, item))
    results = []
    for key, item in items:
        result = download_entry(key, item, token=token)
        results.append(result)
        print(f"{result['status'].upper()} {key} {result['local_dir']}")
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "results": results,
    }
    write_json(args.report_json, report)
    failed = [item for item in results if item["status"] == "failed"]
    if failed:
        print(f"DOWNLOAD_FAILURES {len(failed)} see {args.report_json}")
        if args.strict:
            raise RuntimeError(f"One or more model downloads failed. See {args.report_json}")


if __name__ == "__main__":
    main()
