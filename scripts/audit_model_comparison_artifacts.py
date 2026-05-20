import argparse
import json
from pathlib import Path
import sys


LABELS = ["no", "intrinsic", "extrinsic"]


def load_yaml(path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except ModuleNotFoundError:
        return load_baseline_config_without_yaml(p)


def parse_scalar(value):
    text = value.strip()
    if text in ["true", "True"]:
        return True
    if text in ["false", "False"]:
        return False
    if text in ["null", "None", "~"]:
        return None
    if len(text) >= 2 and text[0] in ["'", '"'] and text[-1] == text[0]:
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_baseline_config_without_yaml(path):
    data = {"baselines": {}}
    in_baselines = False
    current = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        stripped = raw.strip()
        if indent == 0:
            in_baselines = stripped == "baselines:"
            current = None
            continue
        if not in_baselines:
            continue
        if indent == 2 and stripped.endswith(":"):
            key = stripped[:-1]
            data["baselines"][key] = {}
            current = data["baselines"][key]
            continue
        if indent == 4 and current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            value = value.strip()
            current[key.strip()] = parse_scalar(value) if value else None
    return data


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def int_or_none(value):
    if value in [None, ""]:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def paper_include(entry):
    return bool(entry.get("enabled")) and not bool(entry.get("auxiliary_only")) and entry.get("paper_include", True) is not False


def config_entries(config, paper_main_only, include_disabled):
    baselines = config.get("baselines", {}) if isinstance(config, dict) else {}
    items = {}
    for key, entry in baselines.items():
        if not include_disabled and not entry.get("enabled"):
            continue
        if paper_main_only and not paper_include(entry):
            continue
        items[key] = dict(entry)
    return items


def status_dirs(root):
    result = {}
    for path in sorted(Path(root).glob("*/status.json")):
        result[path.parent.name] = path.parent
    return result


def label_distribution(path):
    p = Path(path)
    if not p.exists():
        return 0, {}, "missing_predictions"
    import csv

    with p.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "predict_label" not in (reader.fieldnames or []):
            rows = list(reader)
            return len(rows), {}, "missing_predict_label"
        counts = {label: 0 for label in LABELS}
        bad_values = set()
        rows = 0
        for row in reader:
            rows += 1
            value = str(row.get("predict_label", ""))
            if value == "":
                return rows, counts, "nan_predict_label"
            if value in counts:
                counts[value] += 1
            else:
                bad_values.add(value)
    bad = sorted(bad_values)
    if bad:
        return rows, counts, "invalid_labels:" + ",".join(bad)
    return rows, counts, ""


def inspect_model(root, model_key, entry, expected_rows):
    out = Path(root) / model_key
    status_path = out / "status.json"
    pred_path = out / "predictions.csv"
    row = {
        "model_key": model_key,
        "method_type": entry.get("method_type", ""),
        "status": "missing_status",
        "prediction_rows": "",
        "status_rows": "",
        "expected_rows": expected_rows,
        "requested_limit": "",
        "label_distribution": "{}",
        "integrity": "missing_status",
        "skip_reason": "",
    }
    if not status_path.exists():
        return row, False
    status = read_json(status_path)
    pred_rows, counts, label_error = label_distribution(pred_path)
    status_rows = int_or_none(status.get("rows"))
    status_expected_rows = int_or_none(status.get("expected_rows"))
    requested_limit = int_or_none(status.get("requested_limit"))
    malformed_count = int_or_none(status.get("malformed_count"))
    row.update({
        "method_type": status.get("method_type") or entry.get("method_type", ""),
        "status": status.get("status", ""),
        "prediction_rows": pred_rows,
        "status_rows": status.get("rows", ""),
        "requested_limit": status.get("requested_limit", ""),
        "label_distribution": json.dumps(counts, ensure_ascii=False, sort_keys=True),
    })
    problems = []
    if status.get("status") == "completed":
        required = [
            pred_path,
            out / "summary_metrics.json",
            out / "classification_report.json",
            out / "classification_report.csv",
            out / "confusion_matrix.csv",
            out / "confusion_matrix.png",
            out / "malformed_predictions.csv",
            out / "latency_summary.json",
            out / "latency_summary.csv",
            out / "prediction_config.json",
        ]
        missing = [path.name for path in required if not path.exists() or path.stat().st_size == 0]
        if missing:
            problems.append("missing_required_files:" + ",".join(missing))
        if pred_rows != int(expected_rows):
            problems.append(f"prediction_rows_mismatch:expected={expected_rows}:found={pred_rows}")
        if status_rows != int(expected_rows):
            problems.append(f"status_rows_mismatch:expected={expected_rows}:found={status.get('rows')}")
        if status_expected_rows is not None and status_expected_rows != int(expected_rows):
            problems.append(f"expected_rows_mismatch:expected={expected_rows}:found={status_expected_rows}")
        if requested_limit is not None:
            problems.append(f"requested_limit_not_full:{requested_limit}")
        if label_error:
            problems.append(label_error)
        if malformed_count not in [0, None]:
            problems.append(f"malformed_count:{malformed_count}")
    if problems:
        row["integrity"] = "invalid"
        row["skip_reason"] = "artifact_integrity_error:" + ";".join(problems)
        return row, True
    row["integrity"] = "ok" if status.get("status") == "completed" else "not_completed"
    row["skip_reason"] = status.get("skip_reason", "")
    return row, False


def write_report(rows):
    columns = ["model_key", "method_type", "status", "prediction_rows", "status_rows", "expected_rows", "requested_limit", "label_distribution", "integrity", "skip_reason"]
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(col, "")) for col in columns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/model_comparison")
    parser.add_argument("--config", default="configs/baseline_models.yaml")
    parser.add_argument("--expected-rows", type=int, default=14000)
    parser.add_argument("--paper-main-only", action="store_true")
    parser.add_argument("--include-disabled", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    entries = config_entries(config, args.paper_main_only, args.include_disabled)
    dirs = status_dirs(args.root)
    keys = sorted(set(entries) | set(dirs))
    rows = []
    failed = False
    for key in keys:
        entry = entries.get(key, {"model_key": key})
        if args.paper_main_only and key not in entries:
            continue
        row, bad = inspect_model(args.root, key, entry, args.expected_rows)
        rows.append(row)
        failed = failed or bad
    write_report(rows)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
