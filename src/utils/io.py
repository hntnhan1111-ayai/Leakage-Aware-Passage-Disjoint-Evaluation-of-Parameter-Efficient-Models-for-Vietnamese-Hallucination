from pathlib import Path
import json


def ensure_dir(path):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def assert_nonempty_files(paths):
    missing = []
    for item in paths:
        p = Path(item)
        if not p.exists() or p.stat().st_size == 0:
            missing.append(str(p))
    if missing:
        raise RuntimeError("Missing or empty artifacts: " + ", ".join(missing))
    return True


def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p
