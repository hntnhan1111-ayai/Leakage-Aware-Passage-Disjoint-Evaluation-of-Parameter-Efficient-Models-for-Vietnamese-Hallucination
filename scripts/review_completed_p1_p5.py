#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = ROOT / "reports" / "final_run_review" / stamp
output_dir.mkdir(parents=True, exist_ok=True)

master_logs = sorted(
    LOGS.glob("resume_p1_p5_*.log"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

master_log = master_logs[0] if master_logs else None
log_text = (
    master_log.read_text(encoding="utf-8", errors="replace")
    if master_log
    else ""
)

exitcode_path = LOGS / "resume_p1_p5_latest.exitcode"
exit_code = (
    exitcode_path.read_text(encoding="utf-8").strip()
    if exitcode_path.exists()
    else "MISSING"
)

stage_starts = re.findall(r"^STAGE=([^\n]+)", log_text, re.MULTILINE)
stage_exits = re.findall(
    r"^STAGE_EXIT_CODE=(\d+)",
    log_text,
    re.MULTILINE,
)
failed_stages = re.findall(
    r"^FAILED_STAGE=([^\n]+)",
    log_text,
    re.MULTILINE,
)

fatal_patterns = [
    r"Traceback \(most recent call last\)",
    r"CUDA out of memory",
    r"OutOfMemoryError",
    r"No space left on device",
    r"Segmentation fault",
    r"died with <Signals\.",
    r"FAILED_STAGE=",
]

fatal_matches: list[str] = []

for line_number, line in enumerate(log_text.splitlines(), start=1):
    if any(re.search(pattern, line, re.IGNORECASE) for pattern in fatal_patterns):
        fatal_matches.append(f"L{line_number}: {line}")

warning_lines = [
    f"L{line_number}: {line}"
    for line_number, line in enumerate(log_text.splitlines(), start=1)
    if "warning" in line.lower()
]

completed_files = sorted(RESULTS.glob("models/**/COMPLETED.json"))

model_rows: list[dict[str, Any]] = []

for path in completed_files:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        model_rows.append({
            "model": path.parents[1].name,
            "seed": path.parent.name,
            "error": f"{type(exc).__name__}: {exc}",
        })
        continue

    test_metrics = payload.get("metrics", {}).get("test", {})

    model_rows.append({
        "model": payload.get("model_key", path.parents[1].name),
        "seed": payload.get("seed", path.parent.name),
        "accuracy": test_metrics.get("accuracy"),
        "macro_f1": test_metrics.get("macro_f1"),
        "weighted_f1": test_metrics.get("weighted_f1"),
        "train_seconds": payload.get("train_seconds"),
        "best_checkpoint": payload.get("best_checkpoint"),
        "completed_file": str(path.relative_to(ROOT)),
    })

artifact_files = sorted(
    path
    for path in RESULTS.rglob("*")
    if path.is_file()
    and path.suffix.lower() in {
        ".json", ".csv", ".tex", ".png", ".pdf", ".md"
    }
)

artifact_names = [
    str(path.relative_to(ROOT))
    for path in artifact_files
]

required_groups = {
    "P1 main results": [
        "main_results",
        "models/",
    ],
    "P2 context ablation": [
        "context_ablation",
        "ablation",
    ],
    "P3 statistical validation": [
        "statistics",
        "bootstrap",
        "mcnemar",
    ],
    "P4 error analysis": [
        "error_analysis",
        "error_review",
    ],
    "P5 multi-seed": [
        "multiseed",
        "multi_seed",
    ],
}

required_status: dict[str, bool] = {}

lower_artifacts = [name.lower() for name in artifact_names]

for group, keywords in required_groups.items():
    required_status[group] = any(
        any(keyword in artifact for keyword in keywords)
        for artifact in lower_artifacts
    )

csv_checks: list[dict[str, Any]] = []

for path in sorted(RESULTS.rglob("*.csv")):
    record: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "rows": None,
        "columns": None,
        "empty_cells": None,
        "status": "unknown",
    }

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        columns = reader.fieldnames or []
        empty_cells = sum(
            value is None or str(value).strip() == ""
            for row in rows
            for value in row.values()
        )

        record.update({
            "rows": len(rows),
            "columns": len(columns),
            "empty_cells": empty_cells,
            "status": "ok",
        })
    except Exception as exc:
        record["status"] = f"{type(exc).__name__}: {exc}"

    csv_checks.append(record)

review_passed = (
    exit_code == "0"
    and not failed_stages
    and all(required_status.values())
    and len(completed_files) >= 4
)

report = {
    "created_at": datetime.now().isoformat(),
    "project_root": str(ROOT),
    "master_log": str(master_log.relative_to(ROOT)) if master_log else None,
    "recorded_exit_code": exit_code,
    "review_passed": review_passed,
    "stages_started": stage_starts,
    "stage_exit_codes": stage_exits,
    "failed_stages": failed_stages,
    "fatal_markers": fatal_matches,
    "warning_count": len(warning_lines),
    "warning_lines": warning_lines,
    "completed_models": model_rows,
    "required_artifact_groups": required_status,
    "csv_checks": csv_checks,
    "artifact_count": len(artifact_names),
    "artifacts": artifact_names,
}

json_path = output_dir / "review.json"
json_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

lines = [
    "# P1–P5 Final Run Review",
    "",
    f"- Generated: `{report['created_at']}`",
    f"- Master log: `{report['master_log']}`",
    f"- Exit code: `{exit_code}`",
    f"- Overall review: `{'PASS' if review_passed else 'CHECK REQUIRED'}`",
    "",
    "## Stage status",
    "",
]

for index, stage in enumerate(stage_starts):
    rc = stage_exits[index] if index < len(stage_exits) else "MISSING"
    lines.append(f"- `{stage}`: exit `{rc}`")

lines.extend([
    "",
    "## Completed models",
    "",
    "| Model | Seed | Accuracy | Macro-F1 | Weighted-F1 | Train seconds |",
    "|---|---:|---:|---:|---:|---:|",
])

for row in model_rows:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value if value is not None else "N/A")

    lines.append(
        f"| {row.get('model')} "
        f"| {row.get('seed')} "
        f"| {fmt(row.get('accuracy'))} "
        f"| {fmt(row.get('macro_f1'))} "
        f"| {fmt(row.get('weighted_f1'))} "
        f"| {fmt(row.get('train_seconds'))} |"
    )

lines.extend([
    "",
    "## Required artifacts",
    "",
])

for name, available in required_status.items():
    lines.append(f"- [{'x' if available else ' '}] {name}")

lines.extend([
    "",
    "## Log diagnostics",
    "",
    f"- Fatal-marker lines: `{len(fatal_matches)}`",
    f"- Warning lines: `{len(warning_lines)}`",
])

if fatal_matches:
    lines.extend([
        "",
        "### Fatal markers",
        "",
        "```text",
        *fatal_matches[-100:],
        "```",
    ])

if warning_lines:
    lines.extend([
        "",
        "### Warnings",
        "",
        "```text",
        *warning_lines[-100:],
        "```",
    ])

lines.extend([
    "",
    "## CSV integrity",
    "",
    "| File | Rows | Columns | Empty cells | Status |",
    "|---|---:|---:|---:|---|",
])

for row in csv_checks:
    lines.append(
        f"| `{row['path']}` "
        f"| {row['rows']} "
        f"| {row['columns']} "
        f"| {row['empty_cells']} "
        f"| {row['status']} |"
    )

markdown_path = output_dir / "REVIEW.md"
markdown_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print(markdown_path)
print()
print(markdown_path.read_text(encoding="utf-8"))
