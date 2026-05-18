from pathlib import Path
import json
import time

import pandas as pd
import torch


class Timer:
    def __init__(self, name="run", samples=0):
        self.name = name
        self.samples = samples
        self.start = None
        self.end = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        return False

    @property
    def total_seconds(self):
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    def summary(self):
        total = float(self.total_seconds or 0.0)
        samples = int(self.samples or 0)
        data = {
            "name": self.name,
            "total_seconds": total,
            "samples": samples,
            "seconds_per_sample": float(total / samples) if samples else None,
            "samples_per_second": float(samples / total) if total > 0 and samples else None,
            "cuda_max_memory_allocated": None,
            "cuda_max_memory_reserved": None,
        }
        if torch.cuda.is_available():
            data["cuda_max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
            data["cuda_max_memory_reserved"] = int(torch.cuda.max_memory_reserved())
        return data


def save_latency_summary(rows, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out / "latency_summary.csv"
    json_path = out / "latency_summary.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    for p in [csv_path, json_path]:
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {p}")
    return csv_path
