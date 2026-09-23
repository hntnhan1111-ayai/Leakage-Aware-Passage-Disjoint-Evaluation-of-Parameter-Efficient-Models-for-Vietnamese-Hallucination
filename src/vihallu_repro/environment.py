from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys


def package_versions() -> dict[str, str | None]:
    packages = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "scikit-learn",
        "pandas",
        "numpy",
        "scipy",
        "pyyaml",
        "py_vncorenlp",
    ]
    result = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def git_metadata() -> dict[str, str | None]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": run("git", "status", "--porcelain"),
    }


def environment_metadata() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "git": git_metadata(),
    }
