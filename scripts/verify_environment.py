import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import sys


LABELS = ["no", "intrinsic", "extrinsic"]
REQUIRED_IMPORTS = [
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("datasets", "datasets"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("trl", "trl"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("tqdm", "tqdm"),
    ("huggingface_hub", "huggingface_hub"),
    ("dotenv", "python-dotenv"),
    ("yaml", "pyyaml"),
]
OPTIONAL_IMPORTS = [
    ("bitsandbytes", "bitsandbytes"),
]


def package_version(package, module):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return getattr(module, "__version__", "unknown")


def check_imports(require_bitsandbytes=False):
    modules = {}
    versions = {}
    errors = []
    warnings = []
    for module_name, package_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            modules[module_name] = module
            versions[package_name] = package_version(package_name, module)
        except Exception as exc:
            errors.append(f"Missing or broken Python dependency: import {module_name} from package {package_name} failed with {type(exc).__name__}: {exc}")
    for module_name, package_name in OPTIONAL_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            modules[module_name] = module
            versions[package_name] = package_version(package_name, module)
        except Exception as exc:
            message = f"Optional dependency import {module_name} from package {package_name} failed with {type(exc).__name__}: {exc}"
            if require_bitsandbytes:
                errors.append(message)
            else:
                warnings.append(message)
    if errors:
        raise RuntimeError("\n".join(errors))
    return modules, versions, warnings


def load_manifest(path):
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")
    import yaml

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{manifest_path} must contain a YAML mapping")
    required = ["seed", "labels", "models", "datasets", "generation", "runtime", "lora"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{manifest_path} missing required top-level keys: {missing}")
    if data["seed"] != 42:
        raise ValueError(f"{manifest_path} seed must be 42, found {data['seed']}")
    if data["labels"] != LABELS:
        raise ValueError(f"{manifest_path} labels must be exactly {LABELS}, found {data['labels']}")
    if not isinstance(data["models"], dict) or not data["models"]:
        raise ValueError(f"{manifest_path} models must be a non-empty mapping")
    for name, item in data["models"].items():
        for key in ["hf_id", "local_dir"]:
            if key not in item or not str(item[key]).strip():
                raise ValueError(f"{manifest_path} models.{name}.{key} is required")
    for split, item in data["datasets"].items():
        paths = item.get("fallback_paths") if isinstance(item, dict) else None
        if not paths:
            raise ValueError(f"{manifest_path} datasets.{split}.fallback_paths is required")
    generation = data["generation"]
    for key in ["do_sample", "temperature", "top_p", "max_new_tokens"]:
        if key not in generation:
            raise ValueError(f"{manifest_path} generation.{key} is required")
    runtime = data["runtime"]
    for key in ["torch_dtype", "target_gpu", "quantization"]:
        if key not in runtime:
            raise ValueError(f"{manifest_path} runtime.{key} is required")
    lora = data["lora"]
    for key in ["adapter_search_paths", "adapter_markers", "rank_source"]:
        if key not in lora:
            raise ValueError(f"{manifest_path} lora.{key} is required")
    return data


def resolve_dataset(manifest, dataset_check):
    if dataset_check == "none":
        return {}
    splits = list(manifest["datasets"]) if dataset_check == "all" else [dataset_check]
    found = {}
    for split in splits:
        if split not in manifest["datasets"]:
            raise ValueError(f"Dataset split {split} is not present in manifest")
        paths = manifest["datasets"][split]["fallback_paths"]
        match = None
        for item in paths:
            candidate = Path(item)
            if candidate.exists():
                match = str(candidate)
                break
        if match is None:
            raise FileNotFoundError(f"No dataset path exists for split {split}. Checked: {paths}")
        found[split] = match
    return found


def find_token_source():
    if os.getenv("HF_TOKEN"):
        return "environment"
    env_path = Path(".env")
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.split("=", 1)[0].strip() == "HF_TOKEN":
            return ".env"
    return None


def check_cuda(torch_module, require_cuda=False, require_bf16=False):
    available = bool(torch_module.cuda.is_available())
    data = {
        "cuda_available": available,
        "device_count": 0,
        "device_name": None,
        "bf16_supported": False,
    }
    if available:
        data["device_count"] = int(torch_module.cuda.device_count())
        data["device_name"] = torch_module.cuda.get_device_name(0)
        data["bf16_supported"] = bool(torch_module.cuda.is_bf16_supported())
    if require_cuda and not available:
        raise RuntimeError("CUDA is required for this target check, but torch.cuda.is_available() is false")
    if require_bf16 and not data["bf16_supported"]:
        raise RuntimeError("bf16 support is required for this target check, but torch.cuda.is_bf16_supported() is false")
    return data


def active_environment():
    return os.getenv("VIRTUAL_ENV") or os.getenv("CONDA_PREFIX")


def write_summary(path, summary):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty output: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/experiment_manifest.yaml")
    parser.add_argument("--target-check", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-bf16", action="store_true")
    parser.add_argument("--require-active-env", action="store_true")
    parser.add_argument("--require-hf-token", action="store_true")
    parser.add_argument("--require-bitsandbytes", action="store_true")
    parser.add_argument("--dataset-check", choices=["none", "train", "test", "private_test", "all"], default="test")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    modules, versions, warnings = check_imports(require_bitsandbytes=args.require_bitsandbytes)
    manifest = load_manifest(args.manifest)
    datasets = resolve_dataset(manifest, args.dataset_check)
    token_source = find_token_source()
    if args.require_hf_token and token_source is None:
        raise RuntimeError("HF_TOKEN is required in .env or the process environment")
    env_path = active_environment()
    if args.require_active_env and not env_path:
        raise RuntimeError("No active Python environment detected. Activate uv venv or conda before target execution.")
    cuda = check_cuda(modules["torch"], require_cuda=args.require_cuda, require_bf16=args.require_bf16)
    summary = {
        "mode": "target-check" if args.target_check else "local-check",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "active_environment": bool(env_path),
        "token_key_present": token_source is not None,
        "manifest": args.manifest,
        "seed": manifest["seed"],
        "labels": manifest["labels"],
        "datasets": datasets,
        "runtime": manifest["runtime"],
        "generation": manifest["generation"],
        "lora": manifest["lora"],
        "cuda": cuda,
        "package_versions": versions,
        "warnings": warnings,
    }
    if args.summary_json:
        write_summary(args.summary_json, summary)
    print("ENVIRONMENT_OK")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
