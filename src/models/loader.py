from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer, BitsAndBytesConfig


TOKENIZER_REQUIRED_FILES = ["tokenizer_config.json"]
TOKENIZER_ASSET_FILES = ["tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model", "vocab.txt", "vocab.json"]
PROCESSOR_ASSET_FILES = ["processor_config.json", "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model", "vocab.txt", "vocab.json"]
MODEL_REQUIRED_FILES = ["config.json"]
ADAPTER_REQUIRED_FILES = ["adapter_config.json"]
ADAPTER_WEIGHT_FILES = ["adapter_model.safetensors", "adapter_model.bin"]
MODEL_WEIGHT_FILES = ["pytorch_model.bin", "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin.index.json"]


def assert_model_dir(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")
    if not p.is_dir():
        raise FileNotFoundError(f"Model path is not a directory: {path}")
    return p


def require_file(parent, name, label):
    p = Path(parent) / name
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def require_any(parent, names, label):
    checked = []
    for name in names:
        p = Path(parent) / name
        checked.append(str(p))
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(f"{label} not found. Checked: {', '.join(checked)}")


def validate_model_files(path):
    p = assert_model_dir(path)
    for name in MODEL_REQUIRED_FILES:
        require_file(p, name, "Model config")
    for name in TOKENIZER_REQUIRED_FILES:
        require_file(p, name, "Tokenizer config")
    tokenizer_asset = require_any(p, TOKENIZER_ASSET_FILES, "Tokenizer asset")
    return {"model_dir": str(p), "tokenizer_asset": str(tokenizer_asset)}


def validate_text_model_files(path, require_tokenizer_config=True):
    p = assert_model_dir(path)
    for name in MODEL_REQUIRED_FILES:
        require_file(p, name, "Model config")
    if require_tokenizer_config:
        require_file(p, "tokenizer_config.json", "Tokenizer config")
    asset = require_any(p, PROCESSOR_ASSET_FILES, "Tokenizer or processor asset")
    weight = require_any(p, MODEL_WEIGHT_FILES, "Model weights")
    return {"model_dir": str(p), "asset": str(asset), "model_weight": str(weight)}


def validate_adapter_files(path):
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {p}")
    require_file(p, ADAPTER_REQUIRED_FILES[0], "Adapter config")
    adapter_weight = require_any(p, ADAPTER_WEIGHT_FILES, "Adapter weights")
    return {"adapter_dir": str(p), "adapter_weight": str(adapter_weight)}


def nf4_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_pretrained_with_dtype_fallback(model_class, path, kwargs):
    try:
        return model_class.from_pretrained(path, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        fallback = dict(kwargs)
        fallback["torch_dtype"] = fallback.pop("dtype")
        return model_class.from_pretrained(path, **fallback)


def load_causal_lm(local_dir, quantized=True, adapter_dir=None, adapter_required=False):
    info = validate_model_files(local_dir)
    path = info["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    kwargs = {"device_map": "auto", "dtype": torch.bfloat16, "trust_remote_code": True}
    if quantized:
        kwargs["quantization_config"] = nf4_config()
    model = load_pretrained_with_dtype_fallback(AutoModelForCausalLM, path, kwargs)
    if adapter_dir is not None:
        validate_adapter_files(adapter_dir)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    elif adapter_required:
        raise RuntimeError("adapter_dir is required for adapter-based generation")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def load_image_text_model_text_only(local_dir):
    path = str(assert_model_dir(local_dir))
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = load_pretrained_with_dtype_fallback(AutoModelForImageTextToText, path, {"device_map": "auto", "dtype": torch.bfloat16, "trust_remote_code": True})
    model.eval()
    return model, processor


def load_encoder_classifier(local_dir, num_labels=3):
    path = str(assert_model_dir(local_dir))
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer


def tokenizer_from_processor_or_tokenizer(processor_or_tokenizer):
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    return processor_or_tokenizer


def load_text_label_scoring_model(local_dir, expected_loader=None, dtype=torch.bfloat16, quantized=False):
    info = validate_text_model_files(local_dir, require_tokenizer_config=False)
    path = info["model_dir"]
    kwargs = {"device_map": "auto", "dtype": dtype, "trust_remote_code": True}
    if quantized:
        kwargs["quantization_config"] = nf4_config()
    if expected_loader == "AutoModelForImageTextToText":
        processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        model = load_pretrained_with_dtype_fallback(AutoModelForImageTextToText, path, kwargs)
        tokenizer = tokenizer_from_processor_or_tokenizer(processor)
    elif expected_loader == "AutoModelForCausalLM":
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = load_pretrained_with_dtype_fallback(AutoModelForCausalLM, path, kwargs)
    else:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        architectures = [str(item) for item in getattr(config, "architectures", []) or []]
        if any("ImageTextToText" in item or "ConditionalGeneration" in item for item in architectures):
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            model = load_pretrained_with_dtype_fallback(AutoModelForImageTextToText, path, kwargs)
            tokenizer = tokenizer_from_processor_or_tokenizer(processor)
        else:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = load_pretrained_with_dtype_fallback(AutoModelForCausalLM, path, kwargs)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer
