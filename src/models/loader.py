from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer, BitsAndBytesConfig


TOKENIZER_REQUIRED_FILES = ["tokenizer_config.json"]
TOKENIZER_ASSET_FILES = ["tokenizer.json", "tokenizer.model", "sentencepiece.bpe.model", "vocab.txt", "vocab.json"]
MODEL_REQUIRED_FILES = ["config.json"]
ADAPTER_REQUIRED_FILES = ["adapter_config.json"]
ADAPTER_WEIGHT_FILES = ["adapter_model.safetensors", "adapter_model.bin"]


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


def load_causal_lm(local_dir, quantized=True, adapter_dir=None, adapter_required=False):
    info = validate_model_files(local_dir)
    path = info["model_dir"]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if quantized:
        kwargs["quantization_config"] = nf4_config()
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
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
    model = AutoModelForImageTextToText.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    return model, processor


def load_encoder_classifier(local_dir, num_labels=3):
    path = str(assert_model_dir(local_dir))
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer
