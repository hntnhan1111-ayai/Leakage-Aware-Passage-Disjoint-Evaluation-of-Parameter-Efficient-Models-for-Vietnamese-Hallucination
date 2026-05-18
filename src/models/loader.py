from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForSequenceClassification, AutoProcessor, AutoTokenizer, BitsAndBytesConfig


def assert_model_dir(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")
    return str(p)


def nf4_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_causal_lm(local_dir, quantized=True):
    path = assert_model_dir(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if quantized:
        kwargs["quantization_config"] = nf4_config()
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_image_text_model_text_only(local_dir):
    path = assert_model_dir(local_dir)
    processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    return model, processor


def load_encoder_classifier(local_dir, num_labels=3):
    path = assert_model_dir(local_dir)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer
