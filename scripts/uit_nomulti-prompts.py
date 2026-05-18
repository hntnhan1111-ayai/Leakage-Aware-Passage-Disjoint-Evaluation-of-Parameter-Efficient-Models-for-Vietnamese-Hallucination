import os
import torch
import pandas as pd
import numpy as np
import csv
import random
import re
import json
import logging
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging as transformers_logging
)
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import f1_score
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()

HF_TOKEN = os.getenv('HF_TOKEN')
if HF_TOKEN:
    os.environ['HUGGINGFACE_HUB_TOKEN'] = HF_TOKEN

RANDOM_SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(RANDOM_SEED)

MODEL_NAME = "Viet-Mistral/Vistral-7B-Chat"
POSSIBLE_TRAIN_PATHS = ["vihallu-train.csv", "data/vihallu-train.csv"]
TEST_PATH = "vihallu-test.csv"
OUTPUT_DIR = "results_scratch_train_pipeline"
ADAPTER_SAVE_PATH = f"{OUTPUT_DIR}/final_adapter"

CLASS_WEIGHTS = {'intrinsic': 0.3521, 'extrinsic': 0.3520, 'no': 0.2959}

TRAINING_CONFIG = {
    'learning_rate': 1e-4,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 16,
    'num_train_epochs': 3,
    'max_length': 1024,
    'warmup_steps': 50,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'lr_scheduler_type': "cosine",
    'optim': "paged_adamw_32bit",
    'label_smoothing_factor': 0.1
}

LORA_CONFIG = {
    'r': 128,
    'lora_alpha': 256,
    'lora_dropout': 0.05,
    'bias': "none",
    'task_type': "CAUSAL_LM",
    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

def load_csv_robust(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except:
            continue
    raise Exception(f"Could not load file {file_path}")

def compute_jaccard_similarity(text1, text2, tokenizer):
    try:
        tokens1 = set(tokenizer.tokenize(str(text1)))
        tokens2 = set(tokenizer.tokenize(str(text2)))
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union) if len(union) > 0 else 0
    except:
        return 0.5

def create_prompt(sample, tokenizer):
    context = sample['context'] if pd.notna(sample['context']) else ""
    response = sample['response'] if pd.notna(sample['response']) else ""
    jaccard_score = compute_jaccard_similarity(context, response, tokenizer)
    return f"""Phân tích hallucination bằng cách so sánh trực tiếp CONTEXT và RESPONSE.

Jaccard score: {jaccard_score:.4f} (1.0 = giống hệt)

CONTEXT:
{context}

RESPONSE:
{response}

PHÂN TÍCH:
1. RESPONSE có thông tin nào được thêm vào không có trong CONTEXT không?
2. RESPONSE có mâu thuẫn với CONTEXT không?

KẾT LUẬN (no/intrinsic/extrinsic):"""

def train_hallucination_model():
    transformers_logging.set_verbosity_info()
    train_path = next((p for p in POSSIBLE_TRAIN_PATHS if os.path.exists(p)), None)
    if not train_path: raise FileNotFoundError("Training file not found.")
    
    train_df = load_csv_robust(train_path).dropna().reset_index(drop=True)
    temp_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
    
    train_df['text'] = train_df.apply(lambda row: create_prompt(row, temp_tokenizer) + f" {row['label']}", axis=1)
    train_dataset = Dataset.from_pandas(train_df[['text']])
    del temp_tokenizer
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=TRAINING_CONFIG['max_length'],
        packing=False,
        learning_rate=TRAINING_CONFIG['learning_rate'],
        num_train_epochs=TRAINING_CONFIG['num_train_epochs'],
        per_device_train_batch_size=TRAINING_CONFIG['per_device_train_batch_size'],
        gradient_accumulation_steps=TRAINING_CONFIG['gradient_accumulation_steps'],
        warmup_steps=TRAINING_CONFIG['warmup_steps'],
        weight_decay=TRAINING_CONFIG['weight_decay'],
        lr_scheduler_type=TRAINING_CONFIG['lr_scheduler_type'],
        optim=TRAINING_CONFIG['optim'],
        label_smoothing_factor=TRAINING_CONFIG['label_smoothing_factor'],
        bf16=True,
        logging_steps=10,
        report_to="none",
        save_strategy="no"
    )
    
    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_dataset, peft_config=LoraConfig(**LORA_CONFIG))
    
    
    
    trainer.train()
    trainer.model.save_pretrained(ADAPTER_SAVE_PATH)
    tokenizer.save_pretrained(ADAPTER_SAVE_PATH)
    transformers_logging.set_verbosity_error()
    return tokenizer

def predict_label(model, tokenizer, context, response):
    sample = {'context': context, 'response': response}
    prompt = create_prompt(sample, tokenizer)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, do_sample=False)
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction_part = full_output.split("KẾT LUẬN (no/intrinsic/extrinsic):")[-1].strip().lower()
    
    if "intrinsic" in prediction_part: return "intrinsic"
    if "extrinsic" in prediction_part: return "extrinsic"
    return "no"

def run_prediction_pipeline():
    transformers_logging.set_verbosity_error()
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.bfloat16)
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_SAVE_PATH).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_SAVE_PATH)
    
    

    if not os.path.exists(TEST_PATH): raise FileNotFoundError(f"{TEST_PATH} not found.")
    test_df = load_csv_robust(TEST_PATH).dropna(subset=['context', 'response', 'label'])
    
    true_labels = test_df['label'].tolist()
    preds = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference (Single Prompt)"):
        preds.append(predict_label(model, tokenizer, row['context'], row['response']))
    
    # OUTPUT FINAL MACRO-F1
    macro_f1 = f1_score(true_labels, preds, average='macro')
    print(f"\nFINAL MACRO-F1 SCORE: {macro_f1:.4f}")

if __name__ == "__main__":
    train_hallucination_model()
    torch.cuda.empty_cache()
    run_prediction_pipeline()