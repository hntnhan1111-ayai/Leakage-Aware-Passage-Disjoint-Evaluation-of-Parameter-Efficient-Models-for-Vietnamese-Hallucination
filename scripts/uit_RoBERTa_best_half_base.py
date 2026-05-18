import os

import torch

import pandas as pd

import numpy as np

import random

import warnings

from datasets import Dataset

from transformers import (

    AutoModelForSequenceClassification,

    AutoTokenizer,

    TrainingArguments,

    Trainer,

    logging as transformers_logging

)

from sklearn.metrics import f1_score

from tqdm import tqdm



transformers_logging.set_verbosity_info()

warnings.filterwarnings('ignore')

torch.cuda.empty_cache()



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



MODEL_NAME = "models/phobert-base-v2"

TRAIN_PATH = "vihallu-train.csv"

TEST_PATH = "vihallu-test.csv"

OUTPUT_DIR = "roberta_baseline_results"



LABEL_MAP = {'no': 0, 'intrinsic': 1, 'extrinsic': 2}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}



def load_csv_robust(file_path):

    encodings = ['utf-8', 'utf-8-sig', 'latin-1']

    for encoding in encodings:

        try:

            return pd.read_csv(file_path, encoding=encoding)

        except:

            continue

    raise Exception(f"Could not load {file_path}")



def preprocess_data(df, tokenizer):

    texts = []

    labels = []

    for _, row in df.iterrows():

        text = f"{str(row['context'])} </s></s> {str(row['response'])}"

        texts.append(text)

        if 'label' in row:

            labels.append(LABEL_MAP[row['label']])

    

    encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=256)

    if labels:

        encodings['labels'] = labels

    return Dataset.from_dict(encodings)



def train_baseline_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df = load_csv_robust(TRAIN_PATH).dropna().reset_index(drop=True)

    train_dataset = preprocess_data(train_df, tokenizer)



    model = AutoModelForSequenceClassification.from_pretrained(

        MODEL_NAME, 

        num_labels=3

    )



    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        learning_rate=2e-5,

        per_device_train_batch_size=16,

        num_train_epochs=3,

        weight_decay=0.01,

        logging_steps=10,

        eval_strategy="no",

        save_strategy="epoch",

        load_best_model_at_end=False,

        report_to="none"

    )



    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

    )



    trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/final_model")

    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

    return f"{OUTPUT_DIR}/final_model"



def run_baseline_inference(model_path):

    transformers_logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to("cuda")

    model.eval()



    test_df = load_csv_robust(TEST_PATH).dropna(subset=['context', 'response', 'label'])



    preds = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="RoBERTa Baseline Inference"):

        inputs = tokenizer(

            f"{str(row['context'])} </s></s> {str(row['response'])}", 

            return_tensors="pt", 

            truncation=True, 

            padding='max_length', 

            max_length=256

        ).to("cuda")

        

        with torch.no_grad():

            outputs = model(**inputs)

            pred_idx = torch.argmax(outputs.logits, dim=1).item()

            preds.append(INV_LABEL_MAP[pred_idx])



    true_labels = test_df['label'].tolist()

    macro_f1 = f1_score(true_labels, preds, average='macro')

    print(f"\nROBERTA BASELINE MACRO-F1: {macro_f1:.4f}")



if __name__ == "__main__":

    model_path = train_baseline_model()

    torch.cuda.empty_cache()

    run_baseline_inference(model_path)
