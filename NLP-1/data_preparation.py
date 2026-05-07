"""
Stage 1: Data Preparation
- Loads financial_sentiment.csv
- Cleans and maps labels
- Tokenizes with BERT tokenizer
- Splits into train / val / test
- Saves tokenized dataset to data/tokenized_dataset/
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME   = "bert-base-uncased"
DATA_PATH    = os.path.join(_HERE, "data", "financial_sentiment.csv")
TEXT_COL     = "text"
LABEL_COL    = "sentiment"
MAX_LENGTH   = 128
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
RANDOM_SEED  = 42

# ── Label map ─────────────────────────────────────────────────────────────────
LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}  # matches ProsusAI/finbert native mapping
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ── Load & clean ──────────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Columns found: {df.columns.tolist()}")
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[TEXT_COL]  = df[TEXT_COL].str.strip()
    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()
    df = df[df[LABEL_COL].isin(LABEL2ID)]
    df["label_id"] = df[LABEL_COL].map(LABEL2ID)
    print(f"Loaded {len(df)} samples")
    print(df[LABEL_COL].value_counts())
    return df

# ── Split ─────────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=1 - TRAIN_RATIO, stratify=df["label_id"], random_state=RANDOM_SEED
    )
    val_size = VAL_RATIO / (1 - TRAIN_RATIO)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_size, stratify=temp_df["label_id"], random_state=RANDOM_SEED
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df

# ── Tokenize ──────────────────────────────────────────────────────────────────
def build_dataset(train_df, val_df, test_df) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_hf(df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict({
            "text":   df[TEXT_COL].tolist(),
            "labels": df["label_id"].tolist(),
        })

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    raw = DatasetDict({
        "train": to_hf(train_df),
        "val":   to_hf(val_df),
        "test":  to_hf(test_df),
    })

    tokenized = raw.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
    return tokenized

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.join(_HERE, "data", "tokenized_dataset")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)

    df = load_and_clean(DATA_PATH)
    train_df, val_df, test_df = split_data(df)
    dataset = build_dataset(train_df, val_df, test_df)
    dataset.save_to_disk(out_dir)
    print(f"Dataset saved to {out_dir}/")