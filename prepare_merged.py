"""
prepare_merged.py  —  Data Preparation for Merged Dataset + PhraseBank
=======================================================================
Loads data/merged_relabeled.csv, augments training with 80% of the
Financial PhraseBank (AllAgree) split, and holds out the remaining 20%
as the test set for all three models.

Why PhraseBank in training?
  Generic BERT needs direct exposure to the financial sentence style used
  in the benchmark. Adding PhraseBank's train split closes the distribution
  gap and is the standard approach for finance NLP benchmarks.

Split strategy:
  1. merged_relabeled.csv  →  80% train / 10% val / 10% (discarded — we
                               use PhraseBank test instead)
  2. PhraseBank AllAgree   →  80% added to train  / 20% saved as test set
                               (data/phrasebank_test.csv)

Final training set = merged 80% train  +  PhraseBank 80% train
Validation set     = merged 10% val
Test set           = PhraseBank 20% held-out  (used by compare_3way.py)

Run AFTER  : relabel_merged.py
Run BEFORE : finetune_bert_merged.py
"""

import os
import zipfile
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
_HERE              = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME         = "bert-base-uncased"
DATA_PATH          = os.path.join(_HERE, "data", "merged_relabeled.csv")
OUTPUT_DIR         = os.path.join(_HERE, "data", "tokenized_dataset_merged")
PHRASEBANK_TEST    = os.path.join(_HERE, "data", "phrasebank_test.csv")

TEXT_COL    = "text"
LABEL_COL   = "sentiment"
MAX_LENGTH  = 128           # sufficient for headlines; 512 OOMs on MPS (M4)
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
PB_TRAIN_RATIO = 0.80      # 80 % of PhraseBank goes to training
RANDOM_SEED = 42

LABEL2ID = {"positive": 0, "negative": 1, "neutral": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}


# ── Load & clean merged relabeled data ────────────────────────────────────────
def load_merged(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[TEXT_COL, LABEL_COL, "source"]].dropna(subset=[TEXT_COL, LABEL_COL])
    df[TEXT_COL]  = df[TEXT_COL].str.strip()
    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()
    df = df[df[LABEL_COL].isin(LABEL2ID)].reset_index(drop=True)
    df["label_id"] = df[LABEL_COL].map(LABEL2ID)
    print(f"Merged dataset: {len(df)} rows")
    print(df[LABEL_COL].value_counts().to_string())
    return df


# ── Download and split Financial PhraseBank ───────────────────────────────────
def load_phrasebank_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (pb_train_df, pb_test_df) — 80/20 stratified split of
    Financial PhraseBank AllAgree sentences.
    Loads from local HuggingFace cache if offline.
    """
    print("\nLoading Financial PhraseBank ...")
    try:
        zip_path = hf_hub_download(
            repo_id="takala/financial_phrasebank",
            filename="data/FinancialPhraseBank-v1.0.zip",
            repo_type="dataset",
        )
    except Exception as e:
        if "connection" in str(e).lower() or "nodename" in str(e).lower() \
                or "ConnectError" in type(e).__name__:
            print("  Network unavailable — loading from local cache ...")
            zip_path = hf_hub_download(
                repo_id="takala/financial_phrasebank",
                filename="data/FinancialPhraseBank-v1.0.zip",
                repo_type="dataset",
                local_files_only=True,
            )
        else:
            raise
    texts, labels = [], []
    with zipfile.ZipFile(zip_path) as zf:
        target = next(n for n in zf.namelist() if "AllAgree" in n and n.endswith(".txt"))
        print(f"  Parsing {target} ...")
        with zf.open(target) as f:
            for raw in f:
                line = raw.decode("latin-1").strip()
                if "@" not in line:
                    continue
                sentence, label = line.rsplit("@", 1)
                label = label.strip().lower()
                if label in LABEL2ID:
                    texts.append(sentence.strip())
                    labels.append(label)

    pb_df = pd.DataFrame({
        TEXT_COL:   texts,
        LABEL_COL:  labels,
        "source":   "phrasebank",
    })
    pb_df["label_id"] = pb_df[LABEL_COL].map(LABEL2ID)
    print(f"  PhraseBank total: {len(pb_df)} rows")
    print(f"  {pb_df[LABEL_COL].value_counts().to_dict()}")

    pb_train, pb_test = train_test_split(
        pb_df,
        test_size=1 - PB_TRAIN_RATIO,
        stratify=pb_df["label_id"],
        random_state=RANDOM_SEED,
    )
    print(f"  PhraseBank → train: {len(pb_train)}  test (held-out): {len(pb_test)}")
    return pb_train.reset_index(drop=True), pb_test.reset_index(drop=True)


# ── Stratified split of merged data ───────────────────────────────────────────
def split_merged(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df,
        test_size=1 - TRAIN_RATIO,
        stratify=df["label_id"],
        random_state=RANDOM_SEED,
    )
    val_size = VAL_RATIO / (1 - TRAIN_RATIO)
    val_df, _ = train_test_split(           # discard merged test — use PhraseBank test
        temp_df,
        test_size=1 - val_size,
        stratify=temp_df["label_id"],
        random_state=RANDOM_SEED,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# ── Tokenize ─────────────────────────────────────────────────────────────────
def build_dataset(train_df: pd.DataFrame, val_df: pd.DataFrame) -> DatasetDict:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"\nTokenizer: {MODEL_NAME}  |  max_length: {MAX_LENGTH}")

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

    raw = DatasetDict({"train": to_hf(train_df), "val": to_hf(val_df)})
    tokenized = raw.map(tokenize, batched=True, desc="Tokenizing")
    tokenized.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
    )
    return tokenized


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.join(_HERE, "data"), exist_ok=True)
    os.makedirs(os.path.join(_HERE, "results"), exist_ok=True)

    # 1. Load merged relabeled data and split
    merged_df = load_merged(DATA_PATH)
    merged_train, merged_val = split_merged(merged_df)

    # 2. Load PhraseBank and split 80/20
    pb_train, pb_test = load_phrasebank_split()

    # 3. Save held-out PhraseBank test set (used by compare_3way.py)
    pb_test.to_csv(PHRASEBANK_TEST, index=False)
    print(f"\n✓ PhraseBank test set saved → {PHRASEBANK_TEST}  ({len(pb_test)} rows)")

    # 4. Combine merged train + PhraseBank train
    train_df = pd.concat([merged_train, pb_train], ignore_index=True).sample(
        frac=1, random_state=RANDOM_SEED   # shuffle
    ).reset_index(drop=True)

    print(f"\n=== Final training set: {len(train_df)} rows ===")
    print(f"  Merged:      {len(merged_train)}")
    print(f"  PhraseBank:  {len(pb_train)}")
    print(f"  Label dist:  {train_df[LABEL_COL].value_counts().to_dict()}")
    print(f"\nValidation set: {len(merged_val)} rows")

    # 5. Tokenize and save
    dataset = build_dataset(train_df, merged_val)
    dataset.save_to_disk(OUTPUT_DIR)
    print(f"\n✓ Tokenized dataset saved → {OUTPUT_DIR}/")
