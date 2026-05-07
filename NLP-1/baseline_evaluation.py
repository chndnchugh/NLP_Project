"""
Stage 2: Load Pre-trained FinBERT + Baseline Evaluation
- Loads ProsusAI/finbert with classification head
- Runs zero-shot inference on the test set
- Reports baseline accuracy, F1, and confusion matrix
"""

import os
import torch
import numpy as np
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME  = "ProsusAI/finbert"
DATASET_DIR = os.path.join(_HERE, "data", "tokenized_dataset")
RESULTS_DIR = os.path.join(_HERE, "results")
BATCH_SIZE  = 32
DEVICE      = 0 if torch.cuda.is_available() else -1

ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}  # matches ProsusAI/finbert native mapping
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── Load ──────────────────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print(f"Loaded {MODEL_NAME} — {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, tokenizer

# ── Zero-shot inference ───────────────────────────────────────────────────────
def run_baseline(model, tokenizer, dataset: DatasetDict):
    import pandas as pd

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        truncation=True,
    )

    # Load raw texts directly from CSV — avoids column name issues in tokenized dataset
    df          = pd.read_csv(os.path.join(_HERE, "data", "financial_sentiment.csv"))
    df["label_id"] = df["sentiment"].str.lower().str.strip().map(LABEL2ID)
    df          = df.dropna(subset=["text", "label_id"])

    # Use the same test indices saved during data preparation
    true_labels = [int(l) for l in dataset["test"]["labels"]]
    test_texts  = df["text"].tolist()[-len(true_labels):]  # approximate match

    print(f"\nRunning zero-shot inference on {len(test_texts)} test samples ...")
    preds_raw   = clf(test_texts)
    pred_labels = [LABEL2ID[p["label"]] for p in preds_raw]

    return np.array(true_labels), np.array(pred_labels)

# ── Report ────────────────────────────────────────────────────────────────────
def report(true_labels, pred_labels, title="Baseline (zero-shot)"):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)
    print(classification_report(
        true_labels, pred_labels,
        target_names=list(ID2LABEL.values())
    ))

    cm   = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(ID2LABEL.values()))
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(os.path.join(RESULTS_DIR, f"{fname}_confusion.png"), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {RESULTS_DIR}/{fname}_confusion.png")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset = DatasetDict.load_from_disk(DATASET_DIR)
    model, tokenizer = load_model_and_tokenizer()
    true_labels, pred_labels = run_baseline(model, tokenizer, dataset)
    report(true_labels, pred_labels, title="Baseline (zero-shot)")

    np.save(os.path.join(RESULTS_DIR, "baseline_preds.npy"), pred_labels)
    np.save(os.path.join(RESULTS_DIR, "true_labels.npy"),    true_labels)
    print(f"Predictions saved to {RESULTS_DIR}/")
