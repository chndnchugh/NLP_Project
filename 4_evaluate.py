"""
Stage 4: Test Set Evaluation
- Loads fine-tuned model
- Evaluates on held-out test set
- Compares against baseline
- Saves confusion matrix and full report
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
import torch

# ── Config ────────────────────────────────────────────────────────────────────
FINETUNED_DIR = "models/finbert_finetuned"
DATASET_DIR   = "data/tokenized_dataset"
RESULTS_DIR   = "results"
BATCH_SIZE    = 32
DEVICE        = 0 if torch.cuda.is_available() else -1

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model_dir: str, dataset: DatasetDict):
    import pandas as pd

    # Load tokenizer from HuggingFace directly — avoids broken local tokenizer files
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)

    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        batch_size=BATCH_SIZE,
        truncation=True,
    )

    df         = pd.read_csv("data/financial_sentiment.csv")
    df         = df.dropna(subset=["Headline", "Sentiment"])
    df["label_id"] = df["Sentiment"].str.lower().str.strip().map(LABEL2ID)
    df         = df[df["label_id"].notna()]
    test_texts = [str(t) for t in df["Headline"].tolist()[-len(dataset["test"]):]]

    print(f"Running inference on {len(test_texts)} test samples ...")
    preds_raw  = clf(test_texts)
    return np.array([LABEL2ID[p["label"]] for p in preds_raw])

# ── Report ────────────────────────────────────────────────────────────────────
def full_report(true_labels, pred_labels, tag: str):
    labels = list(ID2LABEL.values())

    print(f"\n{'='*52}")
    print(f"  {tag}")
    print('='*52)
    report_str = classification_report(true_labels, pred_labels, target_names=labels)
    print(report_str)

    with open(f"{RESULTS_DIR}/{tag}_report.txt", "w") as f:
        f.write(report_str)

    cm   = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title(tag)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{tag}_confusion.png", dpi=150)
    plt.close()

    return {
        "accuracy":    round(accuracy_score(true_labels, pred_labels), 4),
        "f1_weighted": round(f1_score(true_labels, pred_labels, average="weighted"), 4),
        "f1_macro":    round(f1_score(true_labels, pred_labels, average="macro"), 4),
    }

# ── Compare ───────────────────────────────────────────────────────────────────
def compare(baseline_metrics: dict, finetuned_metrics: dict):
    print(f"\n{'='*52}")
    print("  Baseline vs Fine-tuned comparison")
    print(f"{'='*52}")
    print(f"{'Metric':<20} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>8}")
    print("-"*52)
    for k in baseline_metrics:
        b     = baseline_metrics[k]
        f     = finetuned_metrics[k]
        delta = f - b
        print(f"{k:<20} {b:>10.4f} {f:>12.4f} {delta:>+8.4f}")

    with open(f"{RESULTS_DIR}/comparison.json", "w") as fh:
        json.dump({"baseline": baseline_metrics, "finetuned": finetuned_metrics}, fh, indent=2)
    print(f"\nComparison saved to {RESULTS_DIR}/comparison.json")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    dataset     = DatasetDict.load_from_disk(DATASET_DIR)
    true_labels = np.load(f"{RESULTS_DIR}/true_labels.npy")

    ft_preds   = predict(FINETUNED_DIR, dataset)
    ft_metrics = full_report(true_labels, ft_preds, tag="finetuned")
    np.save(f"{RESULTS_DIR}/finetuned_preds.npy", ft_preds)

    bl_preds   = np.load(f"{RESULTS_DIR}/baseline_preds.npy")
    bl_metrics = full_report(true_labels, bl_preds, tag="baseline")

    compare(bl_metrics, ft_metrics)
