"""
3-Way Evaluation: FinBERT Zero-Shot vs Fine-tuned V1 & V2
==========================================================
Baseline : ProsusAI/finbert  (zero-shot — no fine-tuning on your data)
  - Pre-trained on financial text, so it already understands domain vocabulary
  - Represents the ceiling of what you get from FinBERT without any task training
  - A high baseline here means the pre-training alone is strong; the gap to
    V1/V2 shows how much your labelled data and fine-tuning added on top

V1 / V2  : ProsusAI/finbert fine-tuned on LM+VADER-labelled financial headlines

Test set : Financial PhraseBank "sentences_allagree" (~2200 human-annotated)
  — independent of all three models and the training data

Results saved to: results/finbert_vs_finetuned/
"""

import os
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

# ── Config ─────────────────────────────────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"   # zero-shot baseline
V1_DIR        = "models/finbert_finetuned"
V2_DIR        = "models/finbert_finetuned_v2"
RESULTS_DIR   = "results/finbert_vs_finetuned"
BATCH_SIZE    = 32
DEVICE        = 0 if torch.cuda.is_available() else -1

# Native FinBERT label mapping — must match fine-tuned models too
ID2LABEL    = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID    = {v: k for k, v in ID2LABEL.items()}
LABEL_NAMES = list(ID2LABEL.values())

COLORS = {
    "FinBERT (zero-shot)": "#6c757d",
    "V1":                  "#0d6efd",
    "V2":                  "#198754",
}

# ── Financial PhraseBank ───────────────────────────────────────────────────────
def load_phrasebank() -> tuple:
    print("Loading Financial PhraseBank (sentences_allagree) ...")
    zip_path = hf_hub_download(
        repo_id="takala/financial_phrasebank",
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
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
                    labels.append(LABEL2ID[label])

    labels = np.array(labels)
    print(f"  {len(texts)} samples  |  "
          f"pos={sum(labels==0)}  neg={sum(labels==1)}  neu={sum(labels==2)}")
    return texts, labels

# ── Inference ──────────────────────────────────────────────────────────────────
def predict(model_id: str, tag: str, texts: list) -> np.ndarray:
    print(f"\n[{tag}] Loading: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(model_id)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        truncation=True,
    )
    print(f"[{tag}] Running inference on {len(texts)} samples ...")
    raw   = clf(texts)
    preds = np.array([LABEL2ID[p["label"]] for p in raw])
    print(f"[{tag}] Done.")
    return preds

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(true: np.ndarray, preds: np.ndarray, tag: str) -> dict:
    report_str = classification_report(true, preds, target_names=LABEL_NAMES, zero_division=0)
    print(f"\n{'='*56}\n  {tag}\n{'='*56}")
    print(report_str)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = tag.replace(" ", "_").replace("(", "").replace(")", "")
    with open(f"{RESULTS_DIR}/{safe}_report.txt", "w") as f:
        f.write(f"Model: {tag}\n{'='*56}\n{report_str}")

    _, _, f1_per, _ = precision_recall_fscore_support(
        true, preds, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "accuracy":     round(accuracy_score(true, preds), 4),
        "f1_weighted":  round(f1_score(true, preds, average="weighted", zero_division=0), 4),
        "f1_macro":     round(f1_score(true, preds, average="macro",    zero_division=0), 4),
        "f1_per_class": {LABEL_NAMES[i]: round(float(f1_per[i]), 4) for i in range(3)},
        "confusion_matrix": confusion_matrix(true, preds).tolist(),
    }

# ── Plots ──────────────────────────────────────────────────────────────────────
def _annotate(ax, bars):
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

def _style(ax):
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

def plot_metrics(metrics: dict):
    keys  = ["accuracy", "f1_weighted", "f1_macro"]
    xlbls = ["Accuracy", "F1 Weighted", "F1 Macro"]
    models = list(metrics.keys())
    x, w = np.arange(len(keys)), 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        bars = ax.bar(x + i * w, [metrics[m][k] for k in keys], w,
                      label=m, color=COLORS[m], alpha=0.88, edgecolor="white")
        _annotate(ax, bars)

    ax.set_xticks(x + w); ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score", fontsize=11); _style(ax)
    ax.set_title("FinBERT Zero-Shot vs Fine-tuned V1 & V2 — Overall Metrics\n"
                 "Test: Financial PhraseBank (human-annotated)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/metrics_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/metrics_comparison.png")

def plot_per_class_f1(metrics: dict):
    models = list(metrics.keys())
    x, w = np.arange(len(LABEL_NAMES)), 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        bars = ax.bar(x + i * w, [metrics[m]["f1_per_class"][c] for c in LABEL_NAMES], w,
                      label=m, color=COLORS[m], alpha=0.88, edgecolor="white")
        _annotate(ax, bars)

    ax.set_xticks(x + w); ax.set_xticklabels([c.capitalize() for c in LABEL_NAMES], fontsize=11)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score", fontsize=11); _style(ax)
    ax.set_title("Per-Class F1: FinBERT Zero-Shot vs Fine-tuned V1 & V2",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/per_class_f1.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/per_class_f1.png")

def plot_confusion_matrices(metrics: dict):
    models = list(metrics.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Confusion Matrices: FinBERT Zero-Shot vs Fine-tuned V1 & V2\n"
                 "(test: Financial PhraseBank)",
                 fontsize=12, fontweight="bold", y=1.02)
    for ax, m in zip(axes, models):
        cm = np.array(metrics[m]["confusion_matrix"])
        ConfusionMatrixDisplay(cm, display_labels=[c.capitalize() for c in LABEL_NAMES]
                               ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{m}\nacc={metrics[m]['accuracy']:.3f}",
                     fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/confusion_matrices.png")

def plot_improvement(metrics: dict):
    baseline_m = metrics["FinBERT (zero-shot)"]
    keys   = ["accuracy", "f1_weighted", "f1_macro"]
    xlbls  = ["Accuracy", "F1 Weighted", "F1 Macro"]
    models = ["V1", "V2"]
    x, w   = np.arange(len(keys)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(models):
        deltas = [metrics[m][k] - baseline_m[k] for k in keys]
        bars   = ax.bar(x + i * w, deltas, w, label=m,
                        color=COLORS[m], alpha=0.88, edgecolor="white")
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.003 if d >= 0 else -0.013),
                    f"{d:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(x + w / 2); ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylabel("Δ Score vs FinBERT Zero-Shot", fontsize=11); _style(ax)
    ax.set_title("Fine-tuning Gain over FinBERT Zero-Shot\n"
                 "(positive = fine-tuning helped beyond zero-shot FinBERT)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/improvement_over_finbert.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/improvement_over_finbert.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    texts, true_labels = load_phrasebank()

    runs = [
        ("FinBERT (zero-shot)", FINBERT_MODEL),
        ("V1",                  V1_DIR),
        ("V2",                  V2_DIR),
    ]

    all_metrics = {}
    for tag, model_id in runs:
        preds            = predict(model_id, tag, texts)
        all_metrics[tag] = compute_metrics(true_labels, preds, tag)
        safe = tag.replace(" ", "_").replace("(", "").replace(")", "")
        np.save(f"{RESULTS_DIR}/{safe}_preds.npy", preds)

    # JSON summary
    summary = {k: {m: v for m, v in vals.items() if m != "confusion_matrix"}
               for k, vals in all_metrics.items()}
    with open(f"{RESULTS_DIR}/comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR}/comparison.json")

    # Terminal table
    print(f"\n{'='*66}")
    print(f"  {'Metric':<20} {'FinBERT ZS':>14} {'V1':>10} {'V2':>10}")
    print(f"{'='*66}")
    for m in ["accuracy", "f1_weighted", "f1_macro"]:
        row = f"  {m:<20}"
        for tag in ["FinBERT (zero-shot)", "V1", "V2"]:
            row += f" {all_metrics[tag][m]:>10.4f}"
        print(row)
    print(f"{'='*66}")

    print("\nGenerating charts ...")
    plot_metrics(all_metrics)
    plot_per_class_f1(all_metrics)
    plot_confusion_matrices(all_metrics)
    plot_improvement(all_metrics)

    print(f"\n✓ Done. Results saved to {RESULTS_DIR}/")
