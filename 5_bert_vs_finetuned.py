"""
Evaluation: Fine-tuned BERT-base vs Fine-tuned V1 & V2
=======================================================
All three models are fine-tuned on the same training data.
The only difference is the starting weights:
  - BERT-base : bert-base-uncased  (generic English pre-training)
  - V1        : ProsusAI/finbert   (finance-domain pre-training)
  - V2        : ProsusAI/finbert   (finance-domain, different hyperparams)

Test set: Financial PhraseBank "sentences_allagree" (~2200 human-annotated)
  — independent of all three models and the training data.

This comparison isolates one variable: does starting from a finance-domain
pre-trained model (FinBERT) give a measurable advantage over starting from
generic BERT, when everything else is held equal?

Outputs (saved to results/bert_vs_finetuned/):
  metrics_comparison.png       — Accuracy, F1-weighted, F1-macro
  per_class_f1.png             — per-class F1 breakdown
  confusion_matrices.png       — all 3 confusion matrices
  improvement_over_bert.png    — delta of V1 and V2 over fine-tuned BERT-base
  bert_vs_finetuned.json       — machine-readable metrics
  <model>_report.txt           — full classification report
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
BERT_BASE_V1_DIR = "models/bert_base_finetuned"    # V1: balanced CE loss
BERT_BASE_V2_DIR = "models/bert_base_finetuned_v2" # V2: focal loss + stronger neg weight
V1_DIR           = "models/finbert_finetuned"
V2_DIR           = "models/finbert_finetuned_v2"
RESULTS_DIR      = "results/bert_vs_finetuned"
BATCH_SIZE       = 32
DEVICE           = 0 if torch.cuda.is_available() else -1

ID2LABEL    = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID    = {v: k for k, v in ID2LABEL.items()}
LABEL_NAMES = list(ID2LABEL.values())

COLORS = {
    "BERT-base V1":  "#adb5bd",
    "BERT-base V2":  "#4dabf7",
    "V1 — FinBERT":  "#0d6efd",
    "V2 — FinBERT":  "#198754",
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
def predict(model_dir: str, tag: str, texts: list) -> np.ndarray:
    print(f"\n[{tag}] Loading {model_dir} ...")
    # Each model ships with its own tokenizer — load from the saved directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        truncation=True,
    )
    print(f"  Running inference on {len(texts)} samples ...")
    raw   = clf(texts)
    preds = np.array([LABEL2ID[p["label"]] for p in raw])
    print("  Done.")
    return preds

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(true: np.ndarray, preds: np.ndarray, tag: str) -> dict:
    report_str = classification_report(true, preds, target_names=LABEL_NAMES, zero_division=0)
    print(f"\n{'='*56}\n  {tag}\n{'='*56}")
    print(report_str)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe_tag = tag.replace(" ", "_").replace("—", "-")
    with open(f"{RESULTS_DIR}/{safe_tag}_report.txt", "w") as f:
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

# ── Plot helpers ───────────────────────────────────────────────────────────────
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
    ax.set_title("Fine-tuned BERT-base vs V1 & V2 (FinBERT)\nTest: Financial PhraseBank",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
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
    ax.set_title("Per-Class F1: Fine-tuned BERT-base vs V1 & V2",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/per_class_f1.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/per_class_f1.png")

def plot_confusion_matrices(metrics: dict):
    models = list(metrics.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    fig.suptitle("Confusion Matrices — BERT-base V1 & V2 vs FinBERT V1 & V2\n(test: Financial PhraseBank)",
                 fontsize=12, fontweight="bold", y=1.01)
    for ax, m in zip(axes, models):
        cm = np.array(metrics[m]["confusion_matrix"])
        ConfusionMatrixDisplay(cm, display_labels=[c.capitalize() for c in LABEL_NAMES]
                               ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{m}\nacc={metrics[m]['accuracy']:.3f}  f1={metrics[m]['f1_weighted']:.3f}",
                     fontsize=9, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("True", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/confusion_matrices.png")

def plot_improvement(metrics: dict):
    """Delta of all models vs BERT-base V1 (the weakest fine-tuned baseline)."""
    bert_m = metrics["BERT-base V1"]
    keys   = ["accuracy", "f1_weighted", "f1_macro"]
    xlbls  = ["Accuracy", "F1 Weighted", "F1 Macro"]
    compare = ["BERT-base V2", "V1 — FinBERT", "V2 — FinBERT"]
    x, w   = np.arange(len(keys)), 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, m in enumerate(compare):
        deltas = [metrics[m][k] - bert_m[k] for k in keys]
        bars   = ax.bar(x + i * w, deltas, w, label=m, color=COLORS[m], alpha=0.88, edgecolor="white")
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.003 if d >= 0 else -0.013),
                    f"{d:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(x + w); ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylabel("Δ Score vs BERT-base V1", fontsize=11); _style(ax)
    ax.set_title("Improvement Over BERT-base V1\n"
                 "Shows gain from focal loss (V2) and domain pre-training (FinBERT)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/improvement_over_bert.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/improvement_over_bert.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    texts, true_labels = load_phrasebank()

    runs = [
        ("BERT-base V1",  BERT_BASE_V1_DIR),
        ("BERT-base V2",  BERT_BASE_V2_DIR),
        ("V1 — FinBERT",  V1_DIR),
        ("V2 — FinBERT",  V2_DIR),
    ]

    all_metrics = {}
    for tag, model_dir in runs:
        preds            = predict(model_dir, tag, texts)
        all_metrics[tag] = compute_metrics(true_labels, preds, tag)
        safe_tag         = tag.replace(" ", "_").replace("—", "-")
        np.save(f"{RESULTS_DIR}/{safe_tag}_preds.npy", preds)

    # Save JSON summary
    summary = {k: {m: v for m, v in vals.items() if m != "confusion_matrix"}
               for k, vals in all_metrics.items()}
    with open(f"{RESULTS_DIR}/bert_vs_finetuned.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Terminal table
    col_w = 14
    print(f"\n{'='*80}")
    print(f"  {'Metric':<20}" + "".join(f"{t:>{col_w}}" for t in all_metrics))
    print(f"{'='*80}")
    for m in ["accuracy", "f1_weighted", "f1_macro"]:
        row = f"  {m:<20}"
        for tag in all_metrics:
            row += f" {all_metrics[tag][m]:>{col_w}.4f}"
        print(row)
    print(f"{'='*66}")

    print("\nGenerating charts ...")
    plot_metrics(all_metrics)
    plot_per_class_f1(all_metrics)
    plot_confusion_matrices(all_metrics)
    plot_improvement(all_metrics)

    print(f"\n✓ Done. Results saved to {RESULTS_DIR}/")
