"""
Final 4-Way Comparison: Complete Model Progression
===================================================
Shows the full progression from zero-shot to fine-tuned,
across both generic (BERT) and finance-domain (FinBERT) pre-training:

  ┌─────────────────────┬──────────────────────────┐
  │                     │      Pre-training         │
  │                     │  Generic   │  Finance      │
  ├─────────────────────┼────────────┼──────────────┤
  │ Zero-shot           │ BERT ZS    │ FinBERT ZS   │
  │ Fine-tuned (task)   │ BERT-base  │ FinBERT      │
  └─────────────────────┴────────────┴──────────────┘

This isolates two variables cleanly:
  - Fine-tuning effect  : zero-shot → fine-tuned (same pre-training, add task data)
  - Domain pre-training : BERT → FinBERT (same training regime, better init)

Baselines:
  BERT zero-shot   — bert-base-uncased, fill-mask scoring (no fine-tuning)
  FinBERT zero-shot — ProsusAI/finbert, text-classification (no fine-tuning)

Fine-tuned:
  BERT-base  — bert-base-uncased fine-tuned on LM+VADER-labelled headlines
  FinBERT    — ProsusAI/finbert  fine-tuned on LM+VADER-labelled headlines

Training data: financial_sentiment.csv labelled by LM+VADER (threshold=0.15)
               with class-weighted loss to correct for neutral-class imbalance
Test set     : Financial PhraseBank AllAgree (~2264 sentences, strictly held-out)

Results saved to: results/final_comparison/
"""

import os
import json
import zipfile
import numpy as np
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
BERT_ZS_MODEL    = "bert-base-uncased"           # zero-shot via fill-mask
FINBERT_ZS_MODEL = "ProsusAI/finbert"            # zero-shot via text-classification
BERT_FT_DIR      = "models/bert_base_finetuned"   # fine-tuned bert-base-uncased
FINBERT_FT_DIR   = "models/finbert_finetuned_v2"  # fine-tuned FinBERT (memory-optimised)
RESULTS_DIR      = "results/final_comparison"
BATCH_SIZE       = 32
DEVICE           = 0 if torch.cuda.is_available() else -1

# BERT fill-mask zero-shot settings
ZS_CANDIDATES = ["positive", "negative", "neutral"]
ZS_TEMPLATE   = "Overall the financial sentiment is [MASK] ."

ID2LABEL    = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID    = {v: k for k, v in ID2LABEL.items()}
LABEL_NAMES = list(ID2LABEL.values())

# Color scheme: grays = zero-shot, colors = fine-tuned; blue family = BERT, green = FinBERT
COLORS = {
    "BERT (zero-shot)":      "#adb5bd",  # light gray
    "FinBERT (zero-shot)":   "#6c757d",  # dark gray
    "BERT-base (fine-tuned)":"#74c0fc",  # light blue
    "FinBERT (fine-tuned)":  "#198754",  # green
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

# ── Inference: BERT zero-shot via fill-mask ────────────────────────────────────
def predict_bert_zero_shot(texts: list) -> np.ndarray:
    print(f"\n[BERT (zero-shot)] Loading {BERT_ZS_MODEL} ...")
    filler    = pipeline("fill-mask", model=BERT_ZS_MODEL, device=DEVICE)
    tokenizer = filler.tokenizer

    for w in ZS_CANDIDATES:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"'{w}' is not a single token in {BERT_ZS_MODEL}.")

    print(f"[BERT (zero-shot)] Scoring {len(texts)} samples ...")
    preds = []
    for text in texts:
        prompt  = f"{ZS_TEMPLATE} {text[:300]}"
        results = filler(prompt, targets=ZS_CANDIDATES)
        best    = max(results, key=lambda r: r["score"])
        preds.append(LABEL2ID[best["token_str"].lower()])

    print("[BERT (zero-shot)] Done.")
    return np.array(preds)

# ── Inference: text-classification (FinBERT ZS + both fine-tuned) ─────────────
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
    safe = tag.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
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

# ── Plot helpers ───────────────────────────────────────────────────────────────
def _annotate(ax, bars):
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.007,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

def _style(ax):
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_metrics(metrics: dict):
    keys   = ["accuracy", "f1_weighted", "f1_macro"]
    xlbls  = ["Accuracy", "F1 Weighted", "F1 Macro"]
    models = list(metrics.keys())
    x, w   = np.arange(len(keys)), 0.2

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(models):
        bars = ax.bar(x + i * w, [metrics[m][k] for k in keys], w,
                      label=m, color=COLORS[m], alpha=0.9, edgecolor="white")
        _annotate(ax, bars)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Score", fontsize=11)
    _style(ax)
    ax.set_title("4-Way Model Comparison — Overall Metrics\n"
                 "BERT / FinBERT  ×  Zero-Shot / Fine-Tuned  |  Test: Financial PhraseBank",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/metrics_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/metrics_comparison.png")


def plot_per_class_f1(metrics: dict):
    models = list(metrics.keys())
    x, w   = np.arange(len(LABEL_NAMES)), 0.2

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(models):
        bars = ax.bar(x + i * w, [metrics[m]["f1_per_class"][c] for c in LABEL_NAMES], w,
                      label=m, color=COLORS[m], alpha=0.9, edgecolor="white")
        _annotate(ax, bars)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([c.capitalize() for c in LABEL_NAMES], fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("F1 Score", fontsize=11)
    _style(ax)
    ax.set_title("Per-Class F1 — 4-Way Model Comparison",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/per_class_f1.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/per_class_f1.png")


def plot_confusion_matrices(metrics: dict):
    models = list(metrics.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    fig.suptitle("Confusion Matrices — 4-Way Model Comparison\n(test: Financial PhraseBank)",
                 fontsize=13, fontweight="bold", y=1.01)
    for ax, m in zip(axes, models):
        cm = np.array(metrics[m]["confusion_matrix"])
        ConfusionMatrixDisplay(
            cm, display_labels=[c.capitalize() for c in LABEL_NAMES]
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{m}\nacc={metrics[m]['accuracy']:.3f}  f1={metrics[m]['f1_weighted']:.3f}",
                     fontsize=9, fontweight="bold", pad=8)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/confusion_matrices.png")


def plot_improvement(metrics: dict):
    """Delta of all models vs BERT zero-shot baseline."""
    baseline_m = metrics["BERT (zero-shot)"]
    compare    = ["FinBERT (zero-shot)", "BERT-base (fine-tuned)", "FinBERT (fine-tuned)"]
    keys  = ["accuracy", "f1_weighted", "f1_macro"]
    xlbls = ["Accuracy", "F1 Weighted", "F1 Macro"]
    x, w  = np.arange(len(keys)), 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, m in enumerate(compare):
        deltas = [metrics[m][k] - baseline_m[k] for k in keys]
        bars   = ax.bar(x + i * w, deltas, w, label=m,
                        color=COLORS[m], alpha=0.9, edgecolor="white")
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.003 if d >= 0 else -0.015),
                    f"{d:+.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(x + w)
    ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylabel("Δ Score vs BERT Zero-Shot", fontsize=11)
    _style(ax)
    ax.set_title("Improvement Over BERT Zero-Shot\n"
                 "Isolates contribution of domain pre-training and task fine-tuning",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/improvement_over_bert_zeroshot.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/improvement_over_bert_zeroshot.png")


def plot_interaction_heatmap(metrics: dict):
    """
    2×2 heatmap: rows = pre-training (BERT / FinBERT),
                 cols = fine-tuning (zero-shot / fine-tuned)
    Cell value = weighted F1.
    """
    data = np.array([
        [metrics["BERT (zero-shot)"]["f1_weighted"],
         metrics["BERT-base (fine-tuned)"]["f1_weighted"]],
        [metrics["FinBERT (zero-shot)"]["f1_weighted"],
         metrics["FinBERT (fine-tuned)"]["f1_weighted"]],
    ])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Weighted F1")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Zero-Shot", "Fine-Tuned"], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Generic\n(BERT-base)", "Finance-domain\n(FinBERT)"], fontsize=11)

    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{data[r, c]:.3f}",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if data[r, c] > 0.65 else "black")

    ax.set_title("F1-Weighted: Pre-training × Fine-tuning Interaction",
                 fontsize=11, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/interaction_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/interaction_heatmap.png")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    texts, true_labels = load_phrasebank()

    all_metrics = {}

    # 1. BERT zero-shot (fill-mask)
    bert_zs_preds = predict_bert_zero_shot(texts)
    all_metrics["BERT (zero-shot)"] = compute_metrics(true_labels, bert_zs_preds, "BERT (zero-shot)")
    np.save(f"{RESULTS_DIR}/BERT_zero_shot_preds.npy", bert_zs_preds)

    # 2. FinBERT zero-shot (text-classification, no fine-tuning)
    finbert_zs_preds = predict(FINBERT_ZS_MODEL, "FinBERT (zero-shot)", texts)
    all_metrics["FinBERT (zero-shot)"] = compute_metrics(true_labels, finbert_zs_preds, "FinBERT (zero-shot)")
    np.save(f"{RESULTS_DIR}/FinBERT_zero_shot_preds.npy", finbert_zs_preds)

    # 3. Fine-tuned BERT-base
    bert_ft_preds = predict(BERT_FT_DIR, "BERT-base (fine-tuned)", texts)
    all_metrics["BERT-base (fine-tuned)"] = compute_metrics(true_labels, bert_ft_preds, "BERT-base (fine-tuned)")
    np.save(f"{RESULTS_DIR}/BERT_base_finetuned_preds.npy", bert_ft_preds)

    # 4. Fine-tuned FinBERT
    finbert_ft_preds = predict(FINBERT_FT_DIR, "FinBERT (fine-tuned)", texts)
    all_metrics["FinBERT (fine-tuned)"] = compute_metrics(true_labels, finbert_ft_preds, "FinBERT (fine-tuned)")
    np.save(f"{RESULTS_DIR}/FinBERT_finetuned_preds.npy", finbert_ft_preds)

    # JSON summary
    summary = {k: {m: v for m, v in vals.items() if m != "confusion_matrix"}
               for k, vals in all_metrics.items()}
    with open(f"{RESULTS_DIR}/final_comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved → {RESULTS_DIR}/final_comparison.json")

    # Terminal table
    models = list(all_metrics.keys())
    col_w  = 22
    print(f"\n{'='*86}")
    print(f"  {'Metric':<20}" + "".join(f"{m:>{col_w}}" for m in models))
    print(f"{'='*86}")
    for m in ["accuracy", "f1_weighted", "f1_macro"]:
        row = f"  {m:<20}"
        for tag in models:
            row += f" {all_metrics[tag][m]:>{col_w}.4f}"
        print(row)
    print(f"{'='*86}")

    print("\nGenerating charts ...")
    plot_metrics(all_metrics)
    plot_per_class_f1(all_metrics)
    plot_confusion_matrices(all_metrics)
    plot_improvement(all_metrics)
    plot_interaction_heatmap(all_metrics)

    print(f"\n✓ Done. Results saved to {RESULTS_DIR}/")
