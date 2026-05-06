"""
compare_3way.py  —  3-Way Model Comparison
==========================================
Evaluates and compares three models on the Financial PhraseBank
(sentences_allagree) held-out test set:

  ┌───────────────────────────┬───────────────────────────────────────────┐
  │ Model                     │ Description                               │
  ├───────────────────────────┼───────────────────────────────────────────┤
  │ BERT (Baseline)           │ bert-base-uncased, zero-shot via fill-mask│
  │ BERT (Fine-tuned)         │ bert-base-uncased fine-tuned on merged    │
  │                           │   LM+VADER-relabeled dataset              │
  │ FinBERT                   │ ProsusAI/finbert, zero-shot               │
  │                           │   (finance-domain pre-trained)            │
  └───────────────────────────┴───────────────────────────────────────────┘

Metrics reported:
  - Accuracy
  - Macro F1
  - Weighted F1
  - Per-class Precision, Recall, F1
  - Confusion matrices

Charts saved to results/comparison_3way/:
  metrics_comparison.png     — grouped bar chart of overall metrics
  per_class_f1.png           — per-class F1 comparison
  confusion_matrices.png     — 1×3 grid of confusion matrices
  improvement_over_bert.png  — delta vs BERT baseline

Run AFTER finetune_bert_merged.py has saved a model to
  models/bert_base_finetuned_merged/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
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
_HERE               = os.path.dirname(os.path.abspath(__file__))
BERT_BASELINE_MODEL = "bert-base-uncased"
BERT_FINETUNED_DIR  = os.path.join(_HERE, "models", "bert_base_finetuned_merged")
FINBERT_MODEL       = "ProsusAI/finbert"
RESULTS_DIR         = os.path.join(_HERE, "results", "comparison_3way")
# Held-out 20% PhraseBank test set created by prepare_merged.py
PHRASEBANK_TEST     = os.path.join(_HERE, "data", "phrasebank_test.csv")
BATCH_SIZE          = 32
DEVICE              = 0 if torch.cuda.is_available() else -1

# BERT zero-shot fill-mask settings
ZS_CANDIDATES = ["positive", "negative", "neutral"]
ZS_TEMPLATE   = "Overall the financial sentiment is [MASK] ."

ID2LABEL    = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID    = {v: k for k, v in ID2LABEL.items()}
LABEL_NAMES = list(ID2LABEL.values())

COLORS = {
    "BERT (Baseline)":   "#adb5bd",   # gray
    "BERT (Fine-tuned)": "#74c0fc",   # blue
    "FinBERT":           "#198754",   # green
}

# ── Test set: held-out 20% PhraseBank (created by prepare_merged.py) ──────────
def load_test_set() -> tuple:
    """
    Load the held-out PhraseBank 20% test set saved by prepare_merged.py.
    This is the SAME split that was excluded from training, ensuring a
    clean train/test separation.
    """
    if not os.path.exists(PHRASEBANK_TEST):
        raise FileNotFoundError(
            f"Test set not found at {PHRASEBANK_TEST}.\n"
            "Run prepare_merged.py first to generate the 80/20 PhraseBank split."
        )
    df     = pd.read_csv(PHRASEBANK_TEST)
    texts  = df["text"].tolist()
    labels = np.array(df["label_id"].tolist())
    print(f"Test set loaded: {len(texts)} samples  |  "
          f"pos={sum(labels==0)}  neg={sum(labels==1)}  neu={sum(labels==2)}")
    return texts, labels

# ── Offline-safe model loader ──────────────────────────────────────────────────
def _load_pretrained(cls, model_id: str, **kwargs):
    """
    Try loading with network first; fall back to local cache if offline.
    Raises a clear error if the model has never been downloaded.
    """
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except Exception as e:
        if "connection" in str(e).lower() or "nodename" in str(e).lower() \
                or "ConnectError" in type(e).__name__:
            print(f"  Network unavailable — loading {model_id} from local cache ...")
            try:
                return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
            except Exception:
                raise RuntimeError(
                    f"\n❌  '{model_id}' is not in the local HuggingFace cache "
                    f"and the network is unreachable.\n"
                    f"   Run this once while online to pre-download all models:\n"
                    f"       python download_models.py\n"
                )
        raise

# ── BERT zero-shot via fill-mask ───────────────────────────────────────────────
def predict_bert_baseline(texts: list) -> np.ndarray:
    print(f"\n[BERT (Baseline)] Loading {BERT_BASELINE_MODEL} ...")
    try:
        filler = pipeline("fill-mask", model=BERT_BASELINE_MODEL, device=DEVICE)
    except Exception as e:
        if "connection" in str(e).lower() or "nodename" in str(e).lower():
            print("  Network unavailable — loading from local cache ...")
            filler = pipeline("fill-mask", model=BERT_BASELINE_MODEL,
                              device=DEVICE, local_files_only=True)
        else:
            raise
    tokenizer = filler.tokenizer

    for w in ZS_CANDIDATES:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"'{w}' is not a single token — cannot use as fill-mask target.")

    print(f"[BERT (Baseline)] Scoring {len(texts)} samples ...")
    preds = []
    for text in texts:
        prompt  = f"{ZS_TEMPLATE} {text[:300]}"
        results = filler(prompt, targets=ZS_CANDIDATES)
        best    = max(results, key=lambda r: r["score"])
        preds.append(LABEL2ID[best["token_str"].lower()])

    print("[BERT (Baseline)] Done.")
    return np.array(preds)

# ── Text-classification inference (fine-tuned BERT + FinBERT) ─────────────────
def predict_classifier(model_id: str, tag: str, texts: list) -> np.ndarray:
    print(f"\n[{tag}] Loading: {model_id}")
    tokenizer = _load_pretrained(AutoTokenizer, model_id)
    model     = _load_pretrained(AutoModelForSequenceClassification, model_id)
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
    preds = np.array([LABEL2ID[p["label"].lower()] for p in raw])
    print(f"[{tag}] Done.")
    return preds

# ── Compute & display metrics ──────────────────────────────────────────────────
def compute_metrics(true: np.ndarray, preds: np.ndarray, tag: str) -> dict:
    report_str = classification_report(
        true, preds, target_names=LABEL_NAMES, zero_division=0
    )
    print(f"\n{'='*60}\n  {tag}\n{'='*60}")
    print(report_str)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    safe = tag.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    with open(f"{RESULTS_DIR}/{safe}_report.txt", "w") as fh:
        fh.write(f"Model: {tag}\n{'='*60}\n{report_str}")

    prec, rec, f1_per, _ = precision_recall_fscore_support(
        true, preds, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "accuracy":     round(accuracy_score(true, preds), 4),
        "f1_macro":     round(f1_score(true, preds, average="macro",    zero_division=0), 4),
        "f1_weighted":  round(f1_score(true, preds, average="weighted", zero_division=0), 4),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": round(float(prec[i]), 4),
                "recall":    round(float(rec[i]),  4),
                "f1":        round(float(f1_per[i]), 4),
            }
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(true, preds).tolist(),
    }

# ── Plot: overall metrics bar chart ───────────────────────────────────────────
def plot_metrics(metrics: dict):
    keys   = ["accuracy", "f1_macro", "f1_weighted"]
    xlbls  = ["Accuracy", "F1 Macro", "F1 Weighted"]
    models = list(metrics.keys())
    x, w   = np.arange(len(keys)), 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, m in enumerate(models):
        bars = ax.bar(
            x + i * w, [metrics[m][k] for k in keys], w,
            label=m, color=COLORS[m], alpha=0.9, edgecolor="white",
        )
        for bar in bars:
            v = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.007,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x + w)
    ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "3-Way Model Comparison — Overall Metrics\n"
        "BERT Baseline  |  BERT Fine-tuned  |  FinBERT\n"
        "Test set: Financial PhraseBank (AllAgree)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/metrics_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/metrics_comparison.png")


# ── Plot: per-class F1 ────────────────────────────────────────────────────────
def plot_per_class_f1(metrics: dict):
    models = list(metrics.keys())
    x, w   = np.arange(len(LABEL_NAMES)), 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, m in enumerate(models):
        vals = [metrics[m]["per_class"][c]["f1"] for c in LABEL_NAMES]
        bars = ax.bar(
            x + i * w, vals, w,
            label=m, color=COLORS[m], alpha=0.9, edgecolor="white",
        )
        for bar in bars:
            v = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.007,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xticks(x + w)
    ax.set_xticklabels([c.capitalize() for c in LABEL_NAMES], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Per-Class F1  —  3-Way Comparison", fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/per_class_f1.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/per_class_f1.png")


# ── Plot: confusion matrices ──────────────────────────────────────────────────
def plot_confusion_matrices(metrics: dict):
    models = list(metrics.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Confusion Matrices  —  3-Way Model Comparison\n(Test: Financial PhraseBank AllAgree)",
        fontsize=12, fontweight="bold",
    )
    for ax, m in zip(axes, models):
        cm = np.array(metrics[m]["confusion_matrix"])
        ConfusionMatrixDisplay(
            cm, display_labels=[c.capitalize() for c in LABEL_NAMES]
        ).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(
            f"{m}\nAcc={metrics[m]['accuracy']:.3f}  F1={metrics[m]['f1_weighted']:.3f}",
            fontsize=9, fontweight="bold", pad=8,
        )
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/confusion_matrices.png")


# ── Plot: improvement over BERT baseline ──────────────────────────────────────
def plot_improvement(metrics: dict):
    baseline = metrics["BERT (Baseline)"]
    compare  = ["BERT (Fine-tuned)", "FinBERT"]
    keys     = ["accuracy", "f1_macro", "f1_weighted"]
    xlbls    = ["Accuracy", "F1 Macro", "F1 Weighted"]
    x, w     = np.arange(len(keys)), 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(compare):
        deltas = [metrics[m][k] - baseline[k] for k in keys]
        bars   = ax.bar(
            x + i * w, deltas, w,
            label=m, color=COLORS[m], alpha=0.9, edgecolor="white",
        )
        for bar, d in zip(bars, deltas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.004 if d >= 0 else -0.018),
                f"{d:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(xlbls, fontsize=11)
    ax.set_ylabel("Δ Score vs BERT Baseline", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "Improvement Over BERT Baseline (Zero-Shot)\n"
        "Shows contribution of fine-tuning and domain pre-training",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/improvement_over_bert.png", dpi=150)
    plt.close(fig)
    print(f"Saved → {RESULTS_DIR}/improvement_over_bert.png")


# ── Terminal summary table ─────────────────────────────────────────────────────
def print_summary_table(metrics: dict):
    models = list(metrics.keys())
    cols   = ["accuracy", "f1_macro", "f1_weighted"]
    col_w  = 22
    sep    = "=" * (24 + col_w * len(models))

    print(f"\n{sep}")
    print(f"  {'Metric':<22}" + "".join(f"{m:>{col_w}}" for m in models))
    print(sep)
    for c in cols:
        row = f"  {c:<22}" + "".join(f" {metrics[m][c]:>{col_w-1}.4f}" for m in models)
        print(row)
    print(sep)
    print("\nPer-class F1:")
    for cls in LABEL_NAMES:
        row = f"  F1 {cls:<18}" + "".join(
            f" {metrics[m]['per_class'][cls]['f1']:>{col_w-1}.4f}" for m in models
        )
        print(row)
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    texts, true_labels = load_test_set()
    all_metrics = {}

    # 1. BERT Baseline (zero-shot fill-mask)
    preds = predict_bert_baseline(texts)
    all_metrics["BERT (Baseline)"] = compute_metrics(true_labels, preds, "BERT (Baseline)")
    np.save(f"{RESULTS_DIR}/bert_baseline_preds.npy", preds)

    # 2. BERT Fine-tuned on merged LM+VADER-relabeled dataset
    if not os.path.isdir(BERT_FINETUNED_DIR):
        raise FileNotFoundError(
            f"Fine-tuned model not found at {BERT_FINETUNED_DIR}.\n"
            "Run finetune_bert_merged.py first."
        )
    preds = predict_classifier(BERT_FINETUNED_DIR, "BERT (Fine-tuned)", texts)
    all_metrics["BERT (Fine-tuned)"] = compute_metrics(true_labels, preds, "BERT (Fine-tuned)")
    np.save(f"{RESULTS_DIR}/bert_finetuned_preds.npy", preds)

    # 3. FinBERT zero-shot (finance-domain pre-trained)
    preds = predict_classifier(FINBERT_MODEL, "FinBERT", texts)
    all_metrics["FinBERT"] = compute_metrics(true_labels, preds, "FinBERT")
    np.save(f"{RESULTS_DIR}/finbert_preds.npy", preds)

    # Print table
    print_summary_table(all_metrics)

    # Save JSON
    summary = {
        k: {m: v for m, v in vals.items() if m != "confusion_matrix"}
        for k, vals in all_metrics.items()
    }
    with open(f"{RESULTS_DIR}/comparison_3way.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved → {RESULTS_DIR}/comparison_3way.json")

    # Generate charts
    print("\nGenerating charts ...")
    plot_metrics(all_metrics)
    plot_per_class_f1(all_metrics)
    plot_confusion_matrices(all_metrics)
    plot_improvement(all_metrics)

    print(f"\n✓ Done. Results saved to {RESULTS_DIR}/")
