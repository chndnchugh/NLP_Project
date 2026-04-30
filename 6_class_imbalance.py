"""
Stage 6: Class Imbalance Handling
Four strategies — use Strategy 2 always, add 3 or 4 if ratio > 5:1

  Strategy 1 — Analyse the imbalance
  Strategy 2 — Weighted cross-entropy loss (WeightedTrainer)
  Strategy 3 — Random oversampling
  Strategy 4 — Synonym augmentation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "ProsusAI/finbert"
DATA_PATH  = "data/financial_sentiment.csv"
TEXT_COL   = "Headline"
LABEL_COL  = "Sentiment"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# =============================================================================
# STRATEGY 1 — Analyse the imbalance
# =============================================================================

def analyse_imbalance(df: pd.DataFrame) -> dict:
    counts   = df[LABEL_COL].value_counts()
    total    = len(df)
    majority = counts.max()

    print("\n── Class distribution ───────────────────────────────────────────")
    for label, count in counts.items():
        bar   = "█" * int(30 * count / majority)
        ratio = majority / count
        print(f"  {label:<12} {count:>6} ({100*count/total:5.1f}%)  {bar:<30}  ratio 1:{ratio:.1f}")

    imbalance_ratio = majority / counts.min()
    print(f"\n  Majority/minority ratio: {imbalance_ratio:.1f}x")
    if imbalance_ratio < 2:
        print("  Assessment: balanced — no mitigation needed.")
    elif imbalance_ratio < 5:
        print("  Assessment: mild — use WeightedTrainer (Strategy 2).")
    else:
        print("  Assessment: severe — use WeightedTrainer + augmentation.")

    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3))
    colors  = ["#E24B4A", "#888780", "#1D9E75"]
    ax.bar(counts.index, counts.values, color=colors[:len(counts)])
    ax.set_ylabel("Samples")
    ax.set_title("Class distribution")
    plt.tight_layout()
    plt.savefig("results/class_distribution.png", dpi=150)
    plt.close()
    print("  Saved: results/class_distribution.png")

    return {"counts": counts.to_dict(), "imbalance_ratio": imbalance_ratio}


# =============================================================================
# STRATEGY 2 — Weighted cross-entropy loss
# =============================================================================

def compute_class_weights(label_ids: list) -> torch.Tensor:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(ID2LABEL.keys())),
        y=np.array(label_ids),
    )
    print(f"\nClass weights: { {ID2LABEL[i]: round(w, 3) for i, w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    """
    Drop-in replacement for HuggingFace Trainer with per-class loss weighting.
    Swap this in place of Trainer in 3_finetune.py.
    """

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── How to use WeightedTrainer in 3_finetune.py: ─────────────────────────────
#
#   from 6_class_imbalance import compute_class_weights, WeightedTrainer
#
#   weights = compute_class_weights(dataset["train"]["labels"])
#   trainer = WeightedTrainer(
#       class_weights   = weights,
#       model           = model,
#       args            = get_training_args(),
#       train_dataset   = dataset["train"],
#       eval_dataset    = dataset["val"],
#       compute_metrics = compute_metrics,
#   )


# =============================================================================
# STRATEGY 3 — Random oversampling
# =============================================================================

def oversample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[LABEL_COL].value_counts()
    target = counts.max()
    parts  = []
    for label, count in counts.items():
        subset = df[df[LABEL_COL] == label]
        if count < target:
            extra  = subset.sample(n=target - count, replace=True, random_state=42)
            subset = pd.concat([subset, extra], ignore_index=True)
        parts.append(subset)
    balanced = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
    print(f"\nAfter oversampling: {balanced[LABEL_COL].value_counts().to_dict()}")
    return balanced


# =============================================================================
# STRATEGY 4 — Synonym augmentation
# =============================================================================

def augment_with_synonyms(
    df: pd.DataFrame,
    target_label:  str   = "negative",
    n_augments:    int   = 3,
    replace_ratio: float = 0.2,
) -> pd.DataFrame:
    try:
        import nltk
        from nltk.corpus import wordnet
        nltk.download("wordnet",                    quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("punkt",                      quiet=True)
    except ImportError:
        print("nltk not installed. Run: pip install nltk")
        return df

    def get_synonyms(word: str) -> list:
        syns = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate.lower() != word.lower():
                    syns.add(candidate)
        return list(syns)

    def augment_sentence(text: str) -> str:
        words    = text.split()
        n_swap   = max(1, int(len(words) * replace_ratio))
        indices  = np.random.choice(len(words), size=min(n_swap, len(words)), replace=False)
        new_words = words.copy()
        for idx in indices:
            syns = get_synonyms(words[idx])
            if syns:
                new_words[idx] = np.random.choice(syns)
        return " ".join(new_words)

    minority_df = df[df[LABEL_COL] == target_label].copy()
    new_rows    = []
    for _, row in minority_df.iterrows():
        for _ in range(n_augments):
            new_rows.append({
                TEXT_COL:  augment_sentence(row[TEXT_COL]),
                LABEL_COL: target_label,
            })

    augmented = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    augmented = augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nAfter synonym augmentation for '{target_label}':")
    print(augmented[LABEL_COL].value_counts())
    return augmented


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    df[LABEL_COL] = df[LABEL_COL].str.lower().str.strip()
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df = df[df[LABEL_COL].isin(LABEL2ID)]

    print("=== STRATEGY 1: Analyse ===")
    info = analyse_imbalance(df)

    print("\n=== STRATEGY 2: Class weights ===")
    df["label_id"] = df[LABEL_COL].map(LABEL2ID)
    weights = compute_class_weights(df["label_id"].tolist())
    print(f"Weights tensor: {weights}")

    if info["imbalance_ratio"] >= 5:
        print("\n=== STRATEGY 3: Oversample ===")
        _ = oversample_dataframe(df)

        print("\n=== STRATEGY 4: Synonym augmentation ===")
        minority = df[LABEL_COL].value_counts().idxmin()
        _ = augment_with_synonyms(df, target_label=minority, n_augments=2)

    print("\n── Recommendation ───────────────────────────────────────────────")
    print("  Always:       Strategy 2 (WeightedTrainer)")
    print("  Ratio > 5:1 : Also add Strategy 3 or 4")
