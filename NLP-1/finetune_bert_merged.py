"""
finetune_bert_merged.py  —  Fine-Tune bert-base-uncased on Financial Sentiment Dataset
=======================================================================================
Fine-tunes bert-base-uncased on data/financial_sentiment.csv.

Prerequisites:
  - data/financial_sentiment.csv must exist
  - data/tokenized_dataset_merged/ must exist (run data_preparation.py first)

Output:
  models/bert_base_finetuned_merged/   (best checkpoint, ready for inference)

Architecture choices:
  - All 12 encoder layers trainable (FREEZE_LAYERS = 0)
  - Class-weighted cross-entropy to handle class imbalance
  - Early stopping (patience 3) to prevent overfitting
  - fp16 training when a CUDA GPU is available

Label mapping:
  0 → positive  |  1 → negative  |  2 → neutral
"""

import os
import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ── Config ────────────────────────────────────────────────────────────────────
_HERE         = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME    = "bert-base-uncased"
DATASET_DIR   = os.path.join(_HERE, "data", "tokenized_dataset_merged")
OUTPUT_DIR    = os.path.join(_HERE, "models", "bert_base_finetuned_merged")
LOGGING_DIR   = os.path.join(_HERE, "logs", "bert_merged")

LEARNING_RATE  = 1e-5       # lower LR: generic BERT needs gentler adaptation
NUM_EPOCHS     = 10         # more epochs to compensate for generic pre-training
BATCH_SIZE     = 8          # halved to fit MPS unified memory on Apple M4
GRAD_ACCUM     = 2          # accumulate 2 steps → effective batch = 16 (same as before)
WEIGHT_DECAY   = 0.01
WARMUP_RATIO   = 0.1
EARLY_STOP_PAT = 3          # more patience to match longer training

ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── Class weights ─────────────────────────────────────────────────────────────
def compute_class_weights(label_ids) -> torch.Tensor:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(ID2LABEL.keys())),
        y=np.array(label_ids),
    )
    print(f"Class weights: { {ID2LABEL[i]: round(w, 3) for i, w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float)

# ── Weighted trainer (handles class imbalance) ────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total")
    return model

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro":    f1_score(labels, preds, average="macro"),
    }

# ── Training args ─────────────────────────────────────────────────────────────
def get_training_args():
    return TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        logging_dir                 = LOGGING_DIR,
        num_train_epochs            = NUM_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        warmup_ratio                = WARMUP_RATIO,
        gradient_accumulation_steps = GRAD_ACCUM,  # effective batch = BATCH_SIZE * GRAD_ACCUM
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_weighted",
        greater_is_better           = True,
        lr_scheduler_type           = "cosine",   # cosine decay → smoother convergence
        logging_steps               = 50,
        report_to                   = "none",
        fp16                        = torch.cuda.is_available(),
        seed                        = 42,
    )

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset   = DatasetDict.load_from_disk(DATASET_DIR)
    model     = load_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    label_ids     = [int(l) for l in dataset["train"]["labels"]]
    class_weights = compute_class_weights(label_ids)

    trainer = WeightedTrainer(
        class_weights   = class_weights,
        model           = model,
        args            = get_training_args(),
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["val"],
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PAT)],
    )

    print("\n── Fine-tuning bert-base-uncased on merged dataset ──────────────")
    print(f"   Model:      {MODEL_NAME}")
    print(f"   Dataset:    {DATASET_DIR}  ({len(dataset['train'])} train samples)")
    print(f"   Epochs:     {NUM_EPOCHS}  |  LR: {LEARNING_RATE}  |  Batch: {BATCH_SIZE}")
    trainer.train()

    print("\n── Saving best model ─────────────────────────────────────────────")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Saved → {OUTPUT_DIR}/")
