"""
Stage 3: Fine-Tune FinBERT (Memory-Optimised for 16GB MacBook Air)
====================================================================
Memory fixes:
  1. BATCH_SIZE = 4          (was 16)
  2. GRAD_ACCUM = 4          (effective batch = 4x4 = 16, same as before)
  3. MAX_LENGTH = 128        (was 512 — headlines rarely exceed 128 tokens)
  4. fp16 = False            (MPS/CPU on Mac doesn't support fp16 reliably)
  5. gradient_checkpointing  (trades compute for memory — ~40% RAM saving)
  6. dataloader_pin_memory=False  (prevents extra RAM copy on Mac)
  7. Freeze bottom 6 layers  (reduces optimizer state memory)
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
MODEL_NAME      = "ProsusAI/finbert"
DATASET_DIR     = "data/tokenized_dataset"
OUTPUT_DIR      = "models/finbert_finetuned_v2"
LOGGING_DIR     = "logs"

FREEZE_LAYERS   = 6        # freeze bottom 6 — saves optimizer RAM
LEARNING_RATE   = 5e-6
NUM_EPOCHS      = 3
BATCH_SIZE      = 4        # small batch — fits in 16GB
GRAD_ACCUM      = 4        # effective batch = 4 * 4 = 16
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
EARLY_STOP_PAT  = 2
LABEL_SMOOTHING = 0.1
DROPOUT         = 0.2

# Detect best available device on Mac
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}  # matches ProsusAI/finbert native mapping
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        classifier_dropout=DROPOUT,
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
    )

    # Gradient checkpointing — recomputes activations during backward pass
    # instead of storing them, saving ~40% memory at ~20% speed cost
    model.gradient_checkpointing_enable()

    # Freeze bottom N encoder layers — fewer optimizer states in RAM
    if FREEZE_LAYERS > 0:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in model.bert.encoder.layer[:FREEZE_LAYERS]:
            for param in layer.parameters():
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model

# ── Class weights ─────────────────────────────────────────────────────────────
def compute_class_weights(label_ids) -> torch.Tensor:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(list(ID2LABEL.keys())),
        y=np.array(label_ids),
    )
    print(f"Class weights: { {ID2LABEL[i]: round(w, 3) for i, w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float)

# ── WeightedTrainer ───────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            label_smoothing=LABEL_SMOOTHING,
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

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
        output_dir                        = OUTPUT_DIR,
        logging_dir                       = LOGGING_DIR,
        num_train_epochs                  = NUM_EPOCHS,
        per_device_train_batch_size       = BATCH_SIZE,
        per_device_eval_batch_size        = BATCH_SIZE,
        gradient_accumulation_steps       = GRAD_ACCUM,
        learning_rate                     = LEARNING_RATE,
        weight_decay                      = WEIGHT_DECAY,
        warmup_ratio                      = WARMUP_RATIO,
        eval_strategy                     = "epoch",
        save_strategy                     = "epoch",
        load_best_model_at_end            = True,
        metric_for_best_model             = "f1_weighted",
        greater_is_better                 = True,
        logging_steps                     = 50,
        report_to                         = "none",
        fp16                              = False,   # not reliable on Mac MPS/CPU
        bf16                              = False,
        dataloader_pin_memory             = False,   # prevents extra RAM copy on Mac
        dataloader_num_workers            = 0,       # avoids multiprocess memory overhead
        seed                              = 42,
    )

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

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

    print("\n── Starting fine-tuning (memory-optimised) ──────────────────────")
    print(f"   Batch size:        {BATCH_SIZE}")
    print(f"   Grad accumulation: {GRAD_ACCUM}  (effective batch = {BATCH_SIZE * GRAD_ACCUM})")
    print(f"   Frozen layers:     {FREEZE_LAYERS}")
    print(f"   Device:            {DEVICE}")
    trainer.train()

    print("\n── Saving best model ────────────────────────────────────────────")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}/")