"""
Stage 3 (alt) V2: Fine-Tune bert-base-uncased — Improved for Negative Class
=============================================================================
Problem with V1 (bert_base_finetuned):
  - Negative recall: 0.15  (model almost never predicts negative)
  - Balanced class weights gave negative a 2.5× boost but wasn't enough
  - LR 2e-5 may be too aggressive — model converges to a neutral-heavy local minimum

Changes in V2:
  1. Focal Loss (γ=2)
       Standard CrossEntropy treats all misclassified examples equally.
       Focal loss down-weights easy/already-correct examples and forces the
       model to focus on hard ones — exactly the negative class that keeps
       getting missed. FL(p_t) = -(1 - p_t)^γ · log(p_t)

  2. Stronger manual negative weight
       Balanced weighting computed negative at ~2.5×. This version boosts it
       to ~4× neutral so the model is heavily penalised for missing negatives.
       Weights: positive=2.0, negative=4.0, neutral=1.0

  3. Lower learning rate: 1e-5 (was 2e-5)
       Slower, more careful updates — prevents the model from collapsing to
       predicting neutral as a dominant shortcut early in training.

  4. Label smoothing: 0.1
       Prevents overconfident predictions on noisy LM+VADER labels.

  5. More epochs + patience: 8 epochs, early_stop_patience=3
       Gives the model more time to learn the harder negative pattern.

  6. Higher warmup: 0.15 (was 0.1)
       Longer warmup before full LR — reduces early training instability.

Output: models/bert_base_finetuned_v2/
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME      = "bert-base-uncased"
DATASET_DIR     = "data/tokenized_dataset"
OUTPUT_DIR      = "models/bert_base_finetuned_v2"
LOGGING_DIR     = "logs"

FREEZE_LAYERS   = 0          # no freezing — bert-base needs all layers to adapt
LEARNING_RATE   = 1e-5       # lower than V1 (was 2e-5)
NUM_EPOCHS      = 8          # more room for hard class learning (was 5)
BATCH_SIZE      = 16
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.15       # longer warmup (was 0.1)
EARLY_STOP_PAT  = 3          # more patience (was 2)
FOCAL_GAMMA     = 2.0        # focal loss exponent: 2 is standard
LABEL_SMOOTHING = 0.1

# Manual class weights — aggressively upweight negative to fix recall=0.15
# positive=2.0, negative=4.0, neutral=1.0
# Negative gets 4× the gradient signal of neutral to force the model to learn it
CLASS_WEIGHTS = torch.tensor([2.0, 4.0, 1.0], dtype=torch.float)

ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy (high-confidence) examples so training focuses on
    hard cases — in this project, the negative class the model keeps missing.
    """
    def __init__(self, weight: torch.Tensor, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard CE with class weights and label smoothing
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight.to(logits.device),
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce_loss)
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ── Focal Trainer ──────────────────────────────────────────────────────────────
class FocalTrainer(Trainer):
    def __init__(self, focal_loss_fn: FocalLoss, **kwargs):
        super().__init__(**kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = self.focal_loss_fn(outputs.logits, labels)
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
    print(f"All layers trainable: {trainable:,} / {total:,} params")
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro":    f1_score(labels, preds, average="macro"),
        "f1_negative": f1_score(labels, preds, labels=[1], average="macro"),  # track negative specifically
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
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_macro",    # macro F1 penalises low-negative equally
        greater_is_better           = True,
        logging_steps               = 50,
        report_to                   = "none",
        fp16                        = torch.cuda.is_available(),
        seed                        = 42,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    dataset   = DatasetDict.load_from_disk(DATASET_DIR)
    model     = load_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"\nClass weights: "
          f"positive={CLASS_WEIGHTS[0]:.1f}, "
          f"negative={CLASS_WEIGHTS[1]:.1f}, "
          f"neutral={CLASS_WEIGHTS[2]:.1f}")

    focal_loss_fn = FocalLoss(
        weight=CLASS_WEIGHTS,
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
    )

    trainer = FocalTrainer(
        focal_loss_fn   = focal_loss_fn,
        model           = model,
        args            = get_training_args(),
        train_dataset   = dataset["train"],
        eval_dataset    = dataset["val"],
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PAT)],
    )

    print("\n── Fine-tuning bert-base-uncased V2 ─────────────────────────────")
    print(f"   Model:          {MODEL_NAME}")
    print(f"   Loss:           Focal Loss (γ={FOCAL_GAMMA}, label_smooth={LABEL_SMOOTHING})")
    print(f"   Class weights:  pos={CLASS_WEIGHTS[0]:.1f}  neg={CLASS_WEIGHTS[1]:.1f}  neu={CLASS_WEIGHTS[2]:.1f}")
    print(f"   LR:             {LEARNING_RATE}  |  Epochs: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"   Best model by:  f1_macro  (penalises low negative F1 equally)")
    trainer.train()

    print("\n── Saving best model ────────────────────────────────────────────")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved to {OUTPUT_DIR}/")
