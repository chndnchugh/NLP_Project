"""
plot_training.py  —  Training Curves for bert_base_finetuned_merged
====================================================================
Reads trainer_state.json from the latest checkpoint and saves three
separate figures:

  results/training_loss.png       — Training loss vs step (raw + smoothed)
  results/validation_loss.png     — Validation loss vs epoch
  results/validation_metrics.png  — Accuracy, F1 Weighted, F1 Macro vs epoch

Best checkpoint is marked on all plots.

Usage:
  python plot_training.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(_HERE, "models", "bert_base_finetuned_merged")
RESULTS_DIR = os.path.join(_HERE, "results")

BLUE       = "#4c72b0"
GREEN      = "#55a868"
ORANGE     = "#dd8452"
PURPLE     = "#8172b2"
BEST_COLOR = "#c44e52"

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_trainer_state(model_dir: str) -> str:
    checkpoints = sorted(
        [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return os.path.join(model_dir, checkpoints[-1], "trainer_state.json")


def load_logs(state_path: str):
    with open(state_path) as f:
        state = json.load(f)

    train_logs, val_logs = [], []
    for entry in state["log_history"]:
        if "loss" in entry and "eval_loss" not in entry:
            train_logs.append(entry)
        elif "eval_loss" in entry:
            val_logs.append(entry)

    return train_logs, val_logs, state.get("best_global_step"), state.get("best_metric")


def style_ax(ax):
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ── Plot 1: Training Loss ─────────────────────────────────────────────────────
def plot_training_loss(train_logs, best_step):
    t_steps = [e["step"] for e in train_logs]
    t_loss  = [e["loss"] for e in train_logs]

    window   = 10
    t_smooth = np.convolve(t_loss, np.ones(window) / window, mode="valid")
    s_smooth = t_steps[window - 1:]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_steps, t_loss, color=BLUE, alpha=0.25, linewidth=0.8,
            label="Raw (every 50 steps)")
    ax.plot(s_smooth, t_smooth, color=BLUE, linewidth=2,
            label=f"Smoothed (window={window})")
    if best_step:
        ax.axvline(best_step, color=BEST_COLOR, linestyle="--", linewidth=1.5,
                   label=f"Best checkpoint (step {best_step})")

    ax.set_title("Training Loss — bert-base-uncased Fine-tuned on Financial Sentiment",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    return fig


# ── Plot 2: Validation Loss ───────────────────────────────────────────────────
def plot_validation_loss(val_logs, best_step):
    v_epochs = [e["epoch"]    for e in val_logs]
    v_loss   = [e["eval_loss"] for e in val_logs]

    best_val   = next((e for e in val_logs if e["step"] == best_step), None)
    best_epoch = best_val["epoch"] if best_val else None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(v_epochs, v_loss, color=ORANGE, linewidth=2, marker="o",
            markersize=6, label="Validation Loss")
    if best_epoch:
        best_vl = best_val["eval_loss"]
        ax.scatter([best_epoch], [best_vl], color=BEST_COLOR, zorder=5, s=100,
                   label=f"Best checkpoint — epoch {int(best_epoch)}, loss = {best_vl:.4f}")

    ax.set_title("Validation Loss per Epoch",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    return fig


# ── Plot 3: Validation Metrics ────────────────────────────────────────────────
def plot_validation_metrics(val_logs, best_step):
    v_epochs = [e["epoch"]            for e in val_logs]
    v_acc    = [e["eval_accuracy"]    for e in val_logs]
    v_f1w    = [e["eval_f1_weighted"] for e in val_logs]
    v_f1m    = [e["eval_f1_macro"]    for e in val_logs]

    best_val   = next((e for e in val_logs if e["step"] == best_step), None)
    best_epoch = best_val["epoch"] if best_val else None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(v_epochs, v_acc, color=GREEN,  linewidth=2, marker="o",
            markersize=6, label="Accuracy")
    ax.plot(v_epochs, v_f1w, color=BLUE,   linewidth=2, marker="s",
            markersize=6, label="F1 Weighted")
    ax.plot(v_epochs, v_f1m, color=PURPLE, linewidth=2, marker="^",
            markersize=6, label="F1 Macro")
    if best_epoch:
        ax.axvline(best_epoch, color=BEST_COLOR, linestyle="--", linewidth=1.5,
                   label=f"Best checkpoint — epoch {int(best_epoch)}  "
                         f"(acc={best_val['eval_accuracy']:.4f}, "
                         f"f1w={best_val['eval_f1_weighted']:.4f})")

    ax.set_title("Validation Metrics per Epoch",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    style_ax(ax)
    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    state_path = find_trainer_state(MODEL_DIR)
    print(f"Loading training state from: {state_path}")

    train_logs, val_logs, best_step, best_metric = load_logs(state_path)
    print(f"  Training log entries : {len(train_logs)}")
    print(f"  Validation epochs    : {len(val_logs)}")
    print(f"  Best step            : {best_step}  (F1 weighted = {best_metric:.4f})")
    print()

    plots = [
        ("training_loss.png",       plot_training_loss(train_logs, best_step)),
        ("validation_loss.png",     plot_validation_loss(val_logs, best_step)),
        ("validation_metrics.png",  plot_validation_metrics(val_logs, best_step)),
    ]

    for filename, fig in plots:
        out = os.path.join(RESULTS_DIR, filename)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved → {out}")
