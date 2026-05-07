"""
download_models.py  —  Download all models needed to run the project
=====================================================================
Run this ONCE after cloning the repo. Downloads:

  1. bert-base-uncased        → HuggingFace cache (BERT zero-shot + fine-tuning base)
  2. ProsusAI/finbert         → HuggingFace cache (FinBERT zero-shot)
  3. bert_base_finetuned_merged (checkpoint-3440)
                              → models/bert_base_finetuned_merged/  (local folder)

The fine-tuned model is hosted on HuggingFace Hub because the weights
(~418 MB) are too large for GitHub.

Usage:
  python download_models.py

Requirements:
  pip install -r requirements.txt
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# ── CONFIGURE — HF repo where the fine-tuned model is hosted ──────────────────
HF_FINETUNED_REPO = "chndnchugh/bert-base-financial-sentiment"   # ← update if repo changes
# ─────────────────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_HERE, "models", "bert_base_finetuned_merged")

BASE_MODELS = [
    "bert-base-uncased",
    "ProsusAI/finbert",
]

if __name__ == "__main__":
    # ── 1. Download base models into HuggingFace cache ───────────────────────
    for model_id in BASE_MODELS:
        print(f"\nDownloading: {model_id} ...")
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForSequenceClassification.from_pretrained(model_id)
        print(f"  ✓ {model_id} cached.")

    # ── 2. Download fine-tuned BERT from HuggingFace Hub ─────────────────────
    print(f"\nDownloading fine-tuned model from HuggingFace Hub ...")
    print(f"  Repo: {HF_FINETUNED_REPO}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for fname in ["config.json", "model.safetensors", "trainer_state.json"]:
        dest = os.path.join(MODEL_DIR, fname)
        if os.path.exists(dest):
            print(f"  ✓ {fname} already present, skipping.")
            continue
        print(f"  Downloading {fname} (~418 MB for model.safetensors) ...")
        hf_hub_download(
            repo_id=HF_FINETUNED_REPO,
            filename=fname,
            repo_type="model",
            local_dir=MODEL_DIR,
        )
        print(f"  ✓ {fname}")

    print(f"\n  Fine-tuned model saved → {MODEL_DIR}")

    # ── 3. Download Financial PhraseBank dataset ──────────────────────────────
    print("\nDownloading Financial PhraseBank dataset ...")
    hf_hub_download(
        repo_id="takala/financial_phrasebank",
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
    print("  ✓ Financial PhraseBank cached.")

    print("\n✓ All downloads complete. You can now run the full pipeline.")
