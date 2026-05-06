"""
download_models.py  —  Pre-download all models needed for the comparison
=========================================================================
Run this ONCE while you have internet access. All models are saved to the
HuggingFace local cache (~/.cache/huggingface/hub/) and loaded from there
automatically when compare_3way.py runs offline.

Models downloaded:
  - bert-base-uncased        (BERT Baseline + Fine-tuning base)
  - ProsusAI/finbert         (FinBERT zero-shot)

Also downloads:
  - Financial PhraseBank dataset (via huggingface_hub)

Usage:
  python download_models.py
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

MODELS = [
    "bert-base-uncased",
    "ProsusAI/finbert",
]

if __name__ == "__main__":
    for model_id in MODELS:
        print(f"\nDownloading: {model_id} ...")
        AutoTokenizer.from_pretrained(model_id)
        AutoModelForSequenceClassification.from_pretrained(model_id)
        print(f"  ✓ {model_id} cached.")

    print("\nDownloading Financial PhraseBank ...")
    hf_hub_download(
        repo_id="takala/financial_phrasebank",
        filename="data/FinancialPhraseBank-v1.0.zip",
        repo_type="dataset",
    )
    print("  ✓ Financial PhraseBank cached.")

    print("\n✓ All downloads complete. You can now run compare_3way.py offline.")
