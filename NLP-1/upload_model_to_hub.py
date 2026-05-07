"""
upload_model_to_hub.py  —  Upload best fine-tuned checkpoint to HuggingFace Hub
================================================================================
Run this ONCE (from Chandan's machine) to publish the best checkpoint so
teammates can pull it with download_models.py.

Steps before running:
  1. pip install huggingface_hub
  2. huggingface-cli login          (paste your HF write-access token)
  3. Create a NEW MODEL repo on https://huggingface.co/new
     → Suggested name: "bert-base-financial-sentiment"
     → Visibility: Private (share with teammates) or Public
  4. Set HF_REPO_ID below to "<your-hf-username>/bert-base-financial-sentiment"
  5. python upload_model_to_hub.py

What gets uploaded:
  - config.json
  - model.safetensors   (the fine-tuned weights — ~418 MB)
  - trainer_state.json  (training logs / best-step metadata)

After upload, share the repo URL with teammates and set HF_REPO_ID in
download_models.py to the same value.

Usage:
  python upload_model_to_hub.py
"""

import os
from huggingface_hub import HfApi

# ── CONFIGURE THIS ────────────────────────────────────────────────────────────
HF_REPO_ID    = "YOUR_HF_USERNAME/bert-base-financial-sentiment"   # ← change me
BEST_CKPT     = "checkpoint-3440"   # best_global_step from trainer_state.json
# ─────────────────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR  = os.path.join(_HERE, "models", "bert_base_finetuned_merged", BEST_CKPT)

FILES_TO_UPLOAD = [
    "config.json",
    "model.safetensors",
    "trainer_state.json",
]

if __name__ == "__main__":
    if "YOUR_HF_USERNAME" in HF_REPO_ID:
        raise ValueError(
            "Please set HF_REPO_ID to your actual HuggingFace repo ID before running.\n"
            "Example:  HF_REPO_ID = 'chndnchugh/bert-base-financial-sentiment'"
        )

    api = HfApi()
    print(f"Uploading {BEST_CKPT} → {HF_REPO_ID}\n")

    for fname in FILES_TO_UPLOAD:
        local_path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(local_path):
            print(f"  ✗ Not found, skipping: {local_path}")
            continue
        size_mb = os.path.getsize(local_path) / (1024 ** 2)
        print(f"  Uploading {fname} ({size_mb:.0f} MB) ...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=fname,
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print(f"  ✓ {fname}")

    print(f"\n✓ Upload complete → https://huggingface.co/{HF_REPO_ID}")
    print("  Update HF_REPO_ID in download_models.py to this same value.")
