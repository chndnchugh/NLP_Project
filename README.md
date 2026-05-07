# Financial News Sentiment Analysis

A pipeline for financial news sentiment classification using transformer models.
Compares three approaches — BERT zero-shot, BERT fine-tuned, and FinBERT zero-shot —
on financial news headlines, with support for evaluation on both a held-out test set
and freshly collected unseen data.

---

## Project Structure

```
NLP Project/
│
├── NLP/                              ← Core pipeline scripts
│   ├── data_preparation.py           # Load, clean, tokenize & split financial_sentiment.csv
│   ├── baseline_evaluation.py        # Zero-shot FinBERT benchmark on tokenized test set
│   └── evaluate.py                   # 3-way model comparison (supports --collected flag)
│
├── compare_3way.py                   ← Full 3-way comparison with charts (supports --collected)
├── finetune_bert_merged.py           ← Fine-tune bert-base-uncased on financial_sentiment.csv
├── download_models.py                ← Pre-download HuggingFace models for offline use
├── requirements.txt                  ← Python dependencies
│
├── data/
│   ├── financial_sentiment.csv       # Labelled training dataset (text | sentiment)
│   ├── phrasebank_test.csv           # Held-out PhraseBank test set (compare_3way default)
│   ├── Collected_Test_Data.csv       # Freshly collected unseen data (--collected flag)
│   ├── tokenized_dataset/            # Tokenized splits for NLP/ pipeline
│   └── tokenized_dataset_merged/     # Tokenized splits for finetune_bert_merged.py
│
├── models/
│   └── bert_base_finetuned_merged/   # Fine-tuned BERT model (ready for inference)
│
├── results/                          ← Evaluation outputs (auto-created on run)
│   ├── metrics_comparison.png
│   ├── per_class_f1.png
│   ├── confusion_matrices.png
│   ├── improvement_over_bert.png
│   ├── comparison.json
│   ├── collected_test/               # Results when run with --collected
│   └── comparison_3way/              # Results from compare_3way.py
│       └── comparison_3way_collected/
│
└── logs/                             ← Training logs
```

---

## Models Compared

| Model | Type | Description |
|---|---|---|
| BERT (Zero-shot) | `bert-base-uncased` | Zero-shot via fill-mask — no training |
| BERT (Fine-tuned) | `bert-base-uncased` | Fine-tuned on `financial_sentiment.csv` |
| FinBERT (Zero-shot) | `ProsusAI/finbert` | Finance-domain pre-trained, no fine-tuning |

---

## Setup

```bash
cd "NLP Project"
source .ven/bin/activate          # activate the virtual environment
# or install fresh:
pip install -r requirements.txt
```

For GPU support, install PyTorch separately first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Running the Pipeline

All commands are run from the project root (`NLP Project/`).

### 1 — Prepare & tokenize the dataset
```bash
python NLP/data_preparation.py
```
Loads `data/financial_sentiment.csv`, splits 80/10/10, tokenizes with BERT tokenizer,
saves to `data/tokenized_dataset/`.

### 2 — Baseline evaluation (FinBERT zero-shot)
```bash
python NLP/baseline_evaluation.py
```
Runs zero-shot FinBERT on the held-out test split. Saves predictions and confusion matrix to `results/`.

### 3 — Fine-tune BERT
```bash
python finetune_bert_merged.py
```
Fine-tunes `bert-base-uncased` on the training set with class-weighted loss and early stopping.
Saves best checkpoint to `models/bert_base_finetuned_merged/`.

### 4 — 3-Way evaluation

**On the original held-out test set:**
```bash
python NLP/evaluate.py
```

**On the collected unseen data:**
```bash
python NLP/evaluate.py --collected
```

Compares BERT zero-shot vs BERT fine-tuned vs FinBERT zero-shot.
Saves four charts + classification reports + `comparison.json` to `results/`
(or `results/collected_test/` when using `--collected`).

### 5 — Full 3-way comparison with PhraseBank test set

```bash
python compare_3way.py                # PhraseBank held-out test set
python compare_3way.py --collected    # Collected unseen data
```

Same three-model comparison evaluated on the Financial PhraseBank test set by default.
Results saved to `results/comparison_3way/` or `results/comparison_3way_collected/`.

---

## Output Charts

Each evaluation run produces four charts:

| File | Description |
|---|---|
| `metrics_comparison.png` | Grouped bar chart — Accuracy, F1 Macro, F1 Weighted |
| `per_class_f1.png` | Per-class F1 for Positive / Negative / Neutral |
| `confusion_matrices.png` | 1×3 confusion matrix grid |
| `improvement_over_bert.png` | Delta vs BERT zero-shot baseline |

---

## Fine-Tuning Hyperparameters

Edit in `finetune_bert_merged.py`:

| Parameter | Value | Notes |
|---|---|---|
| `LEARNING_RATE` | `1e-5` | Conservative — generic BERT needs gentle adaptation |
| `NUM_EPOCHS` | `10` | Early stopping kicks in before this |
| `BATCH_SIZE` | `8` | With `GRAD_ACCUM=2` → effective batch of 16 |
| `EARLY_STOP_PAT` | `3` | Stops if val F1 doesn't improve for 3 epochs |
| `WEIGHT_DECAY` | `0.01` | L2 regularisation |
| `WARMUP_RATIO` | `0.1` | 10% of steps used for LR warmup |

---

## Dataset Format

`data/financial_sentiment.csv` — columns used by the pipeline:

| text | sentiment |
|---|---|
| Sebi clears 4 IPOs amid rising demand. | positive |
| IT rout drags Nifty to third straight loss. | negative |
| Sales were broadly in line with expectations. | neutral |
