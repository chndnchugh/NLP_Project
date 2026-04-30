# FinBERT Sentiment Analysis Agent

A complete pipeline to fine-tune FinBERT on your financial sentiment dataset
and deploy it as an inference agent.

## Project structure

```
finbert_agent/
├── 1_data_preparation.py     # Load, clean, tokenize, split
├── 2_baseline_evaluation.py  # Zero-shot FinBERT benchmark
├── 3_finetune.py             # Fine-tune on train data
├── 4_evaluate.py             # Test set evaluation + comparison
├── 5_agent.py                # Inference agent (single / batch / document)
├── requirements.txt
└── README.md

data/
└── financial_sentiment.csv   # Your dataset (text, label columns)

models/
└── finbert_finetuned/        # Saved after step 3

results/
├── baseline_confusion.png
├── finetuned_confusion.png
├── baseline_report.txt
├── finetuned_report.txt
└── comparison.json
```

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
# 1. Prepare and tokenize the dataset
python 1_data_preparation.py

# 2. Evaluate stock FinBERT (baseline, zero-shot)
python 2_baseline_evaluation.py

# 3. Fine-tune on your training data
python 3_finetune.py

# 4. Evaluate fine-tuned model and compare to baseline
python 4_evaluate.py

# 5. Use the agent
python 5_agent.py
```

## Dataset format

Your CSV should have at least two columns:

| text                             | label    |
|----------------------------------|----------|
| Revenue grew 20% year-over-year. | positive |
| Margins declined sharply.        | negative |
| Sales were flat.                 | neutral  |

## Key hyperparameters (edit in 3_finetune.py)

| Parameter      | Default | Notes                                    |
|----------------|---------|------------------------------------------|
| LEARNING_RATE  | 2e-5    | Lower = more stable, higher = faster     |
| NUM_EPOCHS     | 5       | Early stopping kicks in if val F1 stalls |
| BATCH_SIZE     | 16      | Increase to 32 if you have GPU VRAM      |
| FREEZE_LAYERS  | 6       | Freeze bottom N BERT layers (0–11)       |
| WARMUP_RATIO   | 0.1     | 10% of steps used for LR warmup          |

## Agent usage

```python
from 5_agent import FinBERTAgent

agent = FinBERTAgent()

# Single text
result = agent.analyze("The company beat earnings expectations by 20%.")
print(result.label)       # "positive"
print(result.confidence)  # 0.9312

# Batch
results = agent.analyze_batch(["text1", "text2", "text3"])

# Long document (aggregated sentence-by-sentence)
result = agent.analyze_document(long_earnings_call_text)
```
