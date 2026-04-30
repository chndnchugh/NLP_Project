"""
Stage 5: Sentiment Analysis Agent
- Wraps the fine-tuned model as a reusable agent
- Supports single text, batch, and multi-sentence documents
- Returns label, confidence, and per-class scores
"""

from __future__ import annotations
import re
import torch
import numpy as np
from dataclasses import dataclass
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR  = "models/finbert_finetuned"
DEVICE     = 0 if torch.cuda.is_available() else -1
BATCH_SIZE = 32
MAX_LENGTH = 512

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class SentimentResult:
    text:       str
    label:      str
    confidence: float
    scores:     dict

    def __repr__(self):
        bars  = {k: "█" * int(v * 20) for k, v in self.scores.items()}
        lines = [
            f"Text      : {self.text[:80]}{'...' if len(self.text) > 80 else ''}",
            f"Label     : {self.label.upper()}  (conf: {self.confidence:.2%})",
        ] + [f"  {k:<10}: {bars[k]:<20} {v:.2%}" for k, v in self.scores.items()]
        return "\n".join(lines)

# ── Agent ─────────────────────────────────────────────────────────────────────
class FinBERTAgent:
    """
    Financial sentiment analysis agent backed by fine-tuned FinBERT.

    Usage:
        agent = FinBERTAgent()
        result  = agent.analyze("Apple beat earnings expectations.")
        results = agent.analyze_batch(["...", "..."])
        doc     = agent.analyze_document(long_text)
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        print(f"Loading model from '{model_dir}' ...")
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._pipeline  = pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            top_k=None,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        print("Agent ready.\n")

    def _parse(self, raw: list, text: str) -> SentimentResult:
        scores = {r["label"]: round(r["score"], 4) for r in raw}
        label  = max(scores, key=scores.get)
        return SentimentResult(
            text=text,
            label=label,
            confidence=scores[label],
            scores=scores,
        )

    def analyze(self, text: str) -> SentimentResult:
        """Analyze a single sentence or short paragraph."""
        raw = self._pipeline(text)[0]
        return self._parse(raw, text)

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze a list of texts efficiently in batches."""
        raw_batch = self._pipeline(texts)
        return [self._parse(raw, text) for raw, text in zip(raw_batch, texts)]

    def analyze_document(self, text: str, chunk_by: str = "sentence") -> SentimentResult:
        """
        Aggregate sentiment over a long document.
        Splits by sentence (default) or by newline.
        Aggregates by averaging softmax probabilities.
        """
        if chunk_by == "sentence":
            chunks = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        else:
            chunks = [ln.strip() for ln in text.splitlines() if ln.strip()]

        if not chunks:
            return self.analyze(text)

        results    = self.analyze_batch(chunks)
        all_scores = {label: [] for label in ID2LABEL.values()}
        for r in results:
            for label, score in r.scores.items():
                all_scores[label].append(score)

        avg_scores = {label: round(float(np.mean(vals)), 4) for label, vals in all_scores.items()}
        best_label = max(avg_scores, key=avg_scores.get)

        return SentimentResult(
            text=text[:120] + "...",
            label=best_label,
            confidence=avg_scores[best_label],
            scores=avg_scores,
        )

# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = FinBERTAgent()

    print("── Single headline ──────────────────────────────────────────────")
    r = agent.analyze("The company reported record profits, beating analyst expectations by 15%.")
    print(r)

    print("\n── Batch ────────────────────────────────────────────────────────")
    headlines = [
        "Revenue declined sharply amid rising costs and weak demand.",
        "The board approved a $500M share buyback programme.",
        "Operating margins remained flat year-over-year.",
    ]
    for result in agent.analyze_batch(headlines):
        print(f"[{result.label.upper():<8}  {result.confidence:.0%}]  {result.text[:70]}")

    print("\n── Document (aggregated) ────────────────────────────────────────")
    doc = (
        "Q3 results were disappointing. Revenue fell 12% year-over-year as "
        "supply chain disruptions persisted. However, the company launched "
        "a cost-cutting initiative expected to save $200M annually. "
        "Analysts remain cautiously optimistic about the recovery trajectory."
    )
    print(agent.analyze_document(doc))
