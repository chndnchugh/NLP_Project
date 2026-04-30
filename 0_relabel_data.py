"""
Stage 0: Re-label financial_sentiment.csv
==========================================
Labeler: Loughran-McDonald (LM) finance dictionary + VADER sentiment

Why this combination?
  - LM dictionary is domain-specific: words like "liability", "default",
    "restatement" are negative in finance but neutral in general language.
    VADER alone would miss these.
  - VADER captures tone, intensifiers, and punctuation cues that pure
    word-list methods miss ("revenues SURGE!" vs "revenues surge").
  - Neither is BERT-based, so there is zero architectural overlap with
    the FinBERT models being trained and evaluated.

Scoring logic (per headline):
  1. Count LM positive / negative words (case-insensitive, exact match).
  2. Get VADER compound score (-1 … +1).
  3. Combine with weighted vote:
       - LM net score  = lm_pos - lm_neg  (finance-domain signal)
       - VADER score   = compound          (tone / syntax signal)
       - combined      = 0.6 * sign(lm_net) + 0.4 * vader_score
         (LM weighted higher because we are in the finance domain)
  4. Label:  combined > threshold  → positive
             combined < -threshold → negative
             else                  → neutral

Output: data/financial_sentiment_relabeled.csv
  - 'Sentiment'          : new label  (positive / negative / neutral)
  - 'Sentiment_original' : original (noisy) label kept for reference
  - 'LM_pos'             : count of LM positive words found
  - 'LM_neg'             : count of LM negative words found
  - 'VADER_compound'     : raw VADER compound score
  - 'Confidence'         : |combined score| as a proxy for confidence
"""

import os
import re
import zipfile
import requests
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/financial_sentiment.csv"
OUTPUT_PATH    = "data/financial_sentiment_relabeled.csv"
LM_CACHE_PATH  = "data/LM_MasterDictionary.csv"
THRESHOLD      = 0.15        # |combined| below this → neutral
                             # Raised from 0.05: only clearly positive/negative
                             # headlines get labelled as such, reducing noisy
                             # borderline labels that were corrupting training.
LM_WEIGHT      = 0.6         # weight for LM signal (rest goes to VADER)

# LM dictionary hosted on GitHub (widely mirrored, no auth required)
LM_URL = (
    "https://raw.githubusercontent.com/jlondonobo/lm-finance/main/data/"
    "LoughranMcDonald_MasterDictionary_2020.csv"
)
LM_URL_FALLBACK = (
    "https://raw.githubusercontent.com/nickderobertis/loughran-mcdonald/"
    "master/lm/LoughranMcDonald_MasterDictionary_2020.csv"
)

# ── VADER setup ────────────────────────────────────────────────────────────────
print("Setting up VADER ...")
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ── LM dictionary ──────────────────────────────────────────────────────────────
def download_lm(url: str) -> bool:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        os.makedirs("data", exist_ok=True)
        with open(LM_CACHE_PATH, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"  Failed ({e})")
        return False

def load_lm_wordlists():
    if not os.path.exists(LM_CACHE_PATH):
        print("Downloading LM Master Dictionary ...")
        if not download_lm(LM_URL):
            print("  Trying fallback URL ...")
            if not download_lm(LM_URL_FALLBACK):
                print("  Both URLs failed — using built-in word lists.")
                return _builtin_lm_words()

    try:
        df = pd.read_csv(LM_CACHE_PATH, low_memory=False)
        # Columns: Word, Negative, Positive, ... (non-zero value = that attribute)
        lm_pos = set(df.loc[df["Positive"] != 0, "Word"].str.upper())
        lm_neg = set(df.loc[df["Negative"] != 0, "Word"].str.upper())
        print(f"Loaded LM dictionary: {len(lm_pos)} positive, {len(lm_neg)} negative words.")
        return lm_pos, lm_neg
    except Exception as e:
        print(f"  Could not parse LM CSV ({e}) — using built-in word lists.")
        return _builtin_lm_words()

def _builtin_lm_words():
    """
    Curated subset of the LM dictionary covering the most frequent
    finance-specific sentiment words. Used only as a fallback.
    """
    positive = {
        "ACHIEVE", "ACHIEVED", "ACHIEVEMENT", "ADVANCE", "ADVANTAGE", "AGREEMENT",
        "APPROVE", "APPROVED", "BENEFIT", "BENEFITED", "BEST", "BOOST", "BOOSTED",
        "BREAKTHROUGH", "CONFIDENT", "CONFIDENCE", "CREATE", "DELIVER", "DELIVERED",
        "EFFECTIVE", "EFFICIENT", "ENHANCE", "ENHANCED", "EXCELLENT", "EXCEED",
        "EXCEEDED", "EXPAND", "EXPANDED", "FAVORABLE", "GAIN", "GAINED", "GROWTH",
        "GREW", "IMPROVE", "IMPROVED", "IMPROVEMENT", "INCREASE", "INCREASED",
        "LEADER", "LEADING", "MILESTONE", "MOMENTUM", "OPPORTUNITIES", "OPPORTUNITY",
        "OPTIMAL", "OUTPERFORM", "OUTSTANDING", "PROFITABLE", "PROFIT", "PROGRESS",
        "RECORD", "REVENUE", "RISE", "ROBUST", "STABLE", "STRONG", "STRENGTH",
        "SUCCESS", "SUCCESSFUL", "SUPERIOR", "SURGE", "SURGED", "SUSTAIN",
        "VALUE", "WIN", "WON",
    }
    negative = {
        "ABANDON", "ADVERSE", "ALLEGATIONS", "BANKRUPT", "BANKRUPTCY", "BREACH",
        "BURDEN", "CANCEL", "CATASTROPHE", "CAUTIOUS", "CEASE", "CLAIM", "COLLAPSE",
        "COLLUSION", "CONCERN", "CONVICTION", "CRISIS", "DAMAGE", "DECLINE",
        "DECLINED", "DEFAULT", "DELAY", "DIFFICULTIES", "DISPUTE", "DISTRESS",
        "DOWNTURN", "EMERGENCY", "FAILURE", "FAILED", "FINE", "FORCED", "FRAUD",
        "GUILTY", "HARM", "IMPAIRMENT", "INADEQUATE", "INABILITY", "INSOLVENCY",
        "INSUFFICIENT", "INVESTIGATION", "LAWSUIT", "LAYOFF", "LIABILITY",
        "LIQUIDATION", "LOSS", "LOSSES", "MANIPULATE", "MISLED", "NONCOMPLIANCE",
        "OBSTACLES", "PENALTY", "POOR", "PROBLEM", "RECESSION", "RESTATEMENT",
        "RISK", "SHORTFALL", "SHUTDOWN", "SLOWDOWN", "STRUGGLING", "SUSPECT",
        "TERMINATE", "UNCERTAINTY", "UNFAVORABLE", "VIOLATION", "WEAKNESS",
        "WORSEN", "WORST", "WRITEOFF", "WRITE-OFF",
    }
    print(f"Using built-in LM words: {len(positive)} positive, {len(negative)} negative.")
    return positive, negative

# ── Tokenise (simple — matches LM convention of uppercase bare words) ──────────
def lm_tokenize(text: str):
    return set(re.sub(r"[^a-zA-Z\s]", "", text).upper().split())

# ── Combined scorer ────────────────────────────────────────────────────────────
def score(text: str, lm_pos: set, lm_neg: set) -> dict:
    words    = lm_tokenize(text)
    n_pos    = len(words & lm_pos)
    n_neg    = len(words & lm_neg)
    lm_net   = n_pos - n_neg

    vader    = sia.polarity_scores(text)["compound"]   # -1 … +1

    # Normalise lm_net to [-1, +1] using tanh so it's on the same scale as VADER
    lm_norm  = float(np.tanh(lm_net))

    combined = LM_WEIGHT * lm_norm + (1 - LM_WEIGHT) * vader

    if combined > THRESHOLD:
        label = "positive"
    elif combined < -THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label":      label,
        "lm_pos":     n_pos,
        "lm_neg":     n_neg,
        "vader":      round(vader, 4),
        "combined":   round(combined, 4),
        "confidence": round(abs(combined), 4),
    }

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    lm_pos, lm_neg = load_lm_wordlists()

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Headline"]).reset_index(drop=True)
    print(f"\nScoring {len(df)} headlines ...")

    results = [score(str(row["Headline"]), lm_pos, lm_neg) for _, row in df.iterrows()]

    df["Sentiment_original"] = df["Sentiment"]
    df["Sentiment"]          = [r["label"]      for r in results]
    df["LM_pos"]             = [r["lm_pos"]     for r in results]
    df["LM_neg"]             = [r["lm_neg"]     for r in results]
    df["VADER_compound"]     = [r["vader"]       for r in results]
    df["Confidence"]         = [r["confidence"]  for r in results]

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== New label distribution ===")
    print(df["Sentiment"].value_counts().to_string())

    print("\n=== Confidence stats ===")
    print(df["Confidence"].describe().round(3).to_string())

    low_conf = (df["Confidence"] < THRESHOLD).sum()
    print(f"\nLow-confidence labels (|score| < {THRESHOLD}): {low_conf} "
          f"({low_conf/len(df):.1%}) — these are genuine neutral calls")

    print("\n=== Sample labelled headlines ===")
    for lbl in ["positive", "negative", "neutral"]:
        sample = df[df["Sentiment"] == lbl].nlargest(3, "Confidence")
        print(f"\n-- {lbl.upper()} --")
        for _, row in sample.iterrows():
            print(f"  [{row['Confidence']:.2f}] {row['Headline']}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved → {OUTPUT_PATH}  ({len(df)} rows)")
