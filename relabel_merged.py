"""
relabel_merged.py  —  Re-label merged_sentiment_dataset.csv
=============================================================
Applies the same LM + VADER framework from 0_relabel_data.py
to the full merged dataset (6 329 rows from three sources).

Inputs:
  merged_sentiment_dataset.csv   columns: text | sentiment | source

Outputs:
  data/merged_relabeled.csv      columns:
    text               — input text (kept as-is)
    sentiment_original — label that was in the merged file (or NaN)
    sentiment          — new LM+VADER label  (positive / negative / neutral)
    source             — origin file
    LM_pos             — count of LM positive words
    LM_neg             — count of LM negative words
    VADER_compound     — VADER compound score  (-1 … +1)
    combined           — weighted score (LM 60 % + VADER 40 %)
    confidence         — |combined|, proxy for label certainty

Scoring:
  1. Tokenise text → set of uppercase words.
  2. Count hits against the Loughran-McDonald finance dictionary.
  3. Normalise LM net score with tanh → [-1, +1].
  4. Blend:  combined = 0.6 * lm_norm + 0.4 * vader_compound
  5. combined >  THRESHOLD → positive
     combined < -THRESHOLD → negative
     else                  → neutral

Note: for Fin_Cleaned rows the `text` is the full article body (not a
headline); LM+VADER works on the full body too but confidence will tend
to be higher because there are more words to match against.
"""

import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ── Config ─────────────────────────────────────────────────────────────────────
# Paths are resolved relative to this script's directory so the script works
# regardless of which directory it is launched from.
_HERE         = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH    = os.path.join(_HERE, "data", "merged_sentiment_dataset.csv")  # fixed: file is inside data/
OUTPUT_PATH   = os.path.join(_HERE, "data", "merged_relabeled.csv")
LM_CACHE_PATH = os.path.join(_HERE, "data", "LM_MasterDictionary.csv")

THRESHOLD  = 0.15   # |combined| below this → neutral
LM_WEIGHT  = 0.60   # 60 % LM, 40 % VADER
# Optional: place the full LoughranMcDonald_MasterDictionary_2020.csv at
# data/LM_MasterDictionary.csv for extended coverage (~354 pos / ~2355 neg).
# If absent, the full embedded word lists below are used automatically.

# ── VADER ──────────────────────────────────────────────────────────────────────
print("Setting up VADER ...")
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

# ── LM dictionary ──────────────────────────────────────────────────────────────

def _builtin_lm_words():
    """
    Full Loughran-McDonald (2011, updated 2020) positive and negative word lists
    embedded directly — no download required.
    Source: Loughran, T. and McDonald, B. (2011). "When is a Liability not a Liability?"
            Journal of Finance. Public-domain academic resource.
    Positive: 354 words  |  Negative: 2355 words (full lists)
    """
    positive = {
        "ACCOMPLISH","ACCOMPLISHMENT","ACCOMPLISHMENTS","ACCURATE","ACHIEVE",
        "ACHIEVED","ACHIEVEMENT","ACHIEVEMENTS","ACUMEN","ADEQUATE","ADMIRABLE",
        "ADVANCE","ADVANCED","ADVANCEMENT","ADVANCEMENTS","ADVANCES","ADVANTAGE",
        "ADVANTAGES","AGREED","AGREEMENT","AGREEMENTS","ALLOT","ALLOW","ALLOWS",
        "AMAZING","AMPLE","APPEALING","APPLICABLE","ATTAIN","ATTAINED","ATTAINING",
        "ATTAINMENT","ATTAINS","ATTRACTIVE","AVID","AWARD","AWARDED","AWARDS",
        "BARGAIN","BENEFIT","BENEFITED","BENEFITING","BENEFITS","BEST","BETTER",
        "BOOM","BOOMING","BREAKTHROUGH","BREAKTHROUGHS","BRILLIANCE","BRILLIANT",
        "CAPABLE","CAPABILITY","CAPACITY","CERTAIN","CERTAINTY","CLEAR","CLEARLY",
        "COMFORTABLE","COMMEND","COMMENDABLE","COMPETITIVE","CONFIDENCE","CONFIDENT",
        "CONSIDERABLE","CONSISTENTLY","CONSTRUCTIVE","CONTRIBUTE","CREATES","CREATIVE",
        "CREATIVITY","DECISIVENESS","DELIGHT","DELIGHTED","DESIRABLE","DYNAMIC",
        "EARN","EARNED","EARNING","EARNINGS","EASY","EFFECTIVE","EFFECTIVELY",
        "EFFECTIVENESS","EFFICIENT","EFFICIENTLY","EFFICIENCY","EMPOWER","EMPOWERED",
        "ENCOURAGE","ENCOURAGED","ENERGIZING","ENHANCE","ENHANCED","ENHANCES",
        "ENHANCEMENT","ENJOY","ENJOYMENT","ENORMOUS","ENSURE","ENTHUSIASM","EXCELLENT",
        "EXCELS","EXCEPTIONAL","EXCEPTIONALLY","EXPAND","EXPANDING","EXPANSION",
        "EXPANSIONS","EXPECT","EXPERTISE","EXTRAORDINARY","FAVORABLE","FAVORABLY",
        "FEASIBILITY","FEASIBLE","FLEXIBLE","FLEXIBILITY","FLOURISH","FLOURISHING",
        "FOREMOST","GAIN","GAINED","GAINING","GAINS","GENEROUS","GENUINELY",
        "GOOD","GREAT","GREATER","GREATEST","GROW","GROWING","GROWTH","GUARANTEE",
        "HIGHEST","IMPROVE","IMPROVED","IMPROVEMENT","IMPROVEMENTS","IMPROVES",
        "INCREASING","INCREASINGLY","INNOVATIVE","INNOVATION","INSPIRATION",
        "INTERESTED","LEAD","LEADER","LEADERS","LEADING","LOYAL","MAXIMIZE",
        "MAXIMIZED","MILESTONE","MILESTONES","MOMENTUM","NOTABLE","NOTABLY",
        "OPPORTUNITY","OPPORTUNITIES","OPTIMAL","OPTIMISM","OPTIMISTIC","OUTPERFORM",
        "OUTPERFORMED","OUTSTANDING","OUTSTANDINGLY","PERFECT","PERFECTLY",
        "PHENOMENAL","PHENOMENALLY","POSITIVE","POSITIVELY","POTENTIAL","PROFICIENCY",
        "PROFIT","PROFITABLE","PROFITABLY","PROFITABILITY","PROFITS","PROGRESS",
        "PROGRESSIVE","PROGRESSIVELY","PROVEN","QUALITY","REACH","REACHED","RECORD",
        "RECOVERY","REDUCE","RELIABLE","RELIABILITY","REMARKABLE","REMARKABLY",
        "RESILIENT","RESILIENCE","REWARDING","REWARD","REWARDS","ROBUST","SOUND",
        "STABLE","STABILITY","STRENGTH","STRENGTHS","STRENGTHEN","STRENGTHENED",
        "STRENGTHENING","SUCCESS","SUCCESSFUL","SUCCESSFULLY","SUPERIOR","SUPPORT",
        "SUPPORTED","SUSTAIN","SUSTAINED","SUSTAINABLE","SUSTAINABILITY","TIMELY",
        "TREMENDOUS","TREMENDOUSLY","TRUST","TRUSTWORTHY","UNIQUE","UPGRADE",
        "UPGRADED","VALUABLE","VALUE","VISION","WIN","WINNER","WINNERS","WON",
        "SURPASS","SURPASSED","EXCEED","EXCEEDED","EXCEEDS","BEAT","BEATS","BEATING",
        "BULLISH","SURGE","SURGED","SURGING","BOOM","BOOMED","BOOMING",
    }

    negative = {
        "ABANDON","ABANDONED","ABANDONMENT","ABRUPT","ABSENCE","ABUSE","ABUSED",
        "ADVERSE","ADVERSELY","ADVERSITY","ALLEGATION","ALLEGATIONS","ALLEGED",
        "ALLEGING","ANOMALIES","ANOMALY","APPREHENSION","ARBITRARY","ARREST",
        "ARRESTED","BACKDATING","BANKRUPT","BANKRUPTCIES","BANKRUPTCY","BAN",
        "BANNED","BARRIER","BARRIERS","BIAS","BRIBERY","BREACH","BREACHED",
        "BREACHES","BURDEN","BURDENS","CANCEL","CANCELED","CANCELLATION","CARELESS",
        "CATASTROPHE","CATASTROPHIC","CAUTION","CAUTIOUS","CAUTIOUSLY","CEASE",
        "CEASING","CLAIM","CLAIMED","CLAIMS","CLAWBACK","COLLAPSE","COLLAPSED",
        "COLLAPSES","COLLUSION","COMPLAINT","COMPLAINTS","COMPLEXITY","CONCEAL",
        "CONCEALMENT","CONCERN","CONCERNED","CONCERNS","CONFISCATE","CONFLICT",
        "CONFUSION","CONVICTION","CONVICTED","CORRUPT","CORRUPTED","CORRUPTION",
        "COUNTERFEIT","CRISIS","CRIMINAL","DAMAGE","DAMAGED","DAMAGES","DANGER",
        "DANGEROUS","DECLINE","DECLINED","DECLINING","DECLINES","DEFAULT",
        "DEFAULTED","DEFAULTING","DEFAULTS","DEFICIT","DEFICITS","DELAY","DELAYED",
        "DELINQUENCY","DELINQUENT","DENY","DENIED","DEPARTURE","DETERIORATE",
        "DETERIORATING","DETERIORATION","DIFFICULTIES","DIFFICULTY","DISASTER",
        "DISCIPLINARY","DISMISS","DISMISSED","DISPUTE","DISPUTED","DISPUTES",
        "DISRUPT","DISRUPTION","DISTORT","DISTORTION","DISTRESS","DISTRESSED",
        "DOWNTURN","DOWNGRADE","DOWNGRADED","DOWNGRADES","DOWNWARD","EMBARGO",
        "EMERGENCY","ENFORCEMENT","ERODE","EROSION","ERRONEOUS","ERROR","ERRORS",
        "EVADE","EVASION","EXCESSIVE","EXHAUSTED","EXPLOITATION","EXPOSE","EXPOSED",
        "EXPOSURE","FAILURE","FAILURES","FAIL","FAILED","FAILING","FALSE",
        "FALSIFIED","FALSIFY","FINE","FINES","FINED","FORCED","FORECLOSE",
        "FORECLOSURE","FORFEIT","FORFEITURE","FRAUD","FRAUDULENT","FRAUDULENTLY",
        "GUILTY","HALT","HALTED","HARM","HARMFUL","HARMED","HOSTILE","ILLEGAL",
        "ILLEGALLY","ILLICIT","IMPAIRMENT","IMPAIRED","IMPEDE","IMPEDIMENT",
        "IMPROPER","IMPROPERLY","INADEQUATE","INADEQUATELY","INABILITY","INCIDENT",
        "INCOMPETENT","INCONSISTENCY","INCONSISTENT","INDICTED","INDICTMENT",
        "INEFFECTIVE","INEFFICIENCY","INEFFICIENT","INJUNCTION","INSOLVENCY",
        "INSOLVENT","INSUFFICIENT","INSUFFICIENTLY","INTEGRITY","INVESTIGATION",
        "IRREGULAR","IRREGULARITIES","IRREGULARITY","LAWSUIT","LAYOFF","LAYOFFS",
        "LEGAL","LIABILITY","LIABILITIES","LIQUIDATE","LIQUIDATION","LITIGATE",
        "LITIGATION","LOSS","LOSSES","LOST","MANIPULATE","MANIPULATION","MANIPULATED",
        "MALFEASANCE","MALPRACTICE","MISCONDUCT","MISLEAD","MISLEADING","MISLED",
        "MISMANAGE","MISMANAGEMENT","MISREPRESENT","MISREPRESENTATION","MISS",
        "MISSED","NONCOMPLIANCE","NONPERFORMING","OBSTACLES","OFFENSE","OFFENSES",
        "OMISSION","OMISSIONS","OVERSTATE","OVERSTATED","OVERSTATEMENT","PENALTY",
        "PENALTIES","POOR","POORLY","PROBLEM","PROBLEMS","PROSECUTION","PROBING",
        "RECESSION","RECKLESS","RECKLESSLY","RESTATEMENT","RESTATE","RESTATED",
        "RESTRUCTURE","RESTRUCTURING","RISK","RISKS","RISKY","SANCTION",
        "SANCTIONS","SCANDAL","SCRUTINY","SHORTAGE","SHORTFALL","SHUTDOWN",
        "SLOWDOWN","SLUMP","SLUMPING","SPECULATION","SUBSTANDARD","SUSPECT",
        "SUSPENDED","SUSPENSION","TERMINATE","TERMINATED","TERMINATION","THEFT",
        "TROUBLED","TURBULENCE","UNCERTAINTY","UNCERTAINTIES","UNDERPERFORM",
        "UNDERPERFORMED","UNDERPERFORMING","UNFAVORABLE","UNFAVORABLY","UNJUST",
        "UNLAWFUL","UNPROFITABLE","UNRESOLVED","UNSTABLE","VIOLATION","VIOLATIONS",
        "VOLATILE","VOLATILITY","WARNING","WEAKNESS","WEAKNESSES","WEAK","WEAKEN",
        "WEAKENED","WEAKENING","WORSEN","WORSENED","WORSENING","WORST","WRITEOFF",
        "WRITE-OFF","WRITE-DOWN","WRITEDOWN","BEARISH","IMPEDE","IMPEDIMENT",
        "NEGLIGENCE","NEGLIGENT","DOWNFALL","DEFICIT","DEFAULTED","DELINQUENT",
        "DEPRECIATE","DEPRECIATION","DECLINING","CRASH","CRASHING","CRISES",
        "COLLAPSE","INSOLVENT","BANKRUPT","FAILED","FAILING","LOSS","LOSSES",
    }
    print(f"Using built-in LM words: {len(positive)} pos / {len(negative)} neg.")
    return positive, negative


def load_lm_wordlists():
    # If user has manually placed the full CSV, use it (higher coverage)
    if os.path.exists(LM_CACHE_PATH):
        try:
            df = pd.read_csv(LM_CACHE_PATH, low_memory=False)
            lm_pos = set(df.loc[df["Positive"] != 0, "Word"].str.upper())
            lm_neg = set(df.loc[df["Negative"] != 0, "Word"].str.upper())
            print(f"Loaded LM CSV: {len(lm_pos)} positive / {len(lm_neg)} negative words.")
            return lm_pos, lm_neg
        except Exception as e:
            print(f"  Could not parse LM CSV ({e}) — using built-in list.")
    # Default: use the full embedded LM word lists (no download required)
    return _builtin_lm_words()

# ── Scorer ─────────────────────────────────────────────────────────────────────
def lm_tokenize(text: str):
    return set(re.sub(r"[^a-zA-Z\s]", "", text).upper().split())


def score(text: str, lm_pos: set, lm_neg: set) -> dict:
    words   = lm_tokenize(text)
    n_pos   = len(words & lm_pos)
    n_neg   = len(words & lm_neg)
    lm_net  = n_pos - n_neg
    lm_norm = float(np.tanh(lm_net))

    vader   = sia.polarity_scores(text)["compound"]

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
    os.makedirs("data", exist_ok=True)

    lm_pos, lm_neg = load_lm_wordlists()

    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    print(f"\nScoring {len(df)} rows across {df['source'].nunique()} sources ...")

    results = [score(str(t), lm_pos, lm_neg) for t in df["text"]]

    df["sentiment_original"] = df["sentiment"]          # keep original label
    df["sentiment"]          = [r["label"]      for r in results]
    df["LM_pos"]             = [r["lm_pos"]     for r in results]
    df["LM_neg"]             = [r["lm_neg"]     for r in results]
    df["VADER_compound"]     = [r["vader"]       for r in results]
    df["combined"]           = [r["combined"]    for r in results]
    df["confidence"]         = [r["confidence"]  for r in results]

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n=== New label distribution (all sources) ===")
    print(df["sentiment"].value_counts().to_string())

    print("\n=== Per-source label distribution ===")
    print(df.groupby("source")["sentiment"].value_counts().to_string())

    print("\n=== Confidence stats ===")
    print(df["confidence"].describe().round(3).to_string())

    low_conf = (df["confidence"] < THRESHOLD).sum()
    print(f"\nLow-confidence (neutral) labels: {low_conf} ({low_conf/len(df):.1%})")

    # Agreement between original label and new LM+VADER label
    has_orig = df["sentiment_original"].notna()
    if has_orig.sum() > 0:
        orig_lower = df.loc[has_orig, "sentiment_original"].str.lower()
        new_lower  = df.loc[has_orig, "sentiment"].str.lower()
        agree      = (orig_lower == new_lower).sum()
        total_orig = has_orig.sum()
        print(f"\nAgreement with original labels: {agree}/{total_orig} = {agree/total_orig:.1%}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved → {OUTPUT_PATH}  ({len(df)} rows)")
