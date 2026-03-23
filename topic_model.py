"""
topic_model.py
LDA Topic Modelling Module

Reads: data/raw_tweets.csv
Writes: data/topic_features.csv  (per-tweet dominant topic + distribution)
        data/lda_topics.txt       (human-readable topic summary)
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for all environments)

# Gensim LDA
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
CSV_IN     = os.path.join(DATA_DIR, "raw_tweets.csv")
FEAT_OUT   = os.path.join(DATA_DIR, "topic_features.csv")
TOPIC_OUT  = os.path.join(DATA_DIR, "lda_topics.txt")
VIZ_OUT    = os.path.join(DATA_DIR, "topic_distribution.png")

NUM_TOPICS = 3        # increase for larger, more diverse datasets
PASSES     = 10       # LDA training passes (more = better, slower)
MIN_TOKENS = 2        # skip tweets shorter than this after cleaning

EXTRA_STOPWORDS = {
    "aijobs", "ai", "jobs", "job", "work", "people", "keep",
    "talking", "seeing", "opinions", "getting", "topic", "next",
    "big", "might", "trend", "everywhere", "today", "sample"
}

# ── Text helpers ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> list[str]:
    """Lower, remove noise, tokenise, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    stop   = STOPWORDS | EXTRA_STOPWORDS
    tokens = [t for t in tokens if t not in stop and len(t) > 2]
    return tokens


# ── Main ─────────────────────────────────────────────────────────────────────
def run_topic_modelling(csv_path: str = CSV_IN,
                        n_topics: int = NUM_TOPICS) -> pd.DataFrame:

    # 1. Load data
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}. Run collector.py first.")

    df = pd.read_csv(csv_path)
    print(f"[LDA] Loaded {len(df)} tweets")

    # 2. Tokenise
    df["tokens"] = df["text"].apply(preprocess)
    valid = df[df["tokens"].apply(len) >= MIN_TOKENS].copy()
    print(f"[LDA] {len(valid)} tweets after filtering (min {MIN_TOKENS} tokens)")

    if len(valid) < n_topics * 2:
        print("[LDA] Warning: very small dataset — results may not be meaningful")
        n_topics = max(2, len(valid) // 3)

    # 3. Build corpus
    dictionary = corpora.Dictionary(valid["tokens"])
    dictionary.filter_extremes(no_below=1, no_above=0.95)
    corpus = [dictionary.doc2bow(tokens) for tokens in valid["tokens"]]

    # 4. Train LDA
    print(f"[LDA] Training with {n_topics} topics, {PASSES} passes…")
    lda = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=n_topics,
        passes=PASSES,
        random_state=42,
        alpha="auto"
    )

    # 5. Extract per-document topic distribution
    topic_cols = [f"topic_{i}_prob" for i in range(n_topics)]
    topic_rows = []
    for bow in corpus:
        dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
        row  = [dist.get(i, 0.0) for i in range(n_topics)]
        topic_rows.append(row)

    topic_df = pd.DataFrame(topic_rows, columns=topic_cols, index=valid.index)
    topic_df["dominant_topic"] = topic_df[topic_cols].idxmax(axis=1).str.replace("topic_", "").str.replace("_prob", "").astype(int)

    # 6. Merge back
    result = valid[["tweet_id", "created_at", "text"]].join(topic_df)
    os.makedirs(DATA_DIR, exist_ok=True)
    result.to_csv(FEAT_OUT, index=False)
    print(f"[LDA] Saved topic features → {FEAT_OUT}")

    # 7. Print & save topic descriptions
    lines = [f"LDA Topics ({n_topics} topics, trained on {len(valid)} tweets)\n", "="*60]
    for idx in range(n_topics):
        words = lda.show_topic(idx, topn=8)
        word_str = ", ".join(f"{w}({p:.2f})" for w, p in words)
        lines.append(f"\nTopic {idx}: {word_str}")
    topic_summary = "\n".join(lines)
    print(topic_summary)

    with open(TOPIC_OUT, "w", encoding="utf-8") as f:
        f.write(topic_summary)
    print(f"[LDA] Saved topic descriptions → {TOPIC_OUT}")

    # 8. Plot topic distribution
    dist_counts = topic_df["dominant_topic"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar([f"Topic {i}" for i in dist_counts.index],
                  dist_counts.values,
                  color=["#4472C4", "#70AD47", "#ED7D31"][:n_topics])
    ax.set_title("Tweet Count per Dominant Topic", fontsize=12)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of Tweets")
    for bar, val in zip(bars, dist_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(VIZ_OUT, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[LDA] Saved distribution plot → {VIZ_OUT}")

    return result, lda


if __name__ == "__main__":
    result, lda = run_topic_modelling()
    print("\n[LDA] Sample output:")
    print(result[["tweet_id", "dominant_topic"] +
                 [c for c in result.columns if "prob" in c]].head())
