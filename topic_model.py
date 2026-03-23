"""
topic_model.py

runs topic modelling on the collected tweet data

it cleans the tweet text, turns each tweet into tokens,
trains an lda model, and assigns each tweet to its most likely topic

saves output to:
data/topic_features.csv
data/lda_topics.txt
data/topic_distribution.png
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DATA_DIR  = "data"
CSV_IN    = os.path.join(DATA_DIR, "raw_tweets.csv")
FEAT_OUT  = os.path.join(DATA_DIR, "topic_features.csv")
TOPIC_OUT = os.path.join(DATA_DIR, "lda_topics.txt")
VIZ_OUT   = os.path.join(DATA_DIR, "topic_distribution.png")

NUM_TOPICS = 3
PASSES     = 10
MIN_TOKENS = 2

EXTRA_STOPWORDS = {
    "aijobs", "ai", "jobs", "job", "work", "people", "keep",
    "talking", "seeing", "opinions", "getting", "topic", "next",
    "big", "might", "trend", "everywhere", "today", "sample"
}

# check that the input file has usable tweet text
# also warns if too many text values are missing, since that can weaken the topic results
def _validate_input(df: pd.DataFrame) -> None:
    if "text" not in df.columns:
        raise KeyError(
            "'text' column not found in raw_tweets.csv. "
            "Re-run collector.py to regenerate the dataset."
        )
    if df.empty:
        raise ValueError("raw_tweets.csv is empty — no tweets to model.")

    null_frac = df["text"].isna().mean()
    if null_frac > 0.5:
        log.warning(
            "%.0f%% of tweet texts are null — topic results may be unreliable.",
            null_frac * 100
        )

# clean one tweet and return useful tokens for lda
# this removes common noise like links, mentions, symbols and short filler words
def preprocess(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    stop   = STOPWORDS | EXTRA_STOPWORDS
    tokens = [t for t in text.split() if t not in stop and len(t) > 2]
    return tokens

# lower the topic count if there are not enough valid tweets
# this helps stop lda from failing on a very small dataset
def _adjust_topic_count(n_valid: int, n_topics: int) -> int:
    min_needed = n_topics * 2
    if n_valid < min_needed:
        adjusted = max(2, n_valid // 3)
        log.warning(
            "Only %d valid tweets available for %d topics (need %d). "
            "Reducing to %d topics.",
            n_valid, n_topics, min_needed, adjusted
        )
        return adjusted
    return n_topics

# run the full topic modelling step
# loads tweet text, preprocesses it, trains the lda model, saves topic outputs and plots the topic distribution
def run_topic_modelling(csv_path: str = CSV_IN,
                        n_topics: int = NUM_TOPICS):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Run collector.py first."
        )
    # load tweet data saved from the collection step
    df = pd.read_csv(csv_path)
    log.info("Loaded %d tweets from '%s'.", len(df), csv_path)
    _validate_input(df)
    # preprocess text and keep only tweets with enough usuable words
    df["tokens"] = df["text"].apply(preprocess)
    valid = df[df["tokens"].apply(len) >= MIN_TOKENS].copy()

    n_dropped = len(df) - len(valid)
    if n_dropped > 0:
        log.info(
            "Dropped %d tweet(s) with fewer than %d tokens after preprocessing.",
            n_dropped, MIN_TOKENS
        )

    if valid.empty:
        raise ValueError(
            "No tweets survived preprocessing. The dataset may be too small "
            "or every tweet was filtered out as stopwords."
        )

    n_topics = _adjust_topic_count(len(valid), n_topics)

    # turn token lists into bag-of-words format for lda
    # build vocabulary from the cleaned tweets
    dictionary = corpora.Dictionary(valid["tokens"])
    before = len(dictionary)
    dictionary.filter_extremes(no_below=1, no_above=0.95)
    after  = len(dictionary)
    log.info("Dictionary: %d → %d tokens after extreme filtering.", before, after)

    if after == 0:
        raise ValueError(
            "Dictionary is empty after filtering. Try lowering the no_above "
            "threshold or using a larger dataset."
        )
    # convert each tweet into bow using the shared dictionary
    corpus = [dictionary.doc2bow(tokens) for tokens in valid["tokens"]]

    log.info("Training LDA: %d topics, %d passes, corpus size %d.",
             n_topics, PASSES, len(corpus))

    try:
        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            passes=PASSES,
            random_state=42,
            alpha="auto"
        )
    except Exception as exc:
        raise RuntimeError(
            f"LDA training failed: {exc}. "
            "Check that Gensim is installed and the corpus is non-empty."
        ) from exc

    # store full topic probabilities for each tweet
    topic_cols = [f"topic_{i}_prob" for i in range(n_topics)]
    topic_rows = []
    for bow in corpus:
        dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
        row  = [dist.get(i, 0.0) for i in range(n_topics)]
        topic_rows.append(row)

    topic_df = pd.DataFrame(topic_rows, columns=topic_cols, index=valid.index)

    # pick the topic with the highest probabilty for each tweet
    topic_df["dominant_topic"] = (
        topic_df[topic_cols]
        .idxmax(axis=1)
        .str.replace("topic_", "")
        .str.replace("_prob", "")
        .astype(int)
    )
    # combine original tweet info with the topic results
    result = valid[["tweet_id", "created_at", "text"]].join(topic_df)
    os.makedirs(DATA_DIR, exist_ok=True)
    result.to_csv(FEAT_OUT, index=False)
    log.info("Topic features saved → %s  (%d rows)", FEAT_OUT, len(result))

    # save a readable summary of the main words in each topic
    lines = [
        f"LDA Topics ({n_topics} topics, trained on {len(valid)} tweets)",
        "=" * 60,
    ]
    for idx in range(n_topics):
        words    = lda.show_topic(idx, topn=8)
        word_str = ", ".join(f"{w}({p:.2f})" for w, p in words)
        lines.append(f"\nTopic {idx}: {word_str}")
        log.info("Topic %d: %s", idx, word_str)

    topic_summary = "\n".join(lines)
    with open(TOPIC_OUT, "w", encoding="utf-8") as f:
        f.write(topic_summary)
    log.info("Topic descriptions saved → %s", TOPIC_OUT)

    # plot how many tweets fall under each dominant topic
    dist_counts = topic_df["dominant_topic"].value_counts().sort_index()

    if dist_counts.empty:
        log.warning("No dominant topic counts to plot — skipping chart.")
    else:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars    = ax.bar(
            [f"Topic {i}" for i in dist_counts.index],
            dist_counts.values,
            color=["#4472C4", "#70AD47", "#ED7D31", "#9E63B5", "#E16182"][:n_topics]
        )
        ax.set_title("Tweet Count per Dominant Topic", fontsize=12)
        ax.set_xlabel("Topic")
        ax.set_ylabel("Number of Tweets")
        for bar, val in zip(bars, dist_counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=9
            )
        plt.tight_layout()
        plt.savefig(VIZ_OUT, dpi=120, bbox_inches="tight")
        plt.close()
        log.info("Distribution plot saved → %s", VIZ_OUT)

    return result, lda


if __name__ == "__main__":
    result, lda = run_topic_modelling()
    log.info("Sample output:\n%s",
             result[["tweet_id", "dominant_topic"] +
                    [c for c in result.columns if "prob" in c]].head().to_string())
