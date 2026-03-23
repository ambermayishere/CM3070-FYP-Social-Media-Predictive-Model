"""
collector.py
Feature Prototype – Data Collection Module

This script collects tweets for a given hashtag using:
1) Twitter API v2 via Tweepy (if available)
2) A fallback sample dataset (if API access fails)

Outputs:
- data/raw_tweets.csv
- data/raw_tweets.json
"""

import os
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd

try:
    import tweepy
except ImportError:
    tweepy = None


# =========================
# Configuration
# =========================
OUTPUT_DIR = "data"
CSV_FILE = os.path.join(OUTPUT_DIR, "raw_tweets.csv")
JSON_FILE = os.path.join(OUTPUT_DIR, "raw_tweets.json")


@dataclass
class Tweet:
    tweet_id: str
    created_at: str
    text: str
    author_id: str
    username: Optional[str]
    retweet_count: int
    reply_count: int
    like_count: int
    quote_count: int


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Mode 1: Twitter API
# =========================
def collect_from_twitter(hashtag: str, max_results: int = 50) -> List[Tweet]:
    if tweepy is None:
        raise RuntimeError("Tweepy not installed.")

    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise RuntimeError("TWITTER_BEARER_TOKEN not set.")

    client = tweepy.Client(bearer_token=bearer_token)

    if not hashtag.startswith("#"):
        hashtag = "#" + hashtag

    response = client.search_recent_tweets(
        query=hashtag,
        max_results=min(max_results, 100),
        tweet_fields=["created_at", "public_metrics", "author_id"],
        expansions=["author_id"],
        user_fields=["username"]
    )

    tweets = []

    user_lookup = {}
    if response.includes and "users" in response.includes:
        for user in response.includes["users"]:
            user_lookup[str(user.id)] = user.username

    if response.data:
        for t in response.data:
            metrics = t.public_metrics or {}
            tweets.append(
                Tweet(
                    tweet_id=str(t.id),
                    created_at=str(t.created_at),
                    text=t.text,
                    author_id=str(t.author_id),
                    username=user_lookup.get(str(t.author_id)),
                    retweet_count=metrics.get("retweet_count", 0),
                    reply_count=metrics.get("reply_count", 0),
                    like_count=metrics.get("like_count", 0),
                    quote_count=metrics.get("quote_count", 0),
                )
            )

    return tweets


# =========================
# Mode 2: Fallback Dataset
# =========================
def generate_fallback_data(hashtag: str, n: int = 50) -> List[Tweet]:
    if not hashtag.startswith("#"):
        hashtag = "#" + hashtag

    base_texts = [
        f"{hashtag} This topic is gaining attention fast.",
        f"People keep talking about {hashtag}.",
        f"{hashtag} opinions are getting intense online.",
        f"{hashtag} might be the next big trend.",
        f"Seeing {hashtag} everywhere today.",
    ]

    tweets = []
    now = pd.Timestamp.utcnow()

    for i in range(n):
        tweets.append(
            Tweet(
                tweet_id=f"sample_{i+1}",
                created_at=(now - pd.Timedelta(minutes=i * 5)).isoformat(),
                text=base_texts[i % len(base_texts)],
                author_id=f"user_{i % 10}",
                username=f"user_{i % 10}",
                retweet_count=(i * 2) % 20,
                reply_count=(i * 3) % 15,
                like_count=(i * 5) % 50,
                quote_count=(i * 1) % 5,
            )
        )

    return tweets


# =========================
# Save to CSV / JSON
# =========================
def save_tweets(tweets: List[Tweet]):
    ensure_output_dir()

    data = [asdict(t) for t in tweets]
    df = pd.DataFrame(data)

    df.to_csv(CSV_FILE, index=False)
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[Saved] {len(df)} tweets")
    print(f"CSV  -> {CSV_FILE}")
    print(f"JSON -> {JSON_FILE}")


# =========================
# Main Runner
# =========================
def main():
    hashtag = "#AIJobs"
    max_results = 50

    print(f"[Start] Collecting tweets for {hashtag}")

    try:
        tweets = collect_from_twitter(hashtag, max_results)
        if not tweets:
            raise RuntimeError("No tweets returned from API.")
        print("[Mode] Twitter API (Tweepy)")
    except Exception as e:
        print(f"[Warning] API unavailable: {e}")
        print("[Mode] Fallback dataset")
        tweets = generate_fallback_data(hashtag, max_results)

    save_tweets(tweets)


if __name__ == "__main__":
    main()
