"""
collector.py

handles tweet collection for the trend predictor

first attempts to fetch real data using the Twitter API.
if that is not available, it generates sample data so the
rest of the pipeline can still be executed

outputs:
data/raw_tweets.csv
data/raw_tweets.json
"""

import os
import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd

try:
    import tweepy
except ImportError:
    tweepy = None

#set up logging so we can trace what happens during collection
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

#file paths and retry settings for data collection
OUTPUT_DIR    = "data"
CSV_FILE      = os.path.join(OUTPUT_DIR, "raw_tweets.csv")
JSON_FILE     = os.path.join(OUTPUT_DIR, "raw_tweets.json")
MAX_RETRIES   = 3
RETRY_DELAY_S = 5

#simple structure to store tweet data in a consistent format
@dataclass
class Tweet:
    tweet_id:      str
    created_at:    str
    text:          str
    author_id:     str
    username:      Optional[str]
    retweet_count: int
    reply_count:   int
    like_count:    int
    quote_count:   int

# make sure the data folder exists before saving files
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# clean and validate the hashtag input
# ensures it is not empty and always starts with '#'
def _validate_hashtag(hashtag: str) -> str:
    hashtag = hashtag.strip()
    if not hashtag:
        raise ValueError("Hashtag cannot be empty.")
    if not hashtag.startswith("#"):
        hashtag = "#" + hashtag
    if len(hashtag) <= 1:
        raise ValueError("Hashtag must contain at least one character after '#'.")
    return hashtag

# adjust requested tweet count to stay within api limits
def _validate_max_results(n: int) -> int:
    if n < 10:
        log.warning("max_results below API minimum (10) — raising to 10.")
        return 10
    if n > 100:
        log.warning("max_results above API maximum (100) — capping at 100.")
        return 100
    return n

# try collecting real tweets using the twitter api

# fetch tweets from the twitter api
# retries a few times if the request fails, and raises an error if it still cannot get data so we can fall back safely
def collect_from_twitter(hashtag: str, max_results: int = 50) -> List[Tweet]:
    if tweepy is None:
        raise RuntimeError("Tweepy is not installed. Run: pip install tweepy")

    bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise RuntimeError(
            "TWITTER_BEARER_TOKEN environment variable is not set. "
            "Export your bearer token before running this script."
        )

    hashtag     = _validate_hashtag(hashtag)
    max_results = _validate_max_results(max_results)
    client      = tweepy.Client(bearer_token=bearer_token)
    response    = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("API request attempt %d/%d for '%s'", attempt, MAX_RETRIES, hashtag)
            response = client.search_recent_tweets(
                query=hashtag,
                max_results=max_results,
                tweet_fields=["created_at", "public_metrics", "author_id"],
                expansions=["author_id"],
                user_fields=["username"]
            )
            break
        except tweepy.errors.TweepyException as exc:
            log.warning("Tweepy error on attempt %d: %s", attempt, exc)
            if attempt < MAX_RETRIES:
                log.info("Waiting %ds before retry...", RETRY_DELAY_S)
                time.sleep(RETRY_DELAY_S)
            else:
                raise RuntimeError(
                    f"All {MAX_RETRIES} API attempts failed. Last error: {exc}"
                ) from exc

    if response is None or not response.data:
        log.warning("API returned no tweet data for '%s'.", hashtag)
        return []

    user_lookup: dict = {}
    if response.includes and "users" in response.includes:
        for user in response.includes["users"]:
            user_lookup[str(user.id)] = user.username

    tweets: List[Tweet] = []
    for t in response.data:
        metrics = t.public_metrics or {}
        if t.created_at is None:
            log.debug("Skipping tweet %s — missing created_at.", t.id)
            continue
        tweets.append(Tweet(
            tweet_id      = str(t.id),
            created_at    = str(t.created_at),
            text          = t.text or "",
            author_id     = str(t.author_id),
            username      = user_lookup.get(str(t.author_id)),
            retweet_count = int(metrics.get("retweet_count", 0)),
            reply_count   = int(metrics.get("reply_count",   0)),
            like_count    = int(metrics.get("like_count",    0)),
            quote_count   = int(metrics.get("quote_count",   0)),
        ))

    log.info("Collected %d tweets from API.", len(tweets))
    return tweets

# fallback: generate sample tweets if api is unavailable

# generate sample tweets for testing when APO is not available
# timestamps are spaced out and engagement varies slightly so the data still behaves like a real trend
def generate_fallback_data(hashtag: str, n: int = 50) -> List[Tweet]:
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}.")

    hashtag = _validate_hashtag(hashtag)

    base_texts = [
        f"{hashtag} This topic is gaining attention fast.",
        f"People keep talking about {hashtag}.",
        f"{hashtag} opinions are getting intense online.",
        f"{hashtag} might be the next big trend.",
        f"Seeing {hashtag} everywhere today.",
    ]

    tweets: List[Tweet] = []
    now = pd.Timestamp.utcnow()
# create slightly varied tweets so downstream analysis is not too repetitive
    for i in range(n):
        tweets.append(Tweet(
            tweet_id      = f"sample_{i + 1}",
            created_at    = (now - pd.Timedelta(minutes=i * 5)).isoformat(),
            text          = base_texts[i % len(base_texts)],
            author_id     = f"user_{i % 10}",
            username      = f"user_{i % 10}",
            retweet_count = (i * 2) % 20,
            reply_count   = (i * 3) % 15,
            like_count    = (i * 5) % 50,
            quote_count   = (i * 1) % 5,
        ))

    log.info("Generated %d synthetic tweets for '%s'.", len(tweets), hashtag)
    return tweets

# save collected tweets to disk

# save tweets as CSV and JSON
# fails early if the list is empty since later steps depend on this data
def save_tweets(tweets: List[Tweet]):
    if not tweets:
        raise ValueError(
            "Tweet list is empty — nothing to save. "
            "Check that collection succeeded before calling save_tweets()."
        )

    ensure_output_dir()
    data = [asdict(t) for t in tweets]
    df   = pd.DataFrame(data)
    # ensure key fields exist before saving to avoid broken files
    required_cols = {"tweet_id", "created_at", "text", "author_id", "like_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"DataFrame is missing required columns: {missing}")

    df.to_csv(CSV_FILE, index=False)
    log.info("CSV saved  → %s  (%d rows)", CSV_FILE, len(df))

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("JSON saved → %s", JSON_FILE)

# main execution flow

def main():
    hashtag     = "#AIJobs"
    max_results = 50

    log.info("Starting data collection for '%s'", hashtag)

    try:
        # try live data first
        tweets = collect_from_twitter(hashtag, max_results)
        if not tweets:
            raise RuntimeError("API returned zero tweets — switching to fallback.")
        log.info("Mode: Twitter API (Tweepy)")
    except RuntimeError as e:
        log.warning("API unavailable: %s", e)
        log.info("Mode: Fallback synthetic dataset")
        # switch to fallback if API fails
        tweets = generate_fallback_data(hashtag, max_results)
    except ValueError as e:
        log.error("Invalid input: %s", e)
        return

    try:
        # save results for downstream modules
        save_tweets(tweets)
    except (ValueError, RuntimeError) as e:
        log.error("Failed to save tweets: %s", e)


if __name__ == "__main__":
    main()
