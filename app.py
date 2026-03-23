"""
app.py

streamlit dashboard for the social media trend predictor

runs the full pipeline in one place:
data collection → text cleaning → sentiment → feature building →
topic modelling → network analysis → trend prediction

each step is kept separate so if one part fails,
the rest of the app can still run and display results

run with:
    python -m streamlit run app.py
"""

import os
import re
import json
import pickle
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

st.set_page_config(
    page_title="Social Media Trend Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Social Media Trend Predictor")
st.caption("CM3070 Final Project · Bhatnagar Ishita · 10252025")

with st.sidebar:
    st.header("⚙️ Controls")
    hashtag  = st.text_input("Hashtag / topic", value="#AIJobs")
    n_topics = st.slider("Number of LDA topics", 2, 6, 3)
    run_btn  = st.button("▶ Run full analysis", use_container_width=True)
    st.divider()
    st.markdown("**Pipeline steps**")
    st.markdown(
        "1. 🗂 Collect data\n"
        "2. 🧹 Preprocess text\n"
        "3. 💬 Sentiment analysis\n"
        "4. 📊 Feature engineering\n"
        "5. 🔍 Topic modelling\n"
        "6. 🕸 Network analysis\n"
        "7. 🤖 ML prediction"
    )

if not run_btn:
    st.info("Type a hashtag in the sidebar and click **▶ Run full analysis**.")
    st.stop()

# quick sanity check before we run anything heavy
# prevents empty/invalid hashtags from breaking the pipeline later

_clean_tag = hashtag.strip()
if not _clean_tag or _clean_tag == "#":
    st.error("Please enter a valid hashtag before running the analysis.")
    st.stop()

if not _clean_tag.startswith("#"):
    _clean_tag = "#" + _clean_tag

if len(_clean_tag) < 2:
    st.error("Hashtag must contain at least one character after the '#' symbol.")
    st.stop()

hashtag = _clean_tag


# -------------------------
# Core processing functions
# -------------------------
def collect_data(hashtag: str, n: int = 50):
    """
handles tweet collection with a fallback safety net

tries the Twitter API first, but if anything fails
(no token, API error, etc), we generate fake data instead

this keeps the whole app usable even without API accesseturning an empty table.
    """
    if n <= 0:
        raise ValueError(f"Tweet count n must be positive, got {n}.")

    from dataclasses import dataclass, asdict
    from typing import Optional

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

    tag    = hashtag if hashtag.startswith("#") else "#" + hashtag
    source = "fallback (synthetic)"
    tweets = []

    try:
        import tweepy
        bearer = os.getenv("TWITTER_BEARER_TOKEN")
        if not bearer:
            raise RuntimeError("TWITTER_BEARER_TOKEN is not set.")

        client   = tweepy.Client(bearer_token=bearer)
        response = client.search_recent_tweets(
            query=tag,
            max_results=min(max(n, 10), 100),
            tweet_fields=["created_at", "public_metrics", "author_id"],
            expansions=["author_id"],
            user_fields=["username"]
        )

        user_lookup: dict = {}
        if response.includes and "users" in response.includes:
            for u in response.includes["users"]:
                user_lookup[str(u.id)] = u.username

        if not response.data:
            raise RuntimeError("API returned an empty response for this hashtag.")

        for t in response.data:
            m = t.public_metrics or {}
            if t.created_at is None:
                log.debug("Skipping tweet %s — missing created_at.", t.id)
                continue
            tweets.append(Tweet(
                tweet_id      = str(t.id),
                created_at    = str(t.created_at),
                text          = t.text or "",
                author_id     = str(t.author_id),
                username      = user_lookup.get(str(t.author_id)),
                retweet_count = int(m.get("retweet_count", 0)),
                reply_count   = int(m.get("reply_count",   0)),
                like_count    = int(m.get("like_count",    0)),
                quote_count   = int(m.get("quote_count",   0)),
            ))

        if not tweets:
            raise RuntimeError("All API tweets were filtered (missing timestamps).")

        source = "Twitter API"
        log.info("Collected %d tweets via API.", len(tweets))

    except ImportError:
        log.warning("Tweepy not installed — using synthetic data.")
    except RuntimeError as exc:
        log.warning("API unavailable: %s — using synthetic data.", exc)
    except Exception as exc:
        log.warning("Unexpected API error (%s) — using synthetic data.", exc)

    if not tweets:
        base_texts = [
            f"{tag} This topic is gaining attention fast.",
            f"People keep talking about {tag}.",
            f"{tag} opinions are getting intense online.",
            f"{tag} might be the next big trend.",
            f"Seeing {tag} everywhere today.",
        ]
        now = pd.Timestamp.utcnow()
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
        log.info("Generated %d synthetic tweets for '%s'.", len(tweets), tag)

    os.makedirs("data", exist_ok=True)
    data = [asdict(t) for t in tweets]
    df   = pd.DataFrame(data)
    df.to_csv(os.path.join("data", "raw_tweets.csv"), index=False)
    with open(os.path.join("data", "raw_tweets.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info("Saved %d tweets → data/raw_tweets.csv", len(df))
    return df, source


def clean_text(text: str) -> str:
    """
    basic cleaning so the text works nicely for NLP

    handles messy inputs (None, numbers, links, mentions)
    so later steps don’t crash unexpectedly
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_sentiment(text: str):
    """
    return (polarity_score, label) for a cleaned tweet string.
    catches TextBlob exceptions and returns (0.0, 'Neutral') as a safe
    default so a single malformed tweet does not abort the batch.
    """
    try:
        from textblob import TextBlob
        p     = TextBlob(text).sentiment.polarity
        label = "Positive" if p > 0 else ("Negative" if p < 0 else "Neutral")
        return float(p), label
    except Exception as exc:
        log.warning("Sentiment failed for '%.40s': %s", text, exc)
        return 0.0, "Neutral"


def build_features(df: pd.DataFrame):
    """
    turn raw tweets into structured features

    this is where everything becomes usable for ML:
    - clean text
    - sentiment scoring
    - group into time windows

    also catches bad data early so we don’t pass junk downstrea
    """
    required = {"tweet_id", "text", "like_count", "retweet_count"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(f"Tweet DataFrame is missing column(s): {missing}")

    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    n_bad = int(df["created_at"].isna().sum())
    if n_bad > 0:
        log.warning("%d row(s) had unparseable timestamps and were dropped.", n_bad)

    df = df.dropna(subset=["created_at"]).sort_values("created_at")

    if df.empty:
        raise ValueError(
            "No rows remain after timestamp parsing. "
            "Check that created_at values are in a recognisable datetime format."
        )

    df["clean_text"]      = df["text"].apply(clean_text)
    results               = df["clean_text"].apply(get_sentiment)
    df["sentiment_score"] = [r[0] for r in results]
    df["sentiment_label"] = [r[1] for r in results]

    df_time  = df.set_index("created_at")
    features = df_time.resample("15min").agg({
        "tweet_id":        "count",
        "sentiment_score": ["mean", "std"],
        "like_count":      "mean",
        "retweet_count":   "mean"
    })
    features.columns = ["tweet_volume", "sentiment_mean",
                        "sentiment_std", "avg_likes", "avg_retweets"]
    features = features.fillna(0)
    log.info("Feature matrix built: %d windows.", len(features))
    return df, features


def heuristic_score(features: pd.DataFrame) -> float:
    """
    produce a trend likelihood score between 0 and 1.
    uses only the four most recent windows to reflect current momentum.
    returns 0.0 and logs a warning if the feature matrix is empty.
    """
    if features.empty:
        log.warning("Feature matrix is empty — returning heuristic score 0.0.")
        return 0.0

    recent = features.tail(4)
    vol_n  = min(recent["tweet_volume"].mean() / 30.0, 1.0)
    eng_n  = min((recent["avg_likes"].mean() + recent["avg_retweets"].mean()) / 40.0, 1.0)
    emo_n  = min(recent["sentiment_mean"].abs().mean() / 0.6, 1.0)
    score  = float(np.clip(0.45 * vol_n + 0.35 * eng_n + 0.20 * emo_n, 0.0, 1.0))
    log.info("Heuristic score: %.3f", score)
    return score


def run_topic_modelling(df: pd.DataFrame, n_topics: int):
    """
    fit an LDA model and return per-tweet topic assignments.

    validates n_topics, reduces it automatically when the corpus is small,
    and raises RuntimeError with a clear message if the Gensim dictionary
    ends up empty — preventing a confusing internal Gensim error.
    """
    from gensim import corpora, models
    from gensim.parsing.preprocessing import STOPWORDS

    if n_topics < 2:
        raise ValueError(f"n_topics must be at least 2, got {n_topics}.")

    tag_word   = hashtag.lstrip("#").lower()
    extra_stop = {
        "aijobs", "ai", "jobs", "job", "people", "keep", "talking",
        "seeing", "opinions", "getting", "topic", "next", "big",
        "trend", "everywhere", "today", "sample", "might", tag_word
    }

    def tokenise(text):
        tokens = clean_text(text).split()
        return [t for t in tokens if t not in STOPWORDS | extra_stop and len(t) > 2]

    df = df.copy()
    df["tokens"] = df["text"].apply(tokenise)
    valid = df[df["tokens"].apply(len) >= 2].copy()

    if valid.empty:
        raise ValueError(
            "No tweets had enough tokens for topic modelling. "
            "Try a different hashtag or reduce the stopword list."
        )

    if len(valid) < n_topics * 2:
        n_topics = max(2, len(valid) // 3)
        log.warning("Corpus too small — reduced to %d topics.", n_topics)

    dictionary = corpora.Dictionary(valid["tokens"])
    dictionary.filter_extremes(no_below=1, no_above=0.95)

    if len(dictionary) == 0:
        raise RuntimeError(
            "LDA dictionary is empty after filtering. "
            "The dataset may be too small or too repetitive."
        )

    corpus = [dictionary.doc2bow(t) for t in valid["tokens"]]

    try:
        lda = models.LdaModel(corpus=corpus, id2word=dictionary,
                              num_topics=n_topics, passes=10,
                              random_state=42, alpha="auto")
    except Exception as exc:
        raise RuntimeError(f"LDA training failed: {exc}") from exc

    topic_labels = [
        f"Topic {i}: {', '.join(w for w, _ in lda.show_topic(i, topn=5))}"
        for i in range(n_topics)
    ]

    topic_cols = [f"topic_{i}_prob" for i in range(n_topics)]
    rows = []
    for bow in corpus:
        dist = dict(lda.get_document_topics(bow, minimum_probability=0.0))
        rows.append([dist.get(i, 0.0) for i in range(n_topics)])

    topic_df = pd.DataFrame(rows, columns=topic_cols, index=valid.index)
    topic_df["dominant_topic"] = (
        topic_df[topic_cols].idxmax(axis=1)
        .str.replace("topic_", "").str.replace("_prob", "").astype(int)
    )
    log.info("LDA complete: %d topics, %d docs.", n_topics, len(valid))
    return topic_df, topic_labels, n_topics


def run_network_analysis(df: pd.DataFrame):
    """
    build a directed user interaction graph and compute centrality metrics.

    validates that at least two unique users exist before building the graph.
    handles PageRank convergence failures and community detection errors
    separately so a problem with one metric does not cancel the others.
    """
    import networkx as nx

    if "author_id" not in df.columns:
        raise KeyError("'author_id' column not found — re-run data collection.")

    n_users = int(df["author_id"].dropna().nunique())
    if n_users < 2:
        raise ValueError(
            f"Only {n_users} unique user(s) — need at least 2 to build a graph."
        )

    G     = nx.DiGraph()
    users = df["author_id"].dropna().unique()
    G.add_nodes_from(users)

    df_t = df.copy()
    df_t["created_at"] = pd.to_datetime(df_t["created_at"], errors="coerce")
    df_t = df_t.dropna(subset=["created_at"])
    df_t["window"] = df_t["created_at"].dt.floor("15min")

    for _, group in df_t.groupby("window"):
        active = group["author_id"].dropna().tolist()
        for i in range(len(active)):
            for j in range(i + 1, min(i + 4, len(active))):
                a, b = active[i], active[j]
                if a == b:
                    continue
                weight = int(group.iloc[i].get("like_count", 1)) + 1
                if G.has_edge(a, b):
                    G[a][b]["weight"] += weight
                else:
                    G.add_edge(a, b, weight=weight)

    log.info("Graph: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges())

    U      = G.to_undirected()
    in_deg = nx.in_degree_centrality(G)
    btwn   = nx.betweenness_centrality(U, normalized=True)

    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        log.warning("PageRank did not converge — using uniform scores.")
        pr = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}

    communities: dict = {}
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        for cid, comm in enumerate(greedy_modularity_communities(U)):
            for node in comm:
                communities[node] = cid
    except Exception as exc:
        log.warning("Community detection failed (%s) — assigning all to community 0.", exc)
        communities = {n: 0 for n in G.nodes()}

    net_df = pd.DataFrame([{
        "user_id":     n,
        "pagerank":    pr.get(n, 0),
        "betweenness": btwn.get(n, 0),
        "in_degree":   in_deg.get(n, 0),
        "community":   communities.get(n, 0),
    } for n in G.nodes()]).sort_values("pagerank", ascending=False)

    return G, net_df


def ml_predict(features: pd.DataFrame) -> dict:
    """
    score each time window using a saved model or a freshly trained one.

    checks that all required feature columns exist before attempting prediction.
    falls back to a Random Forest trained on proxy labels when no saved model
    is found. Handles the case where predict_proba is unavailable by casting
    binary predictions to float rather than raising AttributeError.
    """
    feat_cols = ["tweet_volume", "sentiment_mean", "sentiment_std",
                 "avg_likes", "avg_retweets"]

    missing = [c for c in feat_cols if c not in features.columns]
    if missing:
        raise KeyError(f"Feature matrix is missing column(s) for ML: {missing}")

    X = features[feat_cols].fillna(0)

    if X.empty:
        raise ValueError("Feature matrix has no rows — cannot run ML prediction.")

    score     = X["tweet_volume"] + X["avg_likes"] + X["avg_retweets"]
    threshold = float(np.percentile(score, 70))
    y         = (score >= threshold).astype(int)

    model_path = os.path.join("data", "best_model.pkl")
    model      = None

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            log.info("Loaded saved model from '%s'.", model_path)
        except (pickle.UnpicklingError, EOFError) as exc:
            log.warning("Saved model file is corrupt (%s) — training fresh.", exc)

    if model is None:
        from sklearn.ensemble      import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline      import Pipeline

        model  = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100,
                                              random_state=42,
                                              class_weight="balanced"))
        ])
        factor  = max(1, 30 // max(len(X), 1))
        X_train = pd.concat([X] * factor, ignore_index=True)
        y_train = pd.concat([y] * factor, ignore_index=True)
        model.fit(X_train, y_train)
        log.info("Trained fresh Random Forest on %d rows.", len(X_train))

    preds = model.predict(X)

    try:
        probs = model.predict_proba(X)[:, 1]
    except AttributeError:
        log.warning("Model has no predict_proba — using binary predictions as scores.")
        probs = preds.astype(float)

    return {"predictions": preds, "probabilities": probs, "labels": y.values}


# -------------------------
# Run full pipeline step by step
# -------------------------

progress = st.progress(0, text="Starting pipeline...")

progress.progress(10, f"🗂 Collecting data for {hashtag}...")
try:
    df_raw, data_source = collect_data(hashtag, n=50)
    st.sidebar.caption(f"Data source: {data_source}")
except ValueError as e:
    st.error(f"Invalid collection settings: {e}")
    st.stop()
except Exception as e:
    st.error(f"Data collection failed unexpectedly: {e}")
    st.stop()

progress.progress(30, "🧹 Preprocessing & sentiment analysis...")
try:
    df_enriched, features = build_features(df_raw)
except KeyError as e:
    st.error(f"Missing column in tweet data: {e}")
    st.stop()
except ValueError as e:
    st.error(f"Feature engineering failed: {e}")
    st.stop()

os.makedirs("data", exist_ok=True)
features.to_csv(os.path.join("data", "ml_features.csv"))
h_score = heuristic_score(features)

progress.progress(50, "🔍 Running LDA topic modelling...")
try:
    topic_df, topic_labels, n_topics_actual = run_topic_modelling(df_raw, n_topics)
    topic_ok  = True
    topic_err = ""
except (ValueError, RuntimeError, KeyError) as e:
    topic_ok  = False
    topic_err = str(e)
    log.warning("Topic modelling failed: %s", e)

progress.progress(68, "🕸 Building interaction network...")
try:
    G, net_df  = run_network_analysis(df_raw)
    network_ok = True
    net_err    = ""
except (ValueError, KeyError) as e:
    network_ok = False
    net_err    = str(e)
    log.warning("Network analysis failed: %s", e)

progress.progress(85, "🤖 Running ML prediction...")
try:
    ml_results = ml_predict(features)
    ml_ok      = True
    ml_err     = ""
except (KeyError, ValueError) as e:
    ml_ok  = False
    ml_err = str(e)
    log.warning("ML prediction failed: %s", e)

progress.progress(100, "✅ Done!")
progress.empty()


# -------------------------
# Display
# -------------------------

st.subheader(f"📊 Trend Overview — {hashtag}")
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

with col1:
    st.metric("Heuristic trend score", f"{h_score:.2f}")
    if h_score >= 0.70:
        st.success("🔥 Likely to trend")
    elif h_score >= 0.40:
        st.warning("⚡ Moderate signal")
    else:
        st.info("💤 Low trend signal")

with col2:
    st.metric("Tweets collected", len(df_raw))
    st.metric("Time windows", len(features))

with col3:
    sc = df_enriched["sentiment_label"].value_counts()
    st.metric("Positive tweets", int(sc.get("Positive", 0)))
    st.metric("Neutral tweets",  int(sc.get("Neutral",  0)))

with col4:
    if ml_ok:
        st.metric("Trending windows (ML)",
                  f"{int(ml_results['predictions'].sum())}/{len(ml_results['predictions'])}")
        st.metric("Avg trend probability",
                  f"{float(ml_results['probabilities'].mean()):.2f}")
    else:
        st.metric("Negative tweets", int(sc.get("Negative", 0)))

st.divider()

#Sentiment charts
st.subheader("💬 Sentiment Analysis")
c1, c2 = st.columns(2)

with c1:
    if sc.empty:
        st.warning("No sentiment data to display.")
    else:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        color_map = {"Positive": "#70AD47", "Neutral": "#4472C4", "Negative": "#ED7D31"}
        bars = ax.bar(sc.index, sc.values,
                      color=[color_map.get(lbl, "#999") for lbl in sc.index])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=9)
        ax.set_title("Sentiment Distribution", fontsize=11)
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with c2:
    if features["sentiment_mean"].isna().all():
        st.warning("Sentiment time-series is empty.")
    else:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        features["sentiment_mean"].plot(marker="o", ax=ax,
                                        color="#4472C4", linewidth=1.5)
        ax.axhline(0, color="#ccc", linewidth=0.8, linestyle="--")
        ax.set_title("Average Sentiment Over Time (15-min bins)", fontsize=11)
        ax.set_xlabel("Time")
        ax.set_ylabel("Avg polarity")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()

# Topic modelling
st.subheader("🔍 LDA Topic Modelling")
if topic_ok:
    tc1, tc2 = st.columns([1, 1.4])
    with tc1:
        st.markdown("**Discovered topics**")
        for label in topic_labels:
            st.markdown(f"- {label}")
    with tc2:
        dist = topic_df["dominant_topic"].value_counts().sort_index()
        if dist.empty:
            st.warning("No dominant topic counts to display.")
        else:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar([f"Topic {i}" for i in dist.index], dist.values,
                   color=["#4472C4", "#70AD47", "#ED7D31", "#9E63B5", "#E16182"][:len(dist)])
            ax.set_title("Tweets per Dominant Topic", fontsize=11)
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
else:
    st.warning(f"Topic modelling could not complete: {topic_err}")

st.divider()

#Network analysis
st.subheader("🕸 Social Network Analysis")
if network_ok:
    import networkx as nx
    import matplotlib.cm as cm

    nc1, nc2 = st.columns([1.2, 1])
    with nc1:
        U        = G.to_undirected()
        nodes    = list(U.nodes())
        pr_map   = net_df.set_index("user_id")["pagerank"].to_dict()
        comm_map = net_df.set_index("user_id")["community"].to_dict()
        n_comm   = max(comm_map.values()) + 1 if comm_map else 1
        cmap     = cm.get_cmap("tab10", n_comm)
        sizes    = [max(80, pr_map.get(n, 0) * 6000) for n in nodes]
        colors   = [cmap(int(comm_map.get(n, 0))) for n in nodes]

        fig, ax = plt.subplots(figsize=(5.5, 4))
        pos     = nx.spring_layout(U, seed=42, k=1.5)
        nx.draw_networkx_edges(U, pos, ax=ax, alpha=0.2,
                               edge_color="#888", arrows=False, width=0.7)
        nx.draw_networkx_nodes(U, pos, ax=ax,
                               node_size=sizes, node_color=colors, alpha=0.85)
        top5 = net_df.head(5)["user_id"].tolist()
        nx.draw_networkx_labels(U, pos, ax=ax,
                                labels={n: n for n in nodes if n in top5},
                                font_size=7)
        ax.set_title("User Interaction Network", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with nc2:
        st.markdown("**Graph stats**")
        st.write(f"- Nodes: {G.number_of_nodes()}")
        st.write(f"- Edges: {G.number_of_edges()}")
        st.write(f"- Density: {nx.density(G):.4f}")
        st.write(f"- Communities: {net_df['community'].nunique()}")
        st.markdown("**Top 5 users by PageRank**")
        st.dataframe(
            net_df[["user_id", "pagerank", "betweenness", "community"]].head(5)
            .rename(columns={"user_id": "User", "pagerank": "PageRank",
                             "betweenness": "Betweenness", "community": "Community"})
            .round(4),
            use_container_width=True, hide_index=True
        )
else:
    st.warning(f"Network analysis could not complete: {net_err}")

st.divider()

#Machine Learning prediction
st.subheader("🤖 Machine Learning Prediction")
if ml_ok:
    mc1, mc2 = st.columns(2)
    with mc1:
        probs = ml_results["probabilities"]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(probs, marker="o", color="#4472C4",
                linewidth=1.5, label="Trend probability")
        ax.axhline(0.5, color="#ED7D31", linewidth=1,
                   linestyle="--", label="Decision boundary (0.5)")
        ax.fill_between(range(len(probs)), probs, 0.5,
                        where=probs >= 0.5,
                        alpha=0.15, color="#70AD47", label="Trending zone")
        ax.set_ylim(0, 1)
        ax.set_title("ML Trend Probability per Time Window", fontsize=11)
        ax.set_xlabel("Window index")
        ax.set_ylabel("Probability")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with mc2:
        st.markdown("**Feature matrix (ML input)**")
        st.dataframe(
            features.reset_index()
            .rename(columns={"created_at": "Time"})
            .head(8).round(3),
            use_container_width=True, hide_index=True
        )
else:
    st.warning(f"ML prediction could not complete: {ml_err}")

st.divider()

with st.expander("🗂 Raw tweet data preview"):
    display_cols = ["created_at", "text", "sentiment_score", "sentiment_label"]
    available    = [c for c in display_cols if c in df_enriched.columns]
    if available:
        st.dataframe(df_enriched[available].head(10),
                     use_container_width=True, hide_index=True)
    else:
        st.warning("No displayable columns found in the enriched tweet table.")

st.caption(
    "ml_features.csv saved to data/ml_features.csv  •  "
    "Run ml_model.py for the full model comparison report"
)
