"""
app.py  —  Social Media Trend Predictor (Full Pipeline)
Integrates: data collection → NLP → topic modelling → network analysis → ML prediction
Run with:   streamlit run app.py
"""

import os
import re
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Trend Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Social Media Trend Predictor")
st.caption("CM3070 Final Project · Bhatnagar Ishita · 10252025")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    hashtag  = st.text_input("Hashtag / topic", value="#AIJobs")
    n_topics = st.slider("Number of LDA topics", 2, 6, 3)
    run_btn  = st.button("▶ Run full analysis", use_container_width=True)
    st.divider()
    st.markdown("**Pipeline steps**")
    st.markdown("1. 🗂 Collect data\n2. 🧹 Preprocess text\n3. 💬 Sentiment analysis\n4. 📊 Feature engineering\n5. 🔍 Topic modelling\n6. 🕸 Network analysis\n7. 🤖 ML prediction")

if not run_btn:
    st.info("Type a hashtag in the sidebar and click **▶ Run full analysis**.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def collect_data(hashtag: str, n: int = 50):
    """Collect tweets for the given hashtag (API or fallback)."""
    import json
    from dataclasses import dataclass, asdict
    from typing import Optional

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

    tag = hashtag if hashtag.startswith("#") else "#" + hashtag

    # Try live API first
    try:
        import tweepy, os as _os
        bearer = _os.getenv("TWITTER_BEARER_TOKEN")
        if not bearer:
            raise RuntimeError("No bearer token")
        client = tweepy.Client(bearer_token=bearer)
        response = client.search_recent_tweets(
            query=tag,
            max_results=min(n, 100),
            tweet_fields=["created_at", "public_metrics", "author_id"],
            expansions=["author_id"],
            user_fields=["username"]
        )
        user_lookup = {}
        if response.includes and "users" in response.includes:
            for u in response.includes["users"]:
                user_lookup[str(u.id)] = u.username
        tweets = []
        if response.data:
            for t in response.data:
                m = t.public_metrics or {}
                tweets.append(Tweet(
                    tweet_id=str(t.id),
                    created_at=str(t.created_at),
                    text=t.text,
                    author_id=str(t.author_id),
                    username=user_lookup.get(str(t.author_id)),
                    retweet_count=m.get("retweet_count", 0),
                    reply_count=m.get("reply_count", 0),
                    like_count=m.get("like_count", 0),
                    quote_count=m.get("quote_count", 0),
                ))
        if tweets:
            source = "Twitter API"
        else:
            raise RuntimeError("No tweets returned")
    except Exception:
        # Fallback: synthetic data for the requested hashtag
        base_texts = [
            f"{tag} This topic is gaining attention fast.",
            f"People keep talking about {tag}.",
            f"{tag} opinions are getting intense online.",
            f"{tag} might be the next big trend.",
            f"Seeing {tag} everywhere today.",
        ]
        now = pd.Timestamp.utcnow()
        tweets = [
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
            for i in range(n)
        ]
        source = "fallback (synthetic)"

    os.makedirs("data", exist_ok=True)
    data = [asdict(t) for t in tweets]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join("data", "raw_tweets.csv"), index=False)
    with open(os.path.join("data", "raw_tweets.json"), "w") as f:
        json.dump(data, f, indent=2)

    return df, source


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def get_sentiment(text: str):
    from textblob import TextBlob
    p = TextBlob(text).sentiment.polarity
    label = "Positive" if p > 0 else ("Negative" if p < 0 else "Neutral")
    return p, label


def build_features(df: pd.DataFrame):
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at")
    df["clean_text"] = df["text"].apply(clean_text)
    df["sentiment_score"], df["sentiment_label"] = zip(*df["clean_text"].apply(get_sentiment))

    df_time = df.set_index("created_at")
    features = df_time.resample("15min").agg({
        "tweet_id":        "count",
        "sentiment_score": ["mean", "std"],
        "like_count":      "mean",
        "retweet_count":   "mean"
    })
    features.columns = ["tweet_volume", "sentiment_mean",
                        "sentiment_std", "avg_likes", "avg_retweets"]
    features = features.fillna(0)
    return df, features


def heuristic_score(features: pd.DataFrame) -> float:
    recent = features.tail(4)
    vol_n  = min(recent["tweet_volume"].mean()   / 30.0, 1.0)
    eng_n  = min((recent["avg_likes"].mean() + recent["avg_retweets"].mean()) / 40.0, 1.0)
    emo_n  = min(recent["sentiment_mean"].abs().mean() / 0.6, 1.0)
    return float(np.clip(0.45 * vol_n + 0.35 * eng_n + 0.20 * emo_n, 0, 1))


def run_topic_modelling(df: pd.DataFrame, n_topics: int):
    from gensim import corpora, models
    from gensim.parsing.preprocessing import STOPWORDS

    tag_word = hashtag.lstrip("#").lower()
    extra_stop = {"aijobs","ai","jobs","job","people","keep","talking",
                  "seeing","opinions","getting","topic","next","big",
                  "trend","everywhere","today","sample","might", tag_word}

    def tokenise(text):
        tokens = clean_text(text).split()
        return [t for t in tokens if t not in STOPWORDS | extra_stop and len(t) > 2]

    df = df.copy()
    df["tokens"] = df["text"].apply(tokenise)
    valid = df[df["tokens"].apply(len) >= 2].copy()
    if len(valid) < n_topics * 2:
        n_topics = max(2, len(valid) // 3)

    dictionary = corpora.Dictionary(valid["tokens"])
    dictionary.filter_extremes(no_below=1, no_above=0.95)
    corpus = [dictionary.doc2bow(t) for t in valid["tokens"]]

    lda = models.LdaModel(corpus=corpus, id2word=dictionary,
                          num_topics=n_topics, passes=10,
                          random_state=42, alpha="auto")

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
    return topic_df, topic_labels, n_topics


def run_network_analysis(df: pd.DataFrame):
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(df["author_id"].dropna().unique())

    df_t = df.copy()
    df_t["created_at"] = pd.to_datetime(df_t["created_at"], errors="coerce")
    df_t["window"] = df_t["created_at"].dt.floor("15min")

    for _, group in df_t.groupby("window"):
        active = group["author_id"].dropna().tolist()
        for i in range(len(active)):
            for j in range(i + 1, min(i + 4, len(active))):
                a, b = active[i], active[j]
                if a != b:
                    w = int(group.iloc[i].get("like_count", 1)) + 1
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += w
                    else:
                        G.add_edge(a, b, weight=w)

    import networkx as nx
    pr    = nx.pagerank(G, alpha=0.85, max_iter=200)
    U     = G.to_undirected()
    btwn  = nx.betweenness_centrality(U, normalized=True)
    in_d  = nx.in_degree_centrality(G)

    communities = {}
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        for cid, comm in enumerate(greedy_modularity_communities(U)):
            for n in comm:
                communities[n] = cid
    except Exception:
        communities = {n: 0 for n in G.nodes()}

    net_df = pd.DataFrame([{
        "user_id":     n,
        "pagerank":    pr.get(n, 0),
        "betweenness": btwn.get(n, 0),
        "in_degree":   in_d.get(n, 0),
        "community":   communities.get(n, 0),
    } for n in G.nodes()]).sort_values("pagerank", ascending=False)

    return G, net_df


def ml_predict(features: pd.DataFrame) -> dict:
    model_path = os.path.join("data", "best_model.pkl")
    feat_cols  = ["tweet_volume", "sentiment_mean", "sentiment_std",
                  "avg_likes", "avg_retweets"]
    X = features[feat_cols].fillna(0)
    score = X["tweet_volume"] + X["avg_likes"] + X["avg_retweets"]
    y = (score >= np.percentile(score, 70)).astype(int)

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        from sklearn.ensemble      import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline      import Pipeline
        model = Pipeline([("scaler", StandardScaler()),
                          ("clf", RandomForestClassifier(n_estimators=100,
                                                          random_state=42,
                                                          class_weight="balanced"))])
        factor  = max(1, 30 // max(len(X), 1))
        X_train = pd.concat([X] * factor, ignore_index=True)
        y_train = pd.concat([y] * factor, ignore_index=True)
        model.fit(X_train, y_train)

    preds = model.predict(X)
    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        probs = preds.astype(float)

    return {"predictions": preds, "probabilities": probs, "labels": y.values}


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
progress = st.progress(0, text="Starting pipeline…")

# Step 1: Collect
progress.progress(10, f"🗂 Collecting data for {hashtag}…")
df_raw, data_source = collect_data(hashtag, n=50)
st.sidebar.caption(f"Data source: {data_source}")

# Step 2-4: Features
progress.progress(30, "🧹 Preprocessing & sentiment analysis…")
try:
    df_enriched, features = build_features(df_raw)
except Exception as e:
    st.error(f"Feature engineering failed: {e}")
    st.stop()

os.makedirs("data", exist_ok=True)
features.to_csv(os.path.join("data", "ml_features.csv"))
h_score = heuristic_score(features)

# Step 5: Topics
progress.progress(50, "🔍 Running LDA topic modelling…")
try:
    topic_df, topic_labels, n_topics_actual = run_topic_modelling(df_raw, n_topics)
    topic_ok = True
except Exception as e:
    topic_ok = False; topic_err = str(e)

# Step 6: Network
progress.progress(68, "🕸 Building interaction network…")
try:
    G, net_df = run_network_analysis(df_raw)
    network_ok = True
except Exception as e:
    network_ok = False; net_err = str(e)

# Step 7: ML
progress.progress(85, "🤖 Running ML prediction…")
try:
    ml_results = ml_predict(features)
    ml_ok = True
except Exception as e:
    ml_ok = False; ml_err = str(e)

progress.progress(100, "✅ Done!")
progress.empty()

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════
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
    st.metric("Positive tweets", sc.get("Positive", 0))
    st.metric("Neutral tweets",  sc.get("Neutral",  0))

with col4:
    if ml_ok:
        st.metric("Trending windows (ML)",
                  f"{ml_results['predictions'].sum()}/{len(ml_results['predictions'])}")
        st.metric("Avg trend probability", f"{ml_results['probabilities'].mean():.2f}")
    else:
        st.metric("Negative tweets", sc.get("Negative", 0))

st.divider()

# Sentiment
st.subheader("💬 Sentiment Analysis")
c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    colors = {"Positive":"#70AD47","Neutral":"#4472C4","Negative":"#ED7D31"}
    bars = ax.bar(sc.index, sc.values,
                  color=[colors.get(l,"#999") for l in sc.index])
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=9)
    ax.set_title("Sentiment Distribution", fontsize=11)
    ax.set_ylabel("Count")
    plt.tight_layout(); st.pyplot(fig); plt.close()

with c2:
    fig, ax = plt.subplots(figsize=(5, 3.2))
    features["sentiment_mean"].plot(marker="o", ax=ax, color="#4472C4", linewidth=1.5)
    ax.axhline(0, color="#ccc", linewidth=0.8, linestyle="--")
    ax.set_title("Average Sentiment Over Time (15-min bins)", fontsize=11)
    ax.set_xlabel("Time"); ax.set_ylabel("Avg polarity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

st.divider()

# Topics
st.subheader("🔍 LDA Topic Modelling")
if topic_ok:
    tc1, tc2 = st.columns([1, 1.4])
    with tc1:
        st.markdown("**Discovered topics**")
        for label in topic_labels:
            st.markdown(f"- {label}")
    with tc2:
        dist = topic_df["dominant_topic"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar([f"Topic {i}" for i in dist.index], dist.values,
               color=["#4472C4","#70AD47","#ED7D31","#9E63B5","#E16182"][:len(dist)])
        ax.set_title("Tweets per Dominant Topic", fontsize=11)
        ax.set_ylabel("Count")
        plt.tight_layout(); st.pyplot(fig); plt.close()
else:
    st.warning(f"Topic modelling failed: {topic_err}")

st.divider()

# Network
st.subheader("🕸 Social Network Analysis")
if network_ok:
    import networkx as nx
    import matplotlib.cm as cm
    nc1, nc2 = st.columns([1.2, 1])
    with nc1:
        U      = G.to_undirected()
        nodes  = list(U.nodes())
        pr_map = net_df.set_index("user_id")["pagerank"].to_dict()
        cm_map = net_df.set_index("user_id")["community"].to_dict()
        n_comm = max(cm_map.values()) + 1
        cmap   = cm.get_cmap("tab10", n_comm)
        sizes  = [max(80, pr_map.get(n,0)*6000) for n in nodes]
        colors = [cmap(cm_map.get(n,0)) for n in nodes]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        pos = nx.spring_layout(U, seed=42, k=1.5)
        nx.draw_networkx_edges(U, pos, ax=ax, alpha=0.2, edge_color="#888",
                               arrows=False, width=0.7)
        nx.draw_networkx_nodes(U, pos, ax=ax, node_size=sizes,
                               node_color=colors, alpha=0.85)
        top5 = net_df.head(5)["user_id"].tolist()
        nx.draw_networkx_labels(U, pos, ax=ax,
                                labels={n:n for n in nodes if n in top5},
                                font_size=7)
        ax.set_title("User Interaction Network", fontsize=10)
        ax.axis("off"); plt.tight_layout(); st.pyplot(fig); plt.close()
    with nc2:
        st.markdown(f"**Graph stats**")
        st.write(f"- Nodes: {G.number_of_nodes()}")
        st.write(f"- Edges: {G.number_of_edges()}")
        st.write(f"- Density: {nx.density(G):.4f}")
        st.write(f"- Communities: {net_df['community'].nunique()}")
        st.markdown("**Top 5 users by PageRank**")
        st.dataframe(
            net_df[["user_id","pagerank","betweenness","community"]].head(5)
            .rename(columns={"user_id":"User","pagerank":"PageRank",
                             "betweenness":"Betweenness","community":"Community"})
            .round(4),
            use_container_width=True, hide_index=True
        )
else:
    st.warning(f"Network analysis failed: {net_err}")

st.divider()

# ML
st.subheader("🤖 Machine Learning Prediction")
if ml_ok:
    mc1, mc2 = st.columns(2)
    with mc1:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(ml_results["probabilities"], marker="o", color="#4472C4",
                linewidth=1.5, label="Trend probability")
        ax.axhline(0.5, color="#ED7D31", linewidth=1, linestyle="--",
                   label="Decision boundary (0.5)")
        ax.fill_between(range(len(ml_results["probabilities"])),
                        ml_results["probabilities"], 0.5,
                        where=ml_results["probabilities"] >= 0.5,
                        alpha=0.15, color="#70AD47", label="Trending")
        ax.set_ylim(0, 1)
        ax.set_title("ML Trend Probability per Time Window", fontsize=11)
        ax.set_xlabel("Window index"); ax.set_ylabel("Probability")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with mc2:
        st.markdown("**Feature matrix (ML input)**")
        st.dataframe(
            features.reset_index().rename(columns={"created_at":"Time"}).head(8).round(3),
            use_container_width=True, hide_index=True
        )
else:
    st.warning(f"ML prediction failed: {ml_err}")

st.divider()

with st.expander("🗂 Raw tweet data preview"):
    st.dataframe(
        df_enriched[["created_at","text","sentiment_score","sentiment_label"]].head(10),
        use_container_width=True, hide_index=True
    )

st.caption("ml_features.csv saved to data/ml_features.csv  •  Run ml_model.py for full model comparison")
