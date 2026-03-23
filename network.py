"""
network.py
Social Network Analysis Module

Reads: data/raw_tweets.csv
Writes: data/network_features.csv   (per-user centrality + community)
        data/network_graph.png       (visualisation)
        data/network_stats.txt       (summary stats)
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DATA_DIR   = "data"
CSV_IN     = os.path.join(DATA_DIR, "raw_tweets.csv")
FEAT_OUT   = os.path.join(DATA_DIR, "network_features.csv")
GRAPH_OUT  = os.path.join(DATA_DIR, "network_graph.png")
STATS_OUT  = os.path.join(DATA_DIR, "network_stats.txt")


# ── Graph construction ────────────────────────────────────────────────────────
def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed interaction graph.
    An edge A → B means tweet from A mentions/retweets B.

    For synthetic/fallback data (no explicit mention data), we infer
    lightweight co-activity edges: users active in the same 15-min window
    get a weak edge (weight = shared activity count).
    """
    G = nx.DiGraph()

    # Add all users as nodes
    users = df["author_id"].dropna().unique()
    G.add_nodes_from(users)

    # -- Strategy 1: explicit retweet/mention edges (real API data)
    if "mentioned_users" in df.columns:
        for _, row in df.iterrows():
            src = row["author_id"]
            targets = str(row.get("mentioned_users", "")).split(",")
            for tgt in targets:
                tgt = tgt.strip()
                if tgt and tgt != src:
                    if G.has_edge(src, tgt):
                        G[src][tgt]["weight"] += 1
                    else:
                        G.add_edge(src, tgt, weight=1)

    # -- Strategy 2: co-activity edges (fallback / synthetic data)
    if G.number_of_edges() == 0:
        df_t = df.copy()
        df_t["created_at"] = pd.to_datetime(df_t["created_at"], errors="coerce")
        df_t["window"] = df_t["created_at"].dt.floor("15min")

        for window, group in df_t.groupby("window"):
            active_users = group["author_id"].dropna().tolist()
            for i in range(len(active_users)):
                for j in range(i + 1, min(i + 4, len(active_users))):  # limit fan-out
                    a, b = active_users[i], active_users[j]
                    if a != b:
                        # engagement of the tweet drives edge weight
                        weight = int(group.iloc[i].get("like_count", 1) + 1)
                        if G.has_edge(a, b):
                            G[a][b]["weight"] += weight
                        else:
                            G.add_edge(a, b, weight=weight)

    return G


# ── Centrality + community ────────────────────────────────────────────────────
def compute_features(G: nx.DiGraph) -> pd.DataFrame:
    """Compute degree, betweenness, pagerank, and community label."""

    # Degree centrality (normalised)
    in_deg   = nx.in_degree_centrality(G)
    out_deg  = nx.out_degree_centrality(G)

    # Betweenness centrality (use undirected for small graphs)
    U        = G.to_undirected()
    between  = nx.betweenness_centrality(U, normalized=True)

    # PageRank (measures influence in directed graph)
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=200)

    # Community detection via greedy modularity (undirected)
    communities = {}
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(U))
        for cid, comm in enumerate(comms):
            for node in comm:
                communities[node] = cid
    except Exception:
        communities = {n: 0 for n in G.nodes()}

    records = []
    for node in G.nodes():
        records.append({
            "user_id":          node,
            "in_degree":        in_deg.get(node, 0),
            "out_degree":       out_deg.get(node, 0),
            "betweenness":      between.get(node, 0),
            "pagerank":         pagerank.get(node, 0),
            "community":        communities.get(node, 0),
        })

    return pd.DataFrame(records).sort_values("pagerank", ascending=False)


# ── Visualisation ─────────────────────────────────────────────────────────────
def visualise_graph(G: nx.DiGraph, features: pd.DataFrame):
    """Draw network with nodes sized by PageRank, coloured by community."""

    # Use undirected for cleaner layout
    U = G.to_undirected()

    # Node sizes proportional to PageRank
    pr_map  = features.set_index("user_id")["pagerank"].to_dict()
    comm_map = features.set_index("user_id")["community"].to_dict()

    nodes   = list(U.nodes())
    sizes   = [max(100, pr_map.get(n, 0) * 8000) for n in nodes]
    comms   = [comm_map.get(n, 0) for n in nodes]
    n_comms = max(comms) + 1

    # Color map
    cmap    = cm.get_cmap("tab10", n_comms)
    colors  = [cmap(c) for c in comms]

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(U, seed=42, k=1.5)

    # Draw edges
    nx.draw_networkx_edges(U, pos, ax=ax, alpha=0.25,
                           edge_color="#888888", arrows=False, width=0.8)

    # Draw nodes
    nx.draw_networkx_nodes(U, pos, ax=ax,
                           node_size=sizes, node_color=colors, alpha=0.85)

    # Labels (only top-5 by PageRank)
    top5 = features.head(5)["user_id"].tolist()
    label_nodes = {n: n for n in nodes if n in top5}
    nx.draw_networkx_labels(U, pos, labels=label_nodes, ax=ax,
                            font_size=8, font_color="black")

    # Legend for communities
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=cmap(i), markersize=9,
                           label=f"Community {i}")
               for i in range(n_comms)]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    ax.set_title("User Interaction Network\n(node size = PageRank, colour = community)",
                 fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_OUT, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[SNA] Saved graph → {GRAPH_OUT}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run_network_analysis(csv_path: str = CSV_IN) -> pd.DataFrame:

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}. Run collector.py first.")

    df = pd.read_csv(csv_path)
    print(f"[SNA] Loaded {len(df)} tweets, {df['author_id'].nunique()} unique users")

    G = build_graph(df)
    print(f"[SNA] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    features = compute_features(G)
    os.makedirs(DATA_DIR, exist_ok=True)
    features.to_csv(FEAT_OUT, index=False)
    print(f"[SNA] Saved network features → {FEAT_OUT}")

    # Stats summary
    stats = [
        f"Network Analysis Summary",
        f"=" * 40,
        f"Nodes (users):      {G.number_of_nodes()}",
        f"Edges (interactions): {G.number_of_edges()}",
        f"Density:            {nx.density(G):.4f}",
        f"Communities found:  {features['community'].nunique()}",
        f"",
        f"Top 5 users by PageRank:",
    ]
    for _, row in features.head(5).iterrows():
        stats.append(f"  {row['user_id']:20s}  PR={row['pagerank']:.4f}  "
                     f"betw={row['betweenness']:.4f}  comm={int(row['community'])}")

    summary = "\n".join(stats)
    print("\n" + summary)
    with open(STATS_OUT, "w") as f:
        f.write(summary)
    print(f"[SNA] Saved stats → {STATS_OUT}")

    visualise_graph(G, features)

    return features


if __name__ == "__main__":
    features = run_network_analysis()
    print("\n[SNA] Sample features:")
    print(features.head())
