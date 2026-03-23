"""
network.py

builds a user interaction graph from the tweet data

it works out simple network measures like centrality,
influence, and community grouping. if direct interactions
are not available, it falls back to linking users who were
active in the same time window

saves output to:
data/network_features.csv
data/network_graph.png
data/network_stats.txt
"""

import os
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# logging setup for graph building and analysis
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# input and output file paths used in this module
DATA_DIR  = "data"
CSV_IN    = os.path.join(DATA_DIR, "raw_tweets.csv")
FEAT_OUT  = os.path.join(DATA_DIR, "network_features.csv")
GRAPH_OUT = os.path.join(DATA_DIR, "network_graph.png")
STATS_OUT = os.path.join(DATA_DIR, "network_stats.txt")

# check that the tweet data has the fields needed for network analysis
# also makes sure there is enough user activity to build a useful graph
def _validate_dataframe(df: pd.DataFrame) -> None:
    required = {"author_id", "created_at", "like_count"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(
            f"raw_tweets.csv is missing required column(s): {missing}. "
            "Re-run collector.py to regenerate the dataset."
        )
    if df.empty:
        raise ValueError("raw_tweets.csv contains no rows — nothing to analyse.")
    n_users = df["author_id"].dropna().nunique()
    if n_users < 2:
        raise ValueError(
            f"Only {n_users} unique author(s) found. At least 2 are needed "
            "to build a meaningful interaction graph."
        )
    log.info("DataFrame validated: %d tweets, %d unique users.", len(df), n_users)

# build the interaction graph from the tweet dataset
# if direct links like mentions exist, use those first. otherwise, connect users who were active in the same 15-minute window so the graph is still usable.
def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    # start with all users as nodes, even before edges are added
    G     = nx.DiGraph()
    users = df["author_id"].dropna().unique()
    G.add_nodes_from(users)
    log.info("Graph initialised with %d nodes.", G.number_of_nodes())

    # use direct interactions first if the dataset has them
    if "mentioned_users" in df.columns:
        for _, row in df.iterrows():
            src     = row["author_id"]
            raw_tgt = str(row.get("mentioned_users", ""))
            targets = [t.strip() for t in raw_tgt.split(",") if t.strip()]
            for tgt in targets:
                if tgt == src:
                    continue   # ignore self-links
                if G.has_edge(src, tgt):
                    G[src][tgt]["weight"] += 1
                else:
                    G.add_edge(src, tgt, weight=1)
        log.info("Explicit edges added: %d.", G.number_of_edges())

    # if no direct links exist, connect users who posted around the same time
    if G.number_of_edges() == 0:
        log.info("No explicit edges found — building co-activity edges.")
        df_t = df.copy()
        df_t["created_at"] = pd.to_datetime(df_t["created_at"], errors="coerce")

        n_bad_ts = df_t["created_at"].isna().sum()
        if n_bad_ts > 0:
            log.warning("%d row(s) have unparseable timestamps and will be skipped.", n_bad_ts)

        df_t = df_t.dropna(subset=["created_at"])
        df_t["window"] = df_t["created_at"].dt.floor("15min")

        for _, group in df_t.groupby("window"):
            active = group["author_id"].dropna().tolist()
            for i in range(len(active)):
                # limit extra links here so the graph does not become overly crowded
                for j in range(i + 1, min(i + 4, len(active))):
                    a, b = active[i], active[j]
                    if a == b:
                        continue
                    weight = int(group.iloc[i].get("like_count", 1)) + 1
                    if G.has_edge(a, b):
                        G[a][b]["weight"] += weight
                    else:
                        G.add_edge(a, b, weight=weight)

        log.info("Co-activity edges added: %d.", G.number_of_edges())

    return G

# calculate network features for each user in the graph
# these values help show who is central, who connects groups and who seems more influential overall
def compute_features(G: nx.DiGraph) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes — cannot compute centrality metrics.")

    in_deg  = nx.in_degree_centrality(G)
    out_deg = nx.out_degree_centrality(G)
    # use an undirected version for measures that do not need edge direction
    U       = G.to_undirected()
    between = nx.betweenness_centrality(U, normalized=True)

    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=200)
    except nx.PowerIterationFailedConvergence:
        log.warning("PageRank did not converge — falling back to uniform scores.")
        pr = {n: 1.0 / G.number_of_nodes() for n in G.nodes()}
    # try grouping users into simple communities
    communities: dict = {}
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(U))
        for cid, comm in enumerate(comms):
            for node in comm:
                communities[node] = cid
        log.info("Community detection found %d communities.", len(comms))
    except Exception as exc:
        log.warning("Community detection failed (%s) — assigning all users to community 0.", exc)
        communities = {n: 0 for n in G.nodes()}

    records = []
    for node in G.nodes():
        records.append({
            "user_id":     node,
            "in_degree":   in_deg.get(node, 0),
            "out_degree":  out_deg.get(node, 0),
            "betweenness": between.get(node, 0),
            "pagerank":    pr.get(node, 0),
            "community":   communities.get(node, 0),
        })

    return pd.DataFrame(records).sort_values("pagerank", ascending=False)

# draw the interaction graph for easier inspection
# node size is based on pagerank and colour shows community grouping, only the top few users are labelled to keep the figure readable
def visualise_graph(G: nx.DiGraph, features: pd.DataFrame):
    if features.empty:
        log.warning("Features table is empty — skipping graph visualisation.")
        return

    U        = G.to_undirected()
    pr_map   = features.set_index("user_id")["pagerank"].to_dict()
    comm_map = features.set_index("user_id")["community"].to_dict()

    nodes   = list(U.nodes())
    sizes   = [max(100, pr_map.get(n, 0) * 8000) for n in nodes]
    comms   = [int(comm_map.get(n, 0)) for n in nodes]
    n_comms = max(comms) + 1 if comms else 1
    cmap    = cm.get_cmap("tab10", n_comms)
    colors  = [cmap(c) for c in comms]

    fig, ax = plt.subplots(figsize=(8, 6))
    pos     = nx.spring_layout(U, seed=42, k=1.5)

    nx.draw_networkx_edges(U, pos, ax=ax, alpha=0.25,
                           edge_color="#888888", arrows=False, width=0.8)
    nx.draw_networkx_nodes(U, pos, ax=ax,
                           node_size=sizes, node_color=colors, alpha=0.85)
    # only label the top few users so the plot stays readable
    top5        = features.head(5)["user_id"].tolist()
    label_nodes = {n: n for n in nodes if n in top5}
    nx.draw_networkx_labels(U, pos, labels=label_nodes, ax=ax,
                            font_size=8, font_color="black")

    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=cmap(i), markersize=9,
                           label=f"Community {i}")
               for i in range(n_comms)]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.set_title(
        "User Interaction Network\n(node size = PageRank, colour = community)",
        fontsize=11
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(GRAPH_OUT, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Graph saved → %s", GRAPH_OUT)

# run the full network analysis flow
# loads the tweet data, builds the graph, extracts features, saves the outputs and creates a graph visualisation
def run_network_analysis(csv_path: str = CSV_IN) -> pd.DataFrame:

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Run collector.py first."
        )
    # load tweet data collected earlier
    df = pd.read_csv(csv_path)
    _validate_dataframe(df)
    # build graph structure from user activity
    G = build_graph(df)
    log.info("Graph built: %d nodes, %d edges.", G.number_of_nodes(), G.number_of_edges())
    # calculate graph-based features for each user
    features = compute_features(G)
    os.makedirs(DATA_DIR, exist_ok=True)
    features.to_csv(FEAT_OUT, index=False)
    log.info("Network features saved → %s  (%d rows)", FEAT_OUT, len(features))
    #prepare a short text summary for quick review
    stats_lines = [
        "Network Analysis Summary",
        "=" * 40,
        f"Nodes (users):        {G.number_of_nodes()}",
        f"Edges (interactions): {G.number_of_edges()}",
        f"Density:              {nx.density(G):.4f}",
        f"Communities found:    {features['community'].nunique()}",
        "",
        "Top 5 users by PageRank:",
    ]
    for _, row in features.head(5).iterrows():
        stats_lines.append(
            f"  {row['user_id']:20s}  PR={row['pagerank']:.4f}  "
            f"betw={row['betweenness']:.4f}  comm={int(row['community'])}"
        )

    summary = "\n".join(stats_lines)
    log.info("\n%s", summary)
    with open(STATS_OUT, "w", encoding="utf-8") as f:
        f.write(summary)
    log.info("Stats saved → %s", STATS_OUT)
    # save a visual version of the graph as well
    visualise_graph(G, features)
    return features


if __name__ == "__main__":
    features = run_network_analysis()
    log.info("Sample features:\n%s", features.head().to_string())
