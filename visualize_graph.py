"""
visualize_graph.py — Visualisasi Graph AML Detection
=====================================================
Script ini memuat graph PyTorch Geometric yang telah dibangun dari
dataset AML dan menghasilkan beberapa visualisasi:

1. Statistik struktur graph
2. Degree distribution (in-degree & out-degree)
3. Sample subgraph dengan transaksi laundering di-highlight
4. Distribusi fitur edges
5. Adjacency pattern untuk top accounts

Usage:
    python visualize_graph.py

Output:
    - aml_graph_visualizations.png  (kompilasi semua plot)
    - aml_subgraph_sample.png       (subgraph zoom)
"""

import os
import sys
import json
import random
import warnings
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Optional: networkx untuk layout graph
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from backend.config import DATA_PATH, ENCODERS_PATH

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load Graph Data
# ---------------------------------------------------------------------------
def load_graph_data(path: str = DATA_PATH):
    """Load processed_data.pt and return PyG Data object."""
    print(f"[1/5] Loading graph from: {path}")
    if not os.path.exists(path):
        print(f"      ERROR: File not found: {path}")
        print("      Pastikan Anda sudah menjalankan: python -m backend.ml.data_pipeline")
        sys.exit(1)

    data = torch.load(path, map_location="cpu", weights_only=False)
    print(f"      [OK] Loaded successfully")
    print(f"        - Nodes (accounts) : {data.num_nodes:,}")
    print(f"        - Edges (tx)       : {data.edge_index.shape[1]:,}")
    print(f"        - Node features    : {data.x.shape[1]} dims")
    print(f"        - Edge features    : {data.edge_attr.shape[1]} dims")
    print(f"        - Labels           : {data.y.shape[0]:,}")
    return data


# ---------------------------------------------------------------------------
# 2. Compute Degree Statistics
# ---------------------------------------------------------------------------
def compute_degrees(data):
    """Compute in-degree and out-degree for all nodes."""
    num_nodes = data.num_nodes
    edge_index = data.edge_index  # [2, num_edges]

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    out_degrees = Counter(src)
    in_degrees = Counter(dst)

    out_deg = np.array([out_degrees.get(i, 0) for i in range(num_nodes)], dtype=np.int64)
    in_deg = np.array([in_degrees.get(i, 0) for i in range(num_nodes)], dtype=np.int64)
    total_deg = out_deg + in_deg

    return in_deg, out_deg, total_deg


# ---------------------------------------------------------------------------
# 3. Plot Degree Distributions
# ---------------------------------------------------------------------------
def plot_degree_distributions(ax, in_deg, out_deg, total_deg):
    """Plot in-degree, out-degree, and total degree histograms (log-log)."""
    # Filter zeros for log scale
    in_nonzero = in_deg[in_deg > 0]
    out_nonzero = out_deg[out_deg > 0]
    total_nonzero = total_deg[total_deg > 0]

    bins = np.logspace(0, np.log10(max(total_nonzero.max(), 10)), 40)

    ax.hist(in_nonzero, bins=bins, alpha=0.5, label=f"In-degree  (μ={in_deg.mean():.1f})", color="steelblue")
    ax.hist(out_nonzero, bins=bins, alpha=0.5, label=f"Out-degree (μ={out_deg.mean():.1f})", color="coral")
    ax.hist(total_nonzero, bins=bins, alpha=0.4, label=f"Total deg  (μ={total_deg.mean():.1f})", color="green")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("Degree Distribution")
    ax.legend(loc="upper right")
    ax.grid(True, ls="--", alpha=0.3)


# ---------------------------------------------------------------------------
# 4. Plot Class Distribution per Split
# ---------------------------------------------------------------------------
def plot_class_distribution(ax, data):
    """Bar chart of normal vs laundering per split."""
    splits = [
        ("Train", data.train_mask),
        ("Val", data.val_mask),
        ("Test", data.test_mask),
    ]

    normal_counts = []
    fraud_counts = []
    labels = []

    for name, mask in splits:
        y_split = data.y[mask].numpy()
        normal = int((y_split == 0).sum())
        fraud = int((y_split == 1).sum())
        normal_counts.append(normal)
        fraud_counts.append(fraud)
        labels.append(name)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, normal_counts, width, label="Normal", color="steelblue")
    bars2 = ax.bar(x + width/2, fraud_counts, width, label="Laundering", color="crimson")

    ax.set_ylabel("Number of Transactions")
    ax.set_title("Class Distribution per Split")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, ls="--", alpha=0.3, axis="y")

    # Annotate ratios
    for i, (n, f) in enumerate(zip(normal_counts, fraud_counts)):
        ratio = n / max(f, 1)
        ax.annotate(f"{ratio:.1f}:1", xy=(i, max(n, f)), ha="center", fontsize=9, color="black")


# ---------------------------------------------------------------------------
# 5. Plot Edge Feature Distributions
# ---------------------------------------------------------------------------
def plot_edge_features(ax, data):
    """Show distribution of selected edge features (unnormalized approx)."""
    edge_attr = data.edge_attr.numpy()
    y = data.y.numpy()

    # Pick Amount (index 0) and temporal_weight (index 1)
    # Note: these are StandardScaler-normalized, so show relative
    feat_names = ["Amount (norm)", "Temp. Weight (norm)", "Delta T min (norm)", "Payment Type (norm)"]
    idxs = [0, 1, 7, 6]

    colors_normal = "steelblue"
    colors_fraud = "crimson"

    for i, (fn, idx) in enumerate(zip(feat_names, idxs)):
        row = i // 2
        col = i % 2
        sub_ax = ax.inset_axes([col * 0.52, (1 - row) * 0.48, 0.48, 0.44])

        vals_normal = edge_attr[y == 0, idx]
        vals_fraud = edge_attr[y == 1, idx]

        # Clamp extreme outliers for visualization
        q99 = np.quantile(edge_attr[:, idx], 0.99)
        q01 = np.quantile(edge_attr[:, idx], 0.01)

        sub_ax.hist(np.clip(vals_normal, q01, q99), bins=50, alpha=0.5, color=colors_normal, label="Normal", density=True)
        sub_ax.hist(np.clip(vals_fraud, q01, q99), bins=50, alpha=0.5, color=colors_fraud, label="Laundering", density=True)

        sub_ax.set_title(fn, fontsize=10)
        sub_ax.set_yticks([])
        if i == 0:
            sub_ax.legend(fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Edge Feature Distributions (Normalized)")


# ---------------------------------------------------------------------------
# 6. Subgraph Visualization (NetworkX)
# ---------------------------------------------------------------------------
def _build_tree_layout(tree_edges, root, max_depth=7):
    """Compute hierarchical (x,y) positions for a rooted tree."""
    from collections import defaultdict, deque

    children = defaultdict(list)
    for p, c in tree_edges:
        children[p].append(c)

    levels = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if levels[node] >= max_depth:
            continue
        for ch in children.get(node, []):
            if ch not in levels:
                levels[ch] = levels[node] + 1
                queue.append(ch)

    by_level = defaultdict(list)
    for node, lvl in levels.items():
        by_level[lvl].append(node)

    pos = {}
    for lvl, nodes in by_level.items():
        n = len(nodes)
        spread = max(n * 2.2, 14)
        step = spread / max(n, 1)
        start_x = -spread / 2 + step / 2
        for i, node in enumerate(nodes):
            pos[node] = (start_x + i * step, -lvl * 3.0)
    return pos


# Edge severity categories based on normalized Amount (edge_attr[:, 0])
# Only laundering edges (y == 1) are visualized, split into 4 severity levels.
CATEGORY_COLORS = {
    1: "#3498db",   # Low
    2: "#f1c40f",   # Moderate
    3: "#e67e22",   # High
    4: "#c0392b",   # Critical
}

CATEGORY_NAMES = {
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Critical",
}

CATEGORY_WIDTHS = {
    1: 1.0,
    2: 1.3,
    3: 1.7,
    4: 2.2,
}


def compute_edge_categories(data):
    """
    Kategorikan edges ke severity levels berdasarkan normalized Amount.

    Categories:
        0: Normal (y == 0)
        1: Laundering - Low     (y == 1, amount <= q25 dari laundering)
        2: Laundering - Moderate (y == 1, amount <= q50)
        3: Laundering - High    (y == 1, amount <= q75)
        4: Laundering - Critical (y == 1, amount > q75)
    """
    y = data.y.numpy()
    edge_attr = data.edge_attr.numpy()
    amount = edge_attr[:, 0]  # Amount adalah fitur pertama (normalized)

    categories = np.zeros(len(y), dtype=int)

    fraud_mask = y == 1
    fraud_amounts = amount[fraud_mask]

    if len(fraud_amounts) == 0:
        return categories

    q25 = np.quantile(fraud_amounts, 0.25)
    q50 = np.quantile(fraud_amounts, 0.50)
    q75 = np.quantile(fraud_amounts, 0.75)

    fraud_idx = np.where(fraud_mask)[0]
    for idx in fraud_idx:
        val = amount[idx]
        if val <= q25:
            categories[idx] = 1
        elif val <= q50:
            categories[idx] = 2
        elif val <= q75:
            categories[idx] = 3
        else:
            categories[idx] = 4

    return categories


def plot_subgraph(ax, data, max_edges=100, seed=42, edge_categories=None):
    """
    Render a clean, readable directed transaction subgraph using a
    force-directed (spring) layout.  The look is inspired by classic
    network diagrams: large circular nodes, thin labelled edges, and
    plenty of whitespace.

    Edge colors vary by laundering severity category:
        Low, Moderate, High, Critical
    """
    if not HAS_NX:
        ax.text(0.5, 0.5, "NetworkX not installed.\nInstall with: pip install networkx",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Subgraph Visualization (Unavailable)")
        ax.axis("off")
        return

    rng = np.random.default_rng(seed)
    edge_index = data.edge_index.numpy()
    y = data.y.numpy()
    num_edges_total = len(y)

    n_fraud_total = int(y.sum())
    n_normal_total = num_edges_total - n_fraud_total

    COLOR_NORMAL = "forestgreen"
    COLOR_FRAUD = "crimson"

    # ---- 1. Connected edge sampling (BFS-like) ----
    sampled_mask = np.zeros(num_edges_total, dtype=bool)

    # Start from a fraud edge if available, otherwise random
    fraud_edges = np.where(y == 1)[0]
    if len(fraud_edges) > 0:
        start_edge = int(rng.choice(fraud_edges))
    else:
        start_edge = int(rng.integers(0, num_edges_total))

    sampled_mask[start_edge] = True
    sampled_count = 1

    nodes_in_sample = set()
    nodes_in_sample.add(int(edge_index[0, start_edge]))
    nodes_in_sample.add(int(edge_index[1, start_edge]))

    # Greedy BFS expansion: repeatedly pick edges connected to already-selected nodes
    attempts = 0
    max_attempts = max_edges * 20
    while sampled_count < max_edges and attempts < max_attempts:
        node = int(rng.choice(list(nodes_in_sample)))
        src_mask = edge_index[0, :] == node
        dst_mask = edge_index[1, :] == node
        connected = np.where(src_mask | dst_mask)[0]
        if len(connected) == 0:
            attempts += 1
            continue
        e = int(rng.choice(connected))
        if not sampled_mask[e]:
            sampled_mask[e] = True
            sampled_count += 1
            nodes_in_sample.add(int(edge_index[0, e]))
            nodes_in_sample.add(int(edge_index[1, e]))
        attempts += 1

    # If BFS couldn't fill quota, fill with random laundering edges
    if sampled_count < max_edges:
        remaining = max_edges - sampled_count
        fraud_idx_rem = np.where((~sampled_mask) & (y == 1))[0]
        if len(fraud_idx_rem) > 0:
            extra = rng.choice(fraud_idx_rem, size=min(remaining, len(fraud_idx_rem)), replace=False)
            sampled_mask[extra] = True

    sampled_edges = np.where(sampled_mask)[0]

    # Keep only laundering edges for visualization
    sampled_edges = sampled_edges[y[sampled_edges] == 1]
    if len(sampled_edges) < max_edges:
        remaining = max_edges - len(sampled_edges)
        already_used = set(sampled_edges.tolist())
        fraud_idx_rem = np.where(y == 1)[0]
        fraud_idx_rem = np.array([i for i in fraud_idx_rem if i not in already_used])
        if len(fraud_idx_rem) > 0:
            extra = rng.choice(fraud_idx_rem, size=min(remaining, len(fraud_idx_rem)), replace=False)
            sampled_edges = np.concatenate([sampled_edges, extra])

    rng.shuffle(sampled_edges)

    # ---- 2. Build edge lists ----
    sub_edges = []
    edge_colors = []
    edge_widths = []
    nodes_involved = set()
    n_fraud_vis = 0

    for e in sampled_edges:
        s = int(edge_index[0, e])
        d = int(edge_index[1, e])
        sub_edges.append((s, d))
        nodes_involved.add(s)
        nodes_involved.add(d)

        if edge_categories is not None:
            cat = int(edge_categories[e])
            edge_colors.append(CATEGORY_COLORS[cat])
            edge_widths.append(CATEGORY_WIDTHS[cat])
            n_fraud_vis += 1
        else:
            lbl = int(y[e])
            if lbl == 1:
                edge_colors.append(COLOR_FRAUD)
                edge_widths.append(1.4)
                n_fraud_vis += 1
            else:
                edge_colors.append(COLOR_NORMAL)
                edge_widths.append(0.7)

    # ---- 3. NetworkX graph ----
    G = nx.DiGraph()
    G.add_nodes_from(nodes_involved)
    G.add_edges_from(sub_edges)

    # Only remove completely isolated nodes (degree == 0) so the graph stays readable
    # but still preserves leaves and weakly-connected structure.
    deg = dict(G.degree())
    isolated = [n for n, d in deg.items() if d == 0]
    G.remove_nodes_from(isolated)
    deg = dict(G.degree())
    node_list = list(G.nodes())

    # Pick a layout that spreads nodes naturally without a strong circular rim
    if len(node_list) <= 150:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=seed, k=1.2, iterations=150, scale=2.0)

    # Node sizes: larger for high-degree nodes so they stand out
    node_sizes = [400 + 60 * min(deg.get(n, 0), 12) for n in node_list]

    # Draw nodes: white fill, dark-grey border
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=node_sizes,
                           node_color="white", edgecolors="#333333",
                           linewidths=1.0, ax=ax, alpha=0.95)

    # Draw edges per category
    surviving = set(node_list)
    for cat in sorted(CATEGORY_COLORS.keys()):
        cat_edges = [(u, v) for (u, v), c in zip(sub_edges, edge_colors)
                     if c == CATEGORY_COLORS[cat] and u in surviving and v in surviving]
        if cat_edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=cat_edges,
                edge_color=CATEGORY_COLORS[cat],
                arrows=True, arrowsize=9 if cat <= 2 else 11,
                width=CATEGORY_WIDTHS[cat], ax=ax,
                alpha=0.85,
                connectionstyle="arc3,rad=0.08",
                min_source_margin=8, min_target_margin=8
            )

    # Labels: show only for top-N highest-degree nodes so the graph stays readable
    top_nodes = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:15]
    labels = {n: str(n)[:6] for n, _ in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color="#222222",
                            font_weight="bold", ax=ax)

    if edge_categories is not None:
        present_cats = sorted({int(edge_categories[e]) for e in sampled_edges})
        legend_elements = [
            Line2D([0], [0], color=CATEGORY_COLORS[c], lw=CATEGORY_WIDTHS[c], label=CATEGORY_NAMES[c])
            for c in present_cats
        ]
        cat_counts = Counter(int(edge_categories[e]) for e in sampled_edges)
        parts = [f"{CATEGORY_NAMES[c]}={cat_counts.get(c,0)}" for c in sorted(CATEGORY_NAMES.keys()) if cat_counts.get(c, 0) > 0]
        ratio_str = "  |  ".join(parts)
    else:
        ratio_str = f"{n_normal_vis/n_fraud_vis:.1f}:1" if n_fraud_vis > 0 else "N/A"
        legend_elements = [
            Line2D([0], [0], color=COLOR_FRAUD, lw=2.0, label="Laundering"),
            Line2D([0], [0], color=COLOR_NORMAL, lw=2.0, label="Normal"),
        ]

    ax.set_title(f"Laundering Subgraph ({len(node_list)} nodes, {len(sub_edges)} edges)\n"
                 f"{ratio_str}",
                 fontsize=11)
    ax.axis("off")
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# 7. Top Accounts Table
# ---------------------------------------------------------------------------
def plot_top_accounts(ax, in_deg, out_deg, data):
    """Display a text table of top accounts by degree."""
    total_deg = in_deg + out_deg
    top_idx = np.argsort(total_deg)[-10:][::-1]

    rows = []
    rows.append(["Rank", "Node ID", "In-Deg", "Out-Deg", "Total Deg", "Launder In", "Launder Out"])
    rows.append(["─"*5, "─"*10, "─"*8, "─"*9, "─"*9, "─"*10, "─"*11])

    edge_index = data.edge_index.numpy()
    y = data.y.numpy()

    for rank, idx in enumerate(top_idx, 1):
        # Count laundering edges
        out_mask = edge_index[0, :] == idx
        in_mask = edge_index[1, :] == idx
        launder_out = int(y[out_mask].sum())
        launder_in = int(y[in_mask].sum())
        rows.append([
            str(rank), str(idx),
            str(in_deg[idx]), str(out_deg[idx]), str(total_deg[idx]),
            str(launder_in), str(launder_out)
        ])

    ax.axis("off")
    ax.set_title("Top 10 Accounts by Transaction Degree")

    # Build table
    table = ax.table(cellText=rows[2:], colLabels=rows[0],
                     colColours=["#f0f0f0"]*7,
                     cellLoc="center", loc="center",
                     colWidths=[0.08, 0.12, 0.10, 0.10, 0.10, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)


# ---------------------------------------------------------------------------
# 8. Temporal Pattern
# ---------------------------------------------------------------------------
def plot_temporal_pattern(ax, data):
    """Show how laundering edges are distributed over time (edge index order)."""
    y = data.y.numpy()
    n_edges = len(y)

    # Use edge_attr delta_t (index 7, normalized) — but we can approximate with index
    # Just show cumulative count of laundering edges over edge order (temporal)
    window = max(n_edges // 100, 1000)
    laundering_rate = []
    x_points = []

    for i in range(0, n_edges - window, window):
        segment = y[i:i+window]
        rate = segment.sum() / max(len(segment), 1)
        laundering_rate.append(rate)
        x_points.append(i + window//2)

    ax.plot(x_points, laundering_rate, color="crimson", lw=1.2)
    ax.axhline(y=y.mean(), color="black", ls="--", label=f"Global avg = {y.mean():.4f}")
    ax.set_xlabel("Edge Index (Temporal Order)")
    ax.set_ylabel("Laundering Rate")
    ax.set_title("Laundering Rate Over Time")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)


# ---------------------------------------------------------------------------
# 9. Confusion Matrix & Risk Distribution
# ---------------------------------------------------------------------------
def plot_confusion_matrix():
    """Generate proper binary confusion matrix and risk distribution."""
    import seaborn as sns

    test_metrics_path = os.path.join(BASE_DIR, "data", "test_metrics.json")
    if not os.path.exists(test_metrics_path):
        print("      [WARN] test_metrics.json not found, skipping confusion matrix")
        return None

    with open(test_metrics_path, "r") as f:
        metrics = json.load(f)

    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    # --- Subplot 1: Binary Confusion Matrix (seaborn heatmap style) ---
    ax_cm = fig.add_subplot(gs[0, 0])
    cm_array = np.array([[tn, fp], [fn, tp]], dtype=float)
    labels = ["Normal", "Laundering"]

    sns.heatmap(
        cm_array,
        annot=True,
        fmt=",.0f",
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="gray",
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Scale"},
        ax=ax_cm,
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax_cm.set_xticklabels(labels, rotation=0, ha="center")
    ax_cm.set_yticklabels(labels, rotation=0, va="center")
    ax_cm.set_xlabel("Predicted Label", fontsize=11)
    ax_cm.set_ylabel("True Label", fontsize=11)
    ax_cm.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=10)

    # --- Subplot 2: Risk Category Distribution ---
    ax_risk = fig.add_subplot(gs[0, 1])

    train_metrics_path = os.path.join(BASE_DIR, "data", "training_metrics.json")
    if os.path.exists(train_metrics_path):
        with open(train_metrics_path, "r") as f:
            train_metrics = json.load(f)
        risk_dist = train_metrics.get("risk_distribution", {})
    else:
        risk_dist = {}

    if risk_dist:
        categories = ["Low", "Moderate", "High", "Critical"]
        values = [risk_dist.get(c, 0) for c in categories]
        colors = ["#3498db", "#f1c40f", "#e67e22", "#c0392b"]

        bars = ax_risk.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
        ax_risk.set_ylabel("Number of Transactions", fontsize=11)
        ax_risk.set_title("Predicted Risk Distribution (Test Set)", fontsize=12, fontweight="bold")

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax_risk.annotate(
                f"{val:,}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax_risk.grid(True, ls="--", alpha=0.3, axis="y")
    else:
        ax_risk.text(0.5, 0.5, "Risk distribution data not available",
                     ha="center", va="center", transform=ax_risk.transAxes)
        ax_risk.set_title("Predicted Risk Distribution")

    fig.suptitle("AML Model Evaluation — Confusion Matrix & Risk Distribution",
                 fontsize=14, fontweight="bold", y=1.02)

    out_path = os.path.join(BASE_DIR, "aml_confusion_matrix.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      [OK] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 10. Confusion Matrix Standalone Plot
# ---------------------------------------------------------------------------
def plot_confusion_matrix_only():
    """Generate a standalone confusion matrix plot (new file, does not overwrite)."""
    import seaborn as sns

    test_metrics_path = os.path.join(BASE_DIR, "data", "test_metrics.json")
    if not os.path.exists(test_metrics_path):
        print("      [WARN] test_metrics.json not found, skipping standalone CM plot")
        return None

    with open(test_metrics_path, "r") as f:
        metrics = json.load(f)

    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    cm_array = np.array([[tn, fp], [fn, tp]], dtype=float)
    labels = ["Normal", "Laundering"]

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm_array,
        annot=True,
        fmt=",.0f",
        cmap="YlGnBu",
        linewidths=1.0,
        linecolor="gray",
        square=True,
        cbar_kws={"shrink": 0.75, "label": "Scale"},
        ax=ax,
        annot_kws={"size": 18, "weight": "bold"},
    )

    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=12)
    ax.set_yticklabels(labels, rotation=0, va="center", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)

    prec = metrics.get("test_precision", 0)
    rec = metrics.get("test_recall", 0)
    f1 = metrics.get("test_f1", 0)
    ax.set_title(
        "Confusion Matrix\n"
        f"Precision={prec:.4f}  |  Recall={rec:.4f}  |  F1={f1:.4f}",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    fig.tight_layout()
    out_path = os.path.join(BASE_DIR, "aml_confusion_matrix_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      [OK] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 11. Risk Category Confusion Matrix (True Label vs Predicted Risk)
# ---------------------------------------------------------------------------
def plot_risk_confusion_matrix(data):
    """Generate confusion-matrix-style heatmap: True Label vs Predicted Risk Category."""
    import seaborn as sns

    cache_path = os.path.join(BASE_DIR, "data", "risk_cross_tab.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cross_tab = json.load(f)
        print("      [OK] Loaded cached risk cross-tabulation")
    else:
        # Fallback: run inference if cache missing
        from backend.ml.model import EdgeGATModel, get_risk_category

        test_metrics_path = os.path.join(BASE_DIR, "data", "test_metrics.json")
        with open(test_metrics_path, "r") as f:
            metrics = json.load(f)
        hp = metrics["hyperparameters"]

        model = EdgeGATModel(
            node_feat_dim=data.x.shape[1],
            edge_feat_dim=data.edge_attr.shape[1],
            hidden_dim=hp["hidden_dim"],
            num_heads=hp["num_heads"],
            dropout=hp["dropout"],
        )
        model.load_state_dict(
            torch.load(
                os.path.join(BASE_DIR, "data", "best_model.pt"),
                map_location="cpu",
                weights_only=False,
            )
        )
        model.eval()

        test_mask = data.test_mask
        y = data.y[test_mask].numpy()

        with torch.no_grad():
            probs = model.predict(data.x, data.edge_index, data.edge_attr)[test_mask]

        categories = [get_risk_category(p.item()) for p in probs]
        cat_order = ["Low", "Moderate", "High", "Critical"]

        cross_tab = {"Normal": {}, "Laundering": {}}
        for cat in cat_order:
            cross_tab["Normal"][cat] = 0
            cross_tab["Laundering"][cat] = 0

        for i, cat in enumerate(categories):
            true_label = "Normal" if y[i] == 0 else "Laundering"
            cross_tab[true_label][cat] += 1

        with open(cache_path, "w") as f:
            json.dump(cross_tab, f, indent=2)
        print("      [OK] Computed and cached risk cross-tabulation")

    # Build matrix
    cat_order = ["Low", "Moderate", "High", "Critical"]
    row_labels = ["Normal", "Laundering"]
    matrix = np.array(
        [[cross_tab[row][cat] for cat in cat_order] for row in row_labels],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=",.0f",
        cmap="YlGnBu",
        linewidths=1.0,
        linecolor="gray",
        square=False,
        cbar_kws={"shrink": 0.75, "label": "Scale"},
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )

    ax.set_xticklabels(cat_order, rotation=0, ha="center", fontsize=11)
    ax.set_yticklabels(row_labels, rotation=0, va="center", fontsize=11)
    ax.set_xlabel("Predicted Risk Category", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title(
        "Confusion Matrix — True Label vs Predicted Risk Category",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    fig.tight_layout()
    out_path = os.path.join(BASE_DIR, "aml_risk_confusion_matrix.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"      [OK] Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 12. Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    print("+==========================================================+")
    print("|  AML Graph Visualization                                 |")
    print("+==========================================================+\n")

    data = load_graph_data()
    print()

    print("[2/5] Computing degree statistics ...")
    in_deg, out_deg, total_deg = compute_degrees(data)
    print(f"      [OK] Max in-degree  : {in_deg.max():,}")
    print(f"      [OK] Max out-degree : {out_deg.max():,}")
    print(f"      [OK] Avg degree     : {total_deg.mean():.2f}")
    print()

    print("[2a/5] Computing edge severity categories ...")
    edge_categories = compute_edge_categories(data)
    cat_counts = Counter(edge_categories)
    print(f"      [OK] Low      : {cat_counts[1]:,}")
    print(f"      [OK] Moderate : {cat_counts[2]:,}")
    print(f"      [OK] High     : {cat_counts[3]:,}")
    print(f"      [OK] Critical : {cat_counts[4]:,}")
    print()

    print("[3/5] Building visualization panels ...")

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Row 0
    ax_info = fig.add_subplot(gs[0, 0])
    ax_deg = fig.add_subplot(gs[0, 1])
    ax_class = fig.add_subplot(gs[0, 2])

    # Row 1
    ax_subgraph = fig.add_subplot(gs[1, :2])
    ax_temporal = fig.add_subplot(gs[1, 2])

    # Row 2
    ax_edgefeat = fig.add_subplot(gs[2, :2])
    ax_top = fig.add_subplot(gs[2, 2])

    # --- Panel: Info Text ---
    ax_info.axis("off")
    info_text = (
        "AML Transaction Graph\n"
        "----------------------\n\n"
        f"Nodes (accounts) : {data.num_nodes:,}\n"
        f"Edges (tx)       : {data.edge_index.shape[1]:,}\n"
        f"Node features    : {data.x.shape[1]} dims\n"
        f"Edge features    : {data.edge_attr.shape[1]} dims\n\n"
        f"Normal tx        : {(data.y == 0).sum().item():,}\n"
        f"Laundering tx    : {(data.y == 1).sum().item():,}\n"
        f"Imbalance ratio  : {(data.y == 0).sum().item() / max((data.y == 1).sum().item(), 1):.1f} : 1\n\n"
        f"Train edges      : {data.train_mask.sum().item():,}\n"
        f"Val edges        : {data.val_mask.sum().item():,}\n"
        f"Test edges       : {data.test_mask.sum().item():,}"
    )
    ax_info.text(0.1, 0.5, info_text, transform=ax_info.transAxes,
                 fontsize=11, verticalalignment="center",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#f8f8f8", edgecolor="gray"))
    ax_info.set_title("Graph Overview")

    # --- Panel: Degree Distribution ---
    plot_degree_distributions(ax_deg, in_deg, out_deg, total_deg)

    # --- Panel: Class Distribution ---
    plot_class_distribution(ax_class, data)

    # --- Panel: Subgraph ---
    plot_subgraph(ax_subgraph, data, max_edges=200, seed=42, edge_categories=edge_categories)

    # --- Panel: Temporal ---
    plot_temporal_pattern(ax_temporal, data)

    # --- Panel: Edge Features ---
    plot_edge_features(ax_edgefeat, data)

    # --- Panel: Top Accounts ---
    plot_top_accounts(ax_top, in_deg, out_deg, data)

    fig.suptitle("AML Money Laundering Detection — Graph Visualization",
                 fontsize=16, fontweight="bold", y=0.98)

    out_path = os.path.join(BASE_DIR, "aml_graph_visualizations.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"      [OK] Saved: {out_path}")

    # Create proper confusion matrix (binary) + risk distribution
    print("[3a/5] Creating confusion matrix & risk distribution ...")
    plot_confusion_matrix()

    # Create standalone confusion matrix plot (new file)
    print("[3b/5] Creating standalone confusion matrix plot ...")
    plot_confusion_matrix_only()

    # Create risk-category confusion matrix (True Label vs Predicted Risk)
    print("[3c/5] Creating risk-category confusion matrix ...")
    plot_risk_confusion_matrix(data)

    # Also create a dedicated larger subgraph image
    if HAS_NX:
        print("[4/5] Creating dedicated subgraph visualization ...")
        fig2, ax2 = plt.subplots(figsize=(16, 12))
        plot_subgraph(ax2, data, max_edges=400, seed=123, edge_categories=edge_categories)
        out_path2 = os.path.join(BASE_DIR, "aml_subgraph_sample.png")
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"      [OK] Saved: {out_path2}")
        plt.close(fig2)

        # Multiple subgraph samples (2x2 grid, different seeds)
        print("[4a/5] Creating multiple subgraph samples ...")
        fig3, axes3 = plt.subplots(2, 2, figsize=(20, 18))
        sample_configs = [
            (300, 42,  "Seed 42"),
            (300, 123, "Seed 123"),
            (300, 456, "Seed 456"),
            (300, 789, "Seed 789"),
        ]
        for ax3, (me, sd, title_suffix) in zip(axes3.flat, sample_configs):
            plot_subgraph(ax3, data, max_edges=me, seed=sd, edge_categories=edge_categories)
            # Append seed info to the title set inside plot_subgraph
            cur_title = ax3.get_title()
            ax3.set_title(f"{cur_title}\n({title_suffix})", fontsize=10)
        fig3.suptitle("AML Graph — Multiple Subgraph Samples", fontsize=18, fontweight="bold", y=0.98)
        out_path3 = os.path.join(BASE_DIR, "aml_subgraph_samples.png")
        fig3.savefig(out_path3, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"      [OK] Saved: {out_path3}")
        plt.close(fig3)

    plt.close(fig)

    cm_path = os.path.join(BASE_DIR, "aml_confusion_matrix.png")
    cm_plot_path = os.path.join(BASE_DIR, "aml_confusion_matrix_plot.png")

    risk_cm_path = os.path.join(BASE_DIR, "aml_risk_confusion_matrix.png")

    print("\n[5/5] Done! Open these files to view the graphs:")
    print(f"   1. {out_path}")
    print(f"   2. {cm_path}")
    print(f"   3. {cm_plot_path}")
    print(f"   4. {risk_cm_path}")
    if HAS_NX:
        print(f"   5. {out_path2}")
        print(f"   6. {out_path3}")


if __name__ == "__main__":
    main()
