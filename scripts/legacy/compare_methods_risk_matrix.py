"""
compare_methods_risk_matrix.py
==============================
Train two baseline methods (Random Forest & GraphSAGE) and compare
risk-classification confusion matrices against the existing GAT model.

Outputs:
  - data/rf_predictions.json
  - data/sage_predictions.json
  - data/gat_predictions.json   (re-generated for consistency)
  - data/risk_comparison.json
  - docs/images/risk_confusion_comparison.png
"""

import os
import json
import time
import gc
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DATA_DIR = "data"
MODEL_DIR = "data"
DOCS_IMG_DIR = "docs/images"
os.makedirs(DOCS_IMG_DIR, exist_ok=True)

# Paths
TRAIN_CSV = resolve("train.csv")
VAL_CSV   = resolve("val.csv")
TEST_CSV  = resolve("test.csv")
PROC_PATH = resolve("processed_data.pt")
GAT_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
SAGE_MODEL_PATH = os.path.join(MODEL_DIR, "best_sage_model.pt")

# ------------------------------------------------------------------
# Risk category helper (same as model.py)
# ------------------------------------------------------------------
def get_risk_category(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    elif prob < 0.70:
        return "Moderate"
    elif prob < 0.90:
        return "High"
    else:
        return "Critical"


def probs_to_risk_categories(probs: np.ndarray) -> np.ndarray:
    return np.array([get_risk_category(float(p)) for p in probs])


# ------------------------------------------------------------------
# 1. Random Forest Baseline (tabular)
# ------------------------------------------------------------------
def train_random_forest():
    print("\n" + "="*60)
    print(" BASELINE 1: Random Forest (Tabular)")
    print("="*60)

    print("[RF] Loading CSV data ...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    # Combine train+val for training (same as GAT uses train+val logic implicitly)
    train_df = pd.concat([train_df, val_df], ignore_index=True)

    cat_cols = [
        "Payment_currency", "Received_currency",
        "Sender_bank_location", "Receiver_bank_location", "Payment_type"
    ]
    num_cols = ["Amount"]
    feature_cols = cat_cols + num_cols

    # Encode categoricals
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[cat_cols] = encoder.fit_transform(train_df[cat_cols])
    test_df[cat_cols]  = encoder.transform(test_df[cat_cols])

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["Is_laundering"].values.astype(np.int64)
    X_test  = test_df[feature_cols].values.astype(np.float32)
    y_test  = test_df["Is_laundering"].values.astype(np.int64)

    print(f"[RF] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[RF] Class balance (train): {Counter(y_train)}")

    print("[RF] Training Random Forest ...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]

    # Save
    rf_out = {
        "probs": probs.tolist(),
        "labels": y_test.tolist(),
    }
    with open(resolve("rf_predictions.json"), "w") as f:
        json.dump(rf_out, f)

    print(f"[RF] Done. Avg prob: {probs.mean():.4f}")
    return probs, y_test


# ------------------------------------------------------------------
# 2. GraphSAGE Baseline (GNN)
# ------------------------------------------------------------------
class EdgeSAGEModel(nn.Module):
    """
    GraphSAGE-based edge classifier, architecturally aligned with EdgeGATModel
    so the comparison isolates the encoder difference (SAGE vs GAT).
    """
    def __init__(
        self,
        node_feat_dim: int = 7,
        edge_feat_dim: int = 8,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # SAGE Encoder (2 layers)
        self.sage1 = SAGEConv(node_feat_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Edge feature projection
        edge_proj_dim = 32
        self.edge_transform = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_proj_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Edge classifier
        classifier_input_dim = hidden_dim * 2 + edge_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.edge_chunk_size = 200000

    def forward(self, x, edge_index, edge_attr):
        h = self.sage1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.sage2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        src, dst = edge_index[0], edge_index[1]
        num_edges = src.shape[0]
        chunk = self.edge_chunk_size
        all_logits = []

        for i in range(0, num_edges, chunk):
            end = min(i + chunk, num_edges)
            src_emb = h[src[i:end]]
            dst_emb = h[dst[i:end]]
            edge_feat = self.edge_transform(edge_attr[i:end])
            edge_repr = torch.cat([src_emb, dst_emb, edge_feat], dim=1)
            logits = self.classifier(edge_repr).squeeze(-1)
            all_logits.append(logits)

        return torch.cat(all_logits)

    def predict(self, x, edge_index, edge_attr):
        logits = self.forward(x, edge_index, edge_attr)
        return torch.sigmoid(logits)


def get_balanced_edge_indices(y, mask, oversample_ratio=3.0):
    mask_indices = mask.nonzero(as_tuple=True)[0]
    labels = y[mask_indices]
    pos_indices = mask_indices[labels == 1]
    neg_indices = mask_indices[labels == 0]
    num_pos = len(pos_indices)
    if num_pos == 0:
        return neg_indices[torch.randperm(len(neg_indices))[:10000]]
    num_neg_sample = min(int(num_pos * oversample_ratio), len(neg_indices))
    repeat_count = max(num_neg_sample // num_pos, 1)
    pos_oversampled = pos_indices.repeat(repeat_count)
    neg_sampled = neg_indices[torch.randperm(len(neg_indices))[:num_neg_sample]]
    balanced = torch.cat([pos_oversampled, neg_sampled])
    return balanced[torch.randperm(len(balanced))]


def train_graphsage(data):
    print("\n" + "="*60)
    print(" BASELINE 2: GraphSAGE (GNN)")
    print("="*60)

    if os.path.exists(SAGE_MODEL_PATH):
        print("[SAGE] Found existing model. Loading ...")
        model = EdgeSAGEModel(
            node_feat_dim=data.x.shape[1],
            edge_feat_dim=data.edge_attr.shape[1],
            hidden_dim=64,
            dropout=0.3,
        ).to(DEVICE)
        model.load_state_dict(torch.load(SAGE_MODEL_PATH, weights_only=True, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            probs = model.predict(
                data.x.to(DEVICE), data.edge_index.to(DEVICE), data.edge_attr.to(DEVICE)
            )[data.test_mask].cpu().numpy()
        labels = data.y[data.test_mask].cpu().numpy()
        print(f"[SAGE] Loaded. Avg prob: {probs.mean():.4f}")
        return probs, labels

    print("[SAGE] No existing model found. Training from scratch ...")
    model = EdgeSAGEModel(
        node_feat_dim=data.x.shape[1],
        edge_feat_dim=data.edge_attr.shape[1],
        hidden_dim=64,
        dropout=0.3,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    epochs = 30
    oversample_ratio = 5.0
    best_val_f1 = 0.0
    best_state = None
    patience = 8
    patience_counter = 0

    print(f"[SAGE] Epochs={epochs}, patience={patience}")
    print(f"  {'Ep':>4} | {'T.Loss':>8} | {'V.F1':>6} | {'Time':>5}")
    print("  " + "-"*30)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.edge_attr.to(DEVICE))
        balanced_idx = get_balanced_edge_indices(data.y, data.train_mask, oversample_ratio)
        loss = criterion(logits[balanced_idx], data.y[balanced_idx].to(DEVICE).float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.edge_attr.to(DEVICE))
            val_probs = torch.sigmoid(val_logits[data.val_mask]).cpu().numpy()
            val_labels = data.y[data.val_mask].cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        elapsed = time.time() - t0
        if epoch <= 3 or epoch % 2 == 0:
            print(f"  {epoch:4d} | {loss.item():8.5f} | {val_f1:6.4f} | {elapsed:5.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best val F1={best_val_f1:.4f})")
                break

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), SAGE_MODEL_PATH)
    print(f"[SAGE] Best val F1: {best_val_f1:.4f}. Model saved.")

    model.eval()
    with torch.no_grad():
        test_logits = model(data.x.to(DEVICE), data.edge_index.to(DEVICE), data.edge_attr.to(DEVICE))
        probs = torch.sigmoid(test_logits[data.test_mask]).cpu().numpy()
    labels = data.y[data.test_mask].cpu().numpy()
    print(f"[SAGE] Training done. Avg prob: {probs.mean():.4f}")
    return probs, labels


# ------------------------------------------------------------------
# 3. GAT (existing model)
# ------------------------------------------------------------------
def load_gat_predictions(data):
    print("\n" + "="*60)
    print(" MODEL UTAMA: GAT (Graph Attention Network)")
    print("="*60)

    from backend.ml.legacy.model import EdgeGATModel
    model = EdgeGATModel(
        node_feat_dim=data.x.shape[1],
        edge_feat_dim=data.edge_attr.shape[1],
        hidden_dim=64,
        num_heads=4,
        dropout=0.0,
    ).to(DEVICE)
    model.load_state_dict(torch.load(GAT_MODEL_PATH, weights_only=True, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        probs = model.predict(
            data.x.to(DEVICE), data.edge_index.to(DEVICE), data.edge_attr.to(DEVICE)
        )[data.test_mask].cpu().numpy()
    labels = data.y[data.test_mask].cpu().numpy()
    print(f"[GAT] Loaded. Avg prob: {probs.mean():.4f}")
    return probs, labels


# ------------------------------------------------------------------
# 4. Evaluation Helpers
# ------------------------------------------------------------------
def build_cross_tab(labels_binary: np.ndarray, probs: np.ndarray):
    """Build 2x4 cross-tab: Actual (Normal, Laundering) x Predicted Risk Category."""
    risk_cats = probs_to_risk_categories(probs)
    tab = {"Normal": {}, "Laundering": {}}
    for cat in ["Low", "Moderate", "High", "Critical"]:
        tab["Normal"][cat] = int(((labels_binary == 0) & (risk_cats == cat)).sum())
        tab["Laundering"][cat] = int(((labels_binary == 1) & (risk_cats == cat)).sum())
    return tab


def build_pseudo_confusion_matrix(labels_binary: np.ndarray, probs: np.ndarray):
    """
    Build a 4x4 confusion matrix by assigning pseudo ground-truth risk categories:
      Normal      -> Low
      Laundering  -> Critical
    This is explicitly for visualization / thesis comparison purposes only.
    """
    gt_risk = np.where(labels_binary == 0, "Low", "Critical")
    pred_risk = probs_to_risk_categories(probs)
    cats = ["Low", "Moderate", "High", "Critical"]
    cm = np.zeros((4, 4), dtype=int)
    for i, actual in enumerate(cats):
        for j, pred in enumerate(cats):
            cm[i, j] = int(((gt_risk == actual) & (pred_risk == pred)).sum())
    return cm, cats


def print_cross_tab(tab, name):
    print(f"\n{name} — Cross-Tabulation (Actual vs Predicted Risk)")
    print("-" * 55)
    header = f"{'Actual':>12} | {'Low':>8} {'Moderate':>8} {'High':>8} {'Critical':>8}"
    print(header)
    print("-" * 55)
    for actual in ["Normal", "Laundering"]:
        row = f"{actual:>12} | {tab[actual]['Low']:>8,} {tab[actual]['Moderate']:>8,} {tab[actual]['High']:>8,} {tab[actual]['Critical']:>8,}"
        print(row)


def plot_comparison(cross_tabs, pseudo_cms, method_names):
    fig, axes = plt.subplots(2, len(method_names), figsize=(16, 10))
    fig.suptitle("Risk Classification Comparison: GAT vs Baselines", fontsize=16, y=1.02)

    cats = ["Low", "Moderate", "High", "Critical"]
    actual_labels = ["Normal", "Laundering"]

    for idx, (name, tab, cm) in enumerate(zip(method_names, cross_tabs, pseudo_cms)):
        # Top row: 2x4 cross-tab heatmap
        ax1 = axes[0, idx]
        mat = np.array([[tab["Normal"][c] for c in cats],
                        [tab["Laundering"][c] for c in cats]])
        sns.heatmap(mat, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=cats, yticklabels=actual_labels, ax=ax1, cbar=True)
        ax1.set_title(f"{name}\nActual vs Predicted Risk")
        ax1.set_xlabel("Predicted Risk Category")
        ax1.set_ylabel("Actual Class")

        # Bottom row: 4x4 pseudo confusion matrix
        ax2 = axes[1, idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cats, yticklabels=cats, ax=ax2, cbar=True)
        ax2.set_title(f"{name}\nPseudo 4-Class Confusion Matrix")
        ax2.set_xlabel("Predicted Risk Category")
        ax2.set_ylabel("Pseudo Ground-Truth Risk")

    plt.tight_layout()
    out_path = os.path.join(DOCS_IMG_DIR, "risk_confusion_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n[Plot] Saved to: {out_path}")
    plt.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("="*60)
    print("  Risk Confusion Matrix — Method Comparison")
    print("  GAT (Main) vs Random Forest vs GraphSAGE")
    print("="*60)

    # Load graph data once
    print("\n[Setup] Loading processed graph data ...")
    data = torch.load(PROC_PATH, weights_only=False)
    print(f"  Nodes: {data.num_nodes:,} | Edges: {data.edge_index.shape[1]:,}")
    print(f"  Test edges: {data.test_mask.sum().item():,}")

    # --- GAT ---
    gat_probs, gat_labels = load_gat_predictions(data)

    # --- Random Forest ---
    rf_probs, rf_labels = train_random_forest()

    # --- GraphSAGE ---
    sage_probs, sage_labels = train_graphsage(data)

    # Sanity check labels alignment
    assert np.array_equal(gat_labels, rf_labels), "Label mismatch between GAT and RF"
    assert np.array_equal(gat_labels, sage_labels), "Label mismatch between GAT and SAGE"
    y_test = gat_labels

    # Build cross-tabs
    methods = [
        ("GAT", gat_probs),
        ("Random Forest", rf_probs),
        ("GraphSAGE", sage_probs),
    ]

    cross_tabs = []
    pseudo_cms = []
    method_names = []

    print("\n" + "="*60)
    print(" EVALUATION: Risk Classification Cross-Tabs")
    print("="*60)

    for name, probs in methods:
        tab = build_cross_tab(y_test, probs)
        cm, cats = build_pseudo_confusion_matrix(y_test, probs)
        cross_tabs.append(tab)
        pseudo_cms.append(cm)
        method_names.append(name)
        print_cross_tab(tab, name)

        # Print pseudo-CM
        print(f"\n{name} — Pseudo 4-Class Confusion Matrix")
        print("-" * 45)
        print(f"{'':>12} | {'Low':>8} {'Mod':>8} {'High':>8} {'Crit':>8}")
        for i, c in enumerate(cats):
            print(f"{c:>12} | {cm[i,0]:>8,} {cm[i,1]:>8,} {cm[i,2]:>8,} {cm[i,3]:>8,}")

    # Save JSON
    comparison = {
        "methods": {}
    }
    for name, tab, cm in zip(method_names, cross_tabs, pseudo_cms):
        comparison["methods"][name] = {
            "cross_tab_2x4": tab,
            "pseudo_confusion_matrix_4x4": cm.tolist(),
        }

    out_json = resolve("risk_comparison.json")
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[Save] Comparison JSON: {out_json}")

    # Plot
    print("\n[Plot] Generating comparison figure ...")
    plot_comparison(cross_tabs, pseudo_cms, method_names)

    # Summary table
    print("\n" + "="*60)
    print(" SUMMARY: Critical Risk Detection (Laundering -> Critical)")
    print("="*60)
    print(f"{'Method':>18} | {'Critical':>10} | {'Total ML':>10} | {'Capture %':>10}")
    print("-" * 60)
    for name, tab in zip(method_names, cross_tabs):
        critical_ml = tab["Laundering"]["Critical"]
        total_ml = sum(tab["Laundering"].values())
        capture = critical_ml / total_ml * 100 if total_ml else 0
        print(f"{name:>18} | {critical_ml:>10,} | {total_ml:>10,} | {capture:>9.2f}%")

    print("\n[OK] Comparison complete!")


if __name__ == "__main__":
    main()
