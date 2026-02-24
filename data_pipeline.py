"""
data_pipeline.py — Step 1: Data Preprocessing & Graph Construction
==================================================================
Loads the SAML-D CSV, engineers node/edge features, and builds a
PyTorch Geometric Data object for edge-level AML classification.

Usage:
    python data_pipeline.py
"""

import os
import warnings
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SAML-D.csv")
OUTPUT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data.pt")
ENCODERS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "encoders.pkl")

CATEGORICAL_COLS = [
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
]

LABEL_COL = "Is_laundering"


# ─── 1. Load & Clean CSV ─────────────────────────────────────────────────────
def load_and_clean_csv(path: str) -> pd.DataFrame:
    """Load the CSV and parse Date+Time into a single datetime column."""
    print(f"[1/6] Loading CSV from {path} ...")
    df = pd.read_csv(path)
    print(f"      Raw shape: {df.shape}")

    # Combine Date + Time → datetime
    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    # Sort by datetime for temporal consistency
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Drop rows with missing critical fields
    critical_cols = ["Sender_account", "Receiver_account", "Amount", LABEL_COL]
    before = len(df)
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"      Dropped {dropped} rows with missing critical fields.")

    # Ensure types
    df["Sender_account"] = df["Sender_account"].astype(str)
    df["Receiver_account"] = df["Receiver_account"].astype(str)
    df["Amount"] = df["Amount"].astype(float)
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    print(f"      Cleaned shape: {df.shape}")
    return df


# ─── 2. Encode Categorical Features ──────────────────────────────────────────
def encode_categorical_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode categorical columns. Returns modified df + encoders dict."""
    print("[2/6] Encoding categorical features ...")
    encoders: Dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str).fillna("UNKNOWN"))
        encoders[col] = le
        print(f"      {col}: {len(le.classes_)} unique values")

    # Also encode Laundering_type (metadata — not used as input feature)
    le_lt = LabelEncoder()
    df["Laundering_type_encoded"] = le_lt.fit_transform(
        df["Laundering_type"].astype(str).fillna("UNKNOWN")
    )
    encoders["Laundering_type"] = le_lt

    return df, encoders


# ─── 3. Compute Temporal Edge Weights ─────────────────────────────────────────
def compute_temporal_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta_t (in minutes) from the earliest transaction, then
    edge weight: e_ij = Amount / (delta_t + 1).
    """
    print("[3/6] Computing temporal edge weights ...")
    reference_time = df["Datetime"].min()
    df["delta_t_minutes"] = (df["Datetime"] - reference_time).dt.total_seconds() / 60.0
    df["temporal_weight"] = df["Amount"] / (df["delta_t_minutes"] + 1.0)

    print(f"      Time span: {reference_time} → {df['Datetime'].max()}")
    print(f"      Temporal weight range: [{df['temporal_weight'].min():.4f}, {df['temporal_weight'].max():.4f}]")
    return df


# ─── 4. Build Node Features ──────────────────────────────────────────────────
def build_node_features(
    df: pd.DataFrame, node_map: Dict[str, int]
) -> torch.Tensor:
    """
    Aggregate per-account features:
      - total_sent, total_received
      - tx_count_sent, tx_count_received
      - unique_partners
      - foreign_currency_ratio
      - cross_border_ratio
    Returns a (num_nodes × 7) float tensor, StandardScaler-normalized.
    """
    print("[4/6] Building node features (7 per node) ...")
    num_nodes = len(node_map)
    node_feat = np.zeros((num_nodes, 7), dtype=np.float64)

    # Identify foreign-currency and cross-border transactions
    df["is_foreign_currency"] = (
        df["Payment_currency"].astype(str) != df["Received_currency"].astype(str)
    ).astype(int)
    df["is_cross_border"] = (
        df["Sender_bank_location"].astype(str) != df["Receiver_bank_location"].astype(str)
    ).astype(int)

    # ---- Sender-side aggregation ----
    sender_agg = df.groupby("Sender_account").agg(
        total_sent=("Amount", "sum"),
        tx_count_sent=("Amount", "count"),
        unique_receivers=("Receiver_account", "nunique"),
        foreign_sent=("is_foreign_currency", "sum"),
        cross_border_sent=("is_cross_border", "sum"),
    )

    for acc, row in sender_agg.iterrows():
        idx = node_map.get(str(acc))
        if idx is not None:
            node_feat[idx, 0] = row["total_sent"]
            node_feat[idx, 2] = row["tx_count_sent"]
            node_feat[idx, 4] += row["unique_receivers"]
            # ratios will be computed after both sides are done
            node_feat[idx, 5] += row["foreign_sent"]
            node_feat[idx, 6] += row["cross_border_sent"]

    # ---- Receiver-side aggregation ----
    receiver_agg = df.groupby("Receiver_account").agg(
        total_received=("Amount", "sum"),
        tx_count_received=("Amount", "count"),
        unique_senders=("Sender_account", "nunique"),
        foreign_recv=("is_foreign_currency", "sum"),
        cross_border_recv=("is_cross_border", "sum"),
    )

    for acc, row in receiver_agg.iterrows():
        idx = node_map.get(str(acc))
        if idx is not None:
            node_feat[idx, 1] = row["total_received"]
            node_feat[idx, 3] = row["tx_count_received"]
            node_feat[idx, 4] += row["unique_senders"]
            node_feat[idx, 5] += row["foreign_recv"]
            node_feat[idx, 6] += row["cross_border_recv"]

    # Convert foreign & cross-border counts → ratios
    total_tx_per_node = node_feat[:, 2] + node_feat[:, 3]  # sent + received
    total_tx_per_node = np.maximum(total_tx_per_node, 1.0)  # avoid div-by-zero
    node_feat[:, 5] /= total_tx_per_node  # foreign_currency_ratio
    node_feat[:, 6] /= total_tx_per_node  # cross_border_ratio

    # Normalize with StandardScaler
    scaler = StandardScaler()
    node_feat = scaler.fit_transform(node_feat)

    print(f"      Node feature matrix shape: ({num_nodes}, 7)")
    return torch.tensor(node_feat, dtype=torch.float32), scaler


# ─── 5. Build PyG Data Object ─────────────────────────────────────────────────
def build_pyg_data(
    df: pd.DataFrame, encoders: Dict[str, LabelEncoder]
) -> Tuple[Data, Dict]:
    """
    Build the complete PyTorch Geometric Data object.

    Returns:
        data: torch_geometric.data.Data with:
            - x             : (num_nodes, 7)   node features
            - edge_index    : (2, num_edges)   directed sender→receiver
            - edge_attr     : (num_edges, 8)   edge features
            - y             : (num_edges,)     labels (0/1)
        artifacts: dict with node_map, scaler, etc.
    """
    print("[5/6] Assembling PyG Data object ...")

    # ── Node mapping ──
    all_accounts = pd.concat(
        [df["Sender_account"], df["Receiver_account"]]
    ).unique()
    node_map: Dict[str, int] = {str(acc): i for i, acc in enumerate(all_accounts)}
    num_nodes = len(node_map)

    # ── Edge index ──
    src = df["Sender_account"].astype(str).map(node_map).values
    dst = df["Receiver_account"].astype(str).map(node_map).values
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # ── Edge features (8 features) ──
    # [Amount, temporal_weight,
    #  Payment_currency_enc, Received_currency_enc,
    #  Sender_bank_location_enc, Receiver_bank_location_enc,
    #  Payment_type_enc, delta_t_minutes]
    edge_feat_cols = [
        "Amount",
        "temporal_weight",
        "Payment_currency_encoded",
        "Received_currency_encoded",
        "Sender_bank_location_encoded",
        "Receiver_bank_location_encoded",
        "Payment_type_encoded",
        "delta_t_minutes",
    ]
    edge_attr_np = df[edge_feat_cols].values.astype(np.float32)

    # Normalize edge features
    edge_scaler = StandardScaler()
    edge_attr_np = edge_scaler.fit_transform(edge_attr_np)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float32)

    # ── Edge labels ──
    y = torch.tensor(df[LABEL_COL].values, dtype=torch.float32)

    # ── Node features ──
    node_features, node_scaler = build_node_features(df, node_map)

    # ── Train / Val / Test masks (edges) ──
    num_edges = edge_index.shape[1]
    perm = torch.randperm(num_edges)
    train_size = int(0.7 * num_edges)
    val_size = int(0.15 * num_edges)

    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)

    train_mask[perm[:train_size]] = True
    val_mask[perm[train_size : train_size + val_size]] = True
    test_mask[perm[train_size + val_size :]] = True

    # ── Assemble Data ──
    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.num_nodes = num_nodes

    artifacts = {
        "node_map": node_map,
        "node_scaler": node_scaler,
        "edge_scaler": edge_scaler,
        "encoders": encoders,
        "edge_feat_cols": edge_feat_cols,
    }

    print(f"      ✓ num_nodes  = {data.num_nodes:,}")
    print(f"      ✓ num_edges  = {data.edge_index.shape[1]:,}")
    print(f"      ✓ x shape    = {data.x.shape}")
    print(f"      ✓ edge_attr  = {data.edge_attr.shape}")
    print(f"      ✓ y shape    = {data.y.shape}")
    print(f"      ✓ train/val/test = {train_mask.sum().item():,} / {val_mask.sum().item():,} / {test_mask.sum().item():,}")

    return data, artifacts


# ─── 6. Print Dataset Statistics ──────────────────────────────────────────────
def print_dataset_stats(data: Data, df: pd.DataFrame):
    """Print comprehensive dataset statistics."""
    print("\n" + "=" * 60)
    print("   DATASET STATISTICS")
    print("=" * 60)

    num_edges = data.y.shape[0]
    num_pos = int(data.y.sum().item())
    num_neg = num_edges - num_pos
    ratio = num_neg / max(num_pos, 1)

    print(f"  Nodes (accounts)       : {data.num_nodes:,}")
    print(f"  Edges (transactions)   : {num_edges:,}")
    print(f"  Node feature dim       : {data.x.shape[1]}")
    print(f"  Edge feature dim       : {data.edge_attr.shape[1]}")
    print()
    print(f"  Class 0 (Normal)       : {num_neg:,}  ({num_neg/num_edges*100:.2f}%)")
    print(f"  Class 1 (Laundering)   : {num_pos:,}  ({num_pos/num_edges*100:.2f}%)")
    print(f"  Imbalance ratio (0:1)  : {ratio:.1f} : 1")
    print()

    # Split statistics
    for split_name, mask in [
        ("Train", data.train_mask),
        ("Val  ", data.val_mask),
        ("Test ", data.test_mask),
    ]:
        split_y = data.y[mask]
        split_pos = int(split_y.sum().item())
        split_neg = int(mask.sum().item()) - split_pos
        print(
            f"  {split_name}: {mask.sum().item():>8,} edges  "
            f"(Normal: {split_neg:,} | Laundering: {split_pos:,})"
        )

    print()
    print(f"  Unique Payment currencies     : {df['Payment_currency'].nunique()}")
    print(f"  Unique Received currencies    : {df['Received_currency'].nunique()}")
    print(f"  Unique Sender locations       : {df['Sender_bank_location'].nunique()}")
    print(f"  Unique Receiver locations     : {df['Receiver_bank_location'].nunique()}")
    print(f"  Unique Payment types          : {df['Payment_type'].nunique()}")
    print(f"  Unique Laundering types       : {df['Laundering_type'].nunique()}")
    print(f"  Date range                    : {df['Date'].min()} → {df['Date'].max()}")
    print("=" * 60)


# ─── 7. Main Pipeline ─────────────────────────────────────────────────────────
def main():
    """Run the complete data preprocessing and graph construction pipeline."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AML Detection — Data Pipeline (Step 1)                 ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # Step 1: Load CSV
    df = load_and_clean_csv(CSV_PATH)

    # Step 2: Encode categoricals
    df, encoders = encode_categorical_features(df)

    # Step 3: Temporal weights
    df = compute_temporal_weight(df)

    # Step 4-5: Build PyG Data
    data, artifacts = build_pyg_data(df, encoders)

    # Step 6: Statistics
    print_dataset_stats(data, df)

    # Save outputs
    print(f"\n[6/6] Saving outputs ...")
    torch.save(data, OUTPUT_DATA_PATH)
    print(f"      ✓ PyG Data saved to: {OUTPUT_DATA_PATH}")

    joblib.dump(artifacts, ENCODERS_PATH)
    print(f"      ✓ Encoders & scalers saved to: {ENCODERS_PATH}")

    print("\n✅ Pipeline complete! Ready for Step 2 (GAT Model Development).\n")
    return data, artifacts


if __name__ == "__main__":
    main()
