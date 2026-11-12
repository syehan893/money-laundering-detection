"""
show_preprocessing.py — Comprehensive Data Preprocessing Report
================================================================
Displays detailed before/after comparison of all preprocessing steps.

Usage:
    python -m backend.scripts.show_preprocessing
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import CSV_PATH, DATA_PATH


def print_section(title):
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)


def show_raw_stats(df_raw: pd.DataFrame):
    """Show detailed statistics of raw CSV before any cleaning."""
    print_section("1. RAW DATA — BEFORE PREPROCESSING")
    print(f"  File path            : {CSV_PATH}")
    print(f"  Shape                : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    print(f"  Memory usage         : {df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print()

    # Missing values
    print("  -- Missing Values --")
    missing = df_raw.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        for col, count in missing.items():
            if count > 0:
                pct = count / len(df_raw) * 100
                print(f"    {col:35s}: {count:>8,}  ({pct:6.3f}%)")
        print(f"    {'TOTAL':35s}: {total_missing:>8,}")
    else:
        print("    (none — no missing values detected)")

    print()
    print("  -- Duplicate Rows --")
    dup_count = df_raw.duplicated().sum()
    dup_pct = dup_count / len(df_raw) * 100
    print(f"    Exact duplicates     : {dup_count:,}  ({dup_pct:.3f}%)")
    # Check duplicates excluding Time (same transaction at different seconds)
    cols_no_time = [c for c in df_raw.columns if c != "Time"]
    dup_no_time = df_raw.duplicated(subset=cols_no_time).sum()
    print(f"    Duplicates (no Time) : {dup_no_time:,}  ({dup_no_time/len(df_raw)*100:.3f}%)")

    print()
    print("  -- Data Types --")
    for col, dtype in df_raw.dtypes.items():
        print(f"    {col:35s}: {dtype}")

    print()
    print("  -- Numeric Column Statistics --")
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"    {col:35s}: min={df_raw[col].min():>12,.2f}  max={df_raw[col].max():>12,.2f}  mean={df_raw[col].mean():>12,.2f}")

    print()
    print("  -- Categorical Column Cardinality --")
    cat_cols = ["Payment_currency", "Received_currency", "Sender_bank_location",
                "Receiver_bank_location", "Payment_type", "Laundering_type"]
    for col in cat_cols:
        if col in df_raw.columns:
            unique = df_raw[col].nunique(dropna=False)
            print(f"    {col:35s}: {unique:>6,} unique values")

    print()
    print("  -- Class Distribution --")
    if "Is_laundering" in df_raw.columns:
        pos = int(df_raw["Is_laundering"].sum())
        neg = len(df_raw) - pos
        total = len(df_raw)
        print(f"    Normal (0)           : {neg:>12,}  ({neg/total*100:6.2f}%)")
        print(f"    Laundering (1)       : {pos:>12,}  ({pos/total*100:6.2f}%)")
        print(f"    Imbalance ratio      : {neg/max(pos,1):.1f} : 1")

    print()
    print("  -- Date Range --")
    if "Date" in df_raw.columns:
        print(f"    From                 : {df_raw['Date'].min()}")
        print(f"    To                   : {df_raw['Date'].max()}")
        print(f"    Span                 : {df_raw['Date'].nunique()} unique dates")

    print()
    print("  -- Sample Raw Rows (first 3) --")
    print(df_raw.head(3).to_string())


def show_after_cleaning(df_raw: pd.DataFrame, dropped: int):
    """Show what happened after cleaning step."""
    print_section("2. AFTER CLEANING — MISSING VALUES & TYPE CORRECTION")
    before = len(df_raw) + dropped
    after = len(df_raw)
    print(f"  Rows before          : {before:,}")
    print(f"  Rows dropped         : {dropped:,}  ({dropped/before*100:.3f}%)")
    print(f"  Rows after           : {after:,}")
    print()
    print("  Actions performed:")
    print("    [OK] Parsed Date + Time -> unified Datetime column")
    print("    [OK] Sorted entire dataset by datetime (temporal ordering)")
    print("    [OK] Dropped rows with missing critical fields:")
    print("      • Sender_account")
    print("      • Receiver_account")
    print("      • Amount")
    print("      • Is_laundering")
    print("    [OK] Cast Sender_account / Receiver_account -> string")
    print("    [OK] Cast Amount -> float")
    print("    [OK] Cast Is_laundering -> int (0 or 1)")
    print()
    print("  Result: No missing values remain in critical columns.")


def show_after_encoding(encoders: dict):
    """Show encoding results."""
    print_section("3. AFTER CATEGORICAL ENCODING")
    print("  Method: sklearn.LabelEncoder (fit on entire dataset)")
    print()
    print("  Encoded columns:")
    for col, le in encoders.items():
        print(f"    {col:35s}: {len(le.classes_):>5} classes  ->  column '{col}_encoded'")
    print()
    print("  Sample encoding mapping (first 5 per column):")
    for col, le in encoders.items():
        sample = list(zip(le.classes_[:5], range(5)))
        mapping_str = ", ".join([f"'{k}'->{v}" for k, v in sample])
        print(f"    {col:20s}: {mapping_str}{'...' if len(le.classes_) > 5 else ''}")


def show_after_temporal(df: pd.DataFrame):
    """Show temporal feature engineering results."""
    print_section("4. AFTER TEMPORAL FEATURE ENGINEERING")
    print("  New columns created:")
    print("    • delta_t_minutes  : time difference in minutes from earliest transaction")
    print("    • temporal_weight  : Amount / (delta_t_minutes + 1)")
    print()
    print(f"  Reference time       : {df['Datetime'].min()}")
    print(f"  Latest time          : {df['Datetime'].max()}")
    print(f"  Total span           : {df['delta_t_minutes'].max():,.0f} minutes  ({df['delta_t_minutes'].max()/1440:.1f} days)")
    print()
    print("  delta_t_minutes stats:")
    print(f"    min = {df['delta_t_minutes'].min():>12,.2f}")
    print(f"    max = {df['delta_t_minutes'].max():>12,.2f}")
    print(f"    mean = {df['delta_t_minutes'].mean():>12,.2f}")
    print()
    print("  temporal_weight stats:")
    print(f"    min = {df['temporal_weight'].min():>12,.6f}")
    print(f"    max = {df['temporal_weight'].max():>12,.6f}")
    print(f"    mean = {df['temporal_weight'].mean():>12,.6f}")


def show_after_graph(data, artifacts):
    """Show final graph construction results."""
    print_section("5. AFTER GRAPH CONSTRUCTION — FINAL OUTPUT")
    print("  Tensors:")
    print(f"    x (node features)      : {data.x.shape}  -> {data.num_nodes:,} accounts × 7 features")
    print(f"    edge_index             : {data.edge_index.shape}  -> {data.edge_index.shape[1]:,} directed edges")
    print(f"    edge_attr (edge feats) : {data.edge_attr.shape}  -> {data.edge_attr.shape[1]:,} edges × 8 features")
    print(f"    y (labels)             : {data.y.shape}  -> 0=Normal, 1=Laundering")
    print()
    print("  Node Features (7 dims) — StandardScaler normalized:")
    print("    0. total_sent              — total amount sent by account")
    print("    1. total_received          — total amount received by account")
    print("    2. tx_count_sent           — number of outgoing transactions")
    print("    3. tx_count_received       — number of incoming transactions")
    print("    4. unique_partners         — number of unique counterparties")
    print("    5. foreign_currency_ratio  — % of transactions with different currency")
    print("    6. cross_border_ratio      — % of transactions across different countries")
    print("    [WARN] Computed ONLY from training edges to prevent data leakage")
    print()
    print("  Edge Features (8 dims) — StandardScaler fit on train, transform all:")
    print("    0. Amount")
    print("    1. temporal_weight")
    print("    2. Payment_currency_encoded")
    print("    3. Received_currency_encoded")
    print("    4. Sender_bank_location_encoded")
    print("    5. Receiver_bank_location_encoded")
    print("    6. Payment_type_encoded")
    print("    7. delta_t_minutes")
    print()
    print("  Train / Val / Test Split:")
    print(f"    Train mask: {data.train_mask.sum().item():>10,} edges  ({data.train_mask.sum().item()/len(data.y)*100:5.1f}%)")
    print(f"    Val mask  : {data.val_mask.sum().item():>10,} edges  ({data.val_mask.sum().item()/len(data.y)*100:5.1f}%)")
    print(f"    Test mask : {data.test_mask.sum().item():>10,} edges  ({data.test_mask.sum().item()/len(data.y)*100:5.1f}%)")
    print()
    print("  Class distribution per split:")
    for name, mask in [("Train", data.train_mask), ("Val", data.val_mask), ("Test", data.test_mask)]:
        split_y = data.y[mask]
        pos = int(split_y.sum().item())
        neg = int(mask.sum().item()) - pos
        print(f"    {name:6s}: Normal={neg:>10,}  Laundering={pos:>8,}")
    print()
    print("  Output files:")
    print(f"    • processed_data.pt  -> PyTorch Geometric Data object")
    print(f"    • encoders.pkl       -> LabelEncoders, node_map, node_scaler, edge_scaler")


def show_full_summary(df_raw, dropped, data, artifacts):
    """Display comprehensive before vs after table."""
    print_section("6. COMPREHENSIVE SUMMARY — BEFORE vs AFTER")

    before_rows = len(df_raw)
    after_rows = data.edge_index.shape[1]
    after_nodes = data.num_nodes

    print(f"  {'Category':<40} {'Before':>16} {'After':>16}")
    print("  " + "-" * 74)
    print(f"  {'Rows (transactions)':<40} {before_rows:>16,} {after_rows:>16,}")
    print(f"  {'Columns / raw attributes':<40} {len(df_raw.columns):>16} {'—':>16}")
    print(f"  {'Accounts (unique nodes)':<40} {'—':>16} {after_nodes:>16,}")
    print(f"  {'Graph edges (directed)':<40} {'—':>16} {after_rows:>16,}")
    print(f"  {'Missing values':<40} {df_raw.isnull().sum().sum():>16,} {'0':>16}")
    print(f"  {'Duplicate rows (exact)':<40} {df_raw.duplicated().sum():>16,} {'—':>16}")
    print(f"  {'Node feature dimensions':<40} {'—':>16} {'7':>16}")
    print(f"  {'Edge feature dimensions':<40} {'—':>16} {'8':>16}")
    print(f"  {'Data format':<40} {'Tabular CSV':>16} {'PyG Data Object':>16}")
    print(f"  {'Split strategy':<40} {'—':>16} {'Temporal (time-based)':>16}")
    print(f"  {'Data leakage prevention':<40} {'—':>16} {'Yes (train-only fit)':>16}")
    print()

    # Amount statistics comparison
    if "Amount" in df_raw.columns:
        print("  -- Amount Statistics --")
        print(f"  {'Metric':<40} {'Before':>16} {'After (scaled)':>16}")
        print("  " + "-" * 74)
        print(f"  {'Minimum':<40} {df_raw['Amount'].min():>16,.2f} {'—':>16}")
        print(f"  {'Maximum':<40} {df_raw['Amount'].max():>16,.2f} {'—':>16}")
        print(f"  {'Mean':<40} {df_raw['Amount'].mean():>16,.2f} {'—':>16}")
        print(f"  {'Median':<40} {df_raw['Amount'].median():>16,.2f} {'—':>16}")
        print(f"  {'Std Dev':<40} {df_raw['Amount'].std():>16,.2f} {'—':>16}")
        print()

    # Categorical comparison
    print("  -- Categorical Encodings --")
    print(f"  {'Column':<40} {'Unique (raw)':>16} {'Encoded range':>16}")
    print("  " + "-" * 74)
    for col, le in artifacts.get("encoders", {}).items():
        if col in df_raw.columns:
            raw_unique = df_raw[col].nunique(dropna=False)
            print(f"  {col:<40} {raw_unique:>16,} {f'0..{len(le.classes_)-1}':>16}")


def main():
    print("+" + "-" * 74 + "+")
    print("|  AML Guard — Comprehensive Data Preprocessing Report                     |")
    print("|  Before vs After Comparison (Step-by-Step)                               |")
    print("+" + "-" * 74 + "+")

    # Check prerequisites
    if not os.path.exists(DATA_PATH):
        print("\n  [ERROR] processed_data.pt not found!")
        print("     Run: python -m backend.ml.legacy.data_pipeline")
        return

    import torch
    import joblib

    data = torch.load(DATA_PATH, weights_only=False)
    artifacts = joblib.load(os.path.join(os.path.dirname(DATA_PATH), "encoders.pkl"))

    # Load raw CSV
    csv_path = CSV_PATH
    if not os.path.exists(csv_path):
        alt = os.path.join(os.path.dirname(csv_path), "SAML-D.csv")
        csv_path = alt if os.path.exists(alt) else None

    if csv_path and os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path)
        show_raw_stats(df_raw)

        critical_cols = ["Sender_account", "Receiver_account", "Amount", "Is_laundering"]
        dropped = df_raw[critical_cols].isnull().any(axis=1).sum()
        show_after_cleaning(df_raw, dropped)
    else:
        print("\n  [WARN] Raw CSV not found — showing processed data only.")
        df_raw = None
        dropped = 0

    show_after_encoding(artifacts.get("encoders", {}))

    # Try to load cleaned df for temporal stats
    if df_raw is not None:
        df_temp = df_raw.copy()
        df_temp["Datetime"] = pd.to_datetime(
            df_temp["Date"].astype(str) + " " + df_temp["Time"].astype(str),
            format="%Y-%m-%d %H:%M:%S", errors="coerce",
        )
        df_temp = df_temp.sort_values("Datetime").reset_index(drop=True)
        df_temp = df_temp.dropna(subset=["Sender_account", "Receiver_account", "Amount", "Is_laundering"])
        reference_time = df_temp["Datetime"].min()
        df_temp["delta_t_minutes"] = (df_temp["Datetime"] - reference_time).dt.total_seconds() / 60.0
        df_temp["temporal_weight"] = df_temp["Amount"] / (df_temp["delta_t_minutes"] + 1.0)
        show_after_temporal(df_temp)

    show_after_graph(data, artifacts)

    if df_raw is not None:
        show_full_summary(df_raw, dropped, data, artifacts)

    print("\n[DONE] Preprocessing report complete!\n")


if __name__ == "__main__":
    main()
