"""
populate_mongodb.py — Populate MongoDB Without Training
=======================================================
Use trained model + processed data to populate MongoDB collections:
  - transactions
  - accounts
  - training_metrics (optional)

Usage:
    python -m backend.scripts.populate_mongodb
"""

import gc
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.ml.model import EdgeGATModel, get_risk_category
from backend.config import (
    DATA_PATH, MODEL_PATH, METRICS_PATH, CSV_PATH,
    TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH,
)
from backend.database import get_sync_db, setup_sync_indexes, close_sync_db


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(0.95, 0)
    else:
        device = torch.device("cpu")
    return device


def populate():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Populate MongoDB from Trained Model                     ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Check prerequisites ──
    if not os.path.exists(DATA_PATH):
        print(f"  ❌ {DATA_PATH} not found! Run data_pipeline.py first.")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ {MODEL_PATH} not found! Train the model first.")
        return

    # ── Load data ──
    print("[1/4] Loading processed data ...")
    data = torch.load(DATA_PATH, weights_only=False)
    print(f"  Nodes: {data.num_nodes:,} | Edges: {data.edge_index.shape[1]:,}")

    # ── Load model ──
    print("\n[2/4] Loading model ...")
    device = get_device()
    model = EdgeGATModel(
        node_feat_dim=data.x.shape[1],
        edge_feat_dim=data.edge_attr.shape[1],
        hidden_dim=64,
        num_heads=4,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print(f"  ✓ Model loaded: {MODEL_PATH}")

    # ── Run inference ──
    print("\n[3/4] Running inference ...")
    with torch.no_grad():
        preds = model.predict(
            data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        )
    preds_np = preds.cpu().numpy()
    print(f"  ✓ Predictions: {len(preds_np):,} edges")

    # ── Load CSV ──
    print("\n[4/4] Populating MongoDB ...")
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        dfs = []
        for split_path in [TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH]:
            if os.path.exists(split_path):
                dfs.append(pd.read_csv(split_path))
        if not dfs:
            print(f"  ❌ No CSV data found!")
            return
        df = pd.concat(dfs, ignore_index=True)

    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S", errors="coerce",
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    num_preds = len(preds_np)
    df = df.iloc[:num_preds].copy()
    df["prediction_probability"] = preds_np
    df["prediction_risk_category"] = df["prediction_probability"].apply(
        lambda p: get_risk_category(float(p))
    )

    # Normalize column names
    df["sender_account"] = df["Sender_account"].astype(str)
    df["receiver_account"] = df["Receiver_account"].astype(str)
    df["amount"] = df["Amount"]
    df["payment_currency"] = df["Payment_currency"]
    df["received_currency"] = df["Received_currency"]
    df["sender_bank_location"] = df["Sender_bank_location"]
    df["receiver_bank_location"] = df["Receiver_bank_location"]
    df["payment_type"] = df["Payment_type"]
    df["is_laundering"] = df["Is_laundering"].astype(int)

    # ── Build account profiles ──
    print("  🔄 Building account profiles ...")
    accounts = {}

    def _new_account(acc_id):
        return {
            "account_id": acc_id,
            "risk_score": 0.0,
            "risk_category": "Low",
            "avg_risk_score": 0.0,
            "total_sent": 0.0,
            "total_received": 0.0,
            "tx_count_sent": 0,
            "tx_count_received": 0,
            "unique_partners": set(),
            "foreign_tx": 0,
            "cross_border_tx": 0,
            "total_tx": 0,
            "risk_probs": [],
        }

    for acc, grp in df.groupby("sender_account"):
        if acc not in accounts:
            accounts[acc] = _new_account(acc)
        a = accounts[acc]
        a["total_sent"] += float(grp["amount"].sum())
        a["tx_count_sent"] += len(grp)
        a["unique_partners"].update(grp["receiver_account"].unique())
        a["foreign_tx"] += int((grp["payment_currency"] != grp["received_currency"]).sum())
        a["cross_border_tx"] += int((grp["sender_bank_location"] != grp["receiver_bank_location"]).sum())
        a["total_tx"] += len(grp)
        a["risk_probs"].extend(grp["prediction_probability"].tolist())

    for acc, grp in df.groupby("receiver_account"):
        if acc not in accounts:
            accounts[acc] = _new_account(acc)
        a = accounts[acc]
        a["total_received"] += float(grp["amount"].sum())
        a["tx_count_received"] += len(grp)
        a["unique_partners"].update(grp["sender_account"].unique())
        a["total_tx"] += len(grp)
        a["risk_probs"].extend(grp["prediction_probability"].tolist())

    # Finalize accounts
    for acc_id, a in accounts.items():
        total_tx = max(a["total_tx"], 1)
        probs = a["risk_probs"]
        avg_risk = float(np.mean(probs)) if probs else 0.0
        max_risk = float(np.max(probs)) if probs else 0.0
        a["risk_category"] = get_risk_category(max_risk)
        a["risk_score"] = round(max_risk * 100, 2)
        a["avg_risk_score"] = round(avg_risk * 100, 2)
        a["unique_partners"] = len(a["unique_partners"])
        a["foreign_currency_ratio"] = round(a["foreign_tx"] / total_tx, 4)
        a["cross_border_ratio"] = round(a["cross_border_tx"] / total_tx, 4)
        a["total_sent"] = round(a["total_sent"], 2)
        a["total_received"] = round(a["total_received"], 2)
        del a["risk_probs"]
        del a["foreign_tx"]
        del a["cross_border_tx"]
        del a["total_tx"]

    # ── Insert to MongoDB ──
    db = get_sync_db()
    setup_sync_indexes(db)

    # Transactions
    print("  🔄 Inserting transactions into MongoDB ...")
    db.transactions.drop()
    tx_cols = [
        "sender_account", "receiver_account", "amount",
        "payment_currency", "received_currency",
        "sender_bank_location", "receiver_bank_location",
        "payment_type", "datetime", "is_laundering",
        "prediction_probability", "prediction_risk_category",
    ]
    tx_docs = []
    for _, row in df[tx_cols].iterrows():
        doc = row.to_dict()
        if pd.notna(doc.get("datetime")):
            doc["datetime"] = str(doc["datetime"])
        else:
            doc["datetime"] = None
        doc["prediction_probability"] = round(float(doc["prediction_probability"]), 6)
        tx_docs.append(doc)

    batch_size = 10000
    for i in range(0, len(tx_docs), batch_size):
        batch = tx_docs[i:i + batch_size]
        db.transactions.insert_many(batch, ordered=False)
        print(f"    Inserted {min(i + batch_size, len(tx_docs)):,} / {len(tx_docs):,} transactions")
    print(f"  ✓ Transactions: {len(tx_docs):,} documents")

    # Accounts
    print("  🔄 Inserting accounts into MongoDB ...")
    db.accounts.drop()
    acc_docs = list(accounts.values())
    for i in range(0, len(acc_docs), batch_size):
        batch = acc_docs[i:i + batch_size]
        db.accounts.insert_many(batch, ordered=False)
        print(f"    Inserted {min(i + batch_size, len(acc_docs)):,} / {len(acc_docs):,} accounts")
    print(f"  ✓ Accounts: {len(acc_docs):,} documents")

    # Training metrics (if available)
    if os.path.exists(METRICS_PATH):
        print("  🔄 Inserting training metrics ...")
        with open(METRICS_PATH) as f:
            metrics_doc = json.load(f)
        metrics_doc["run_id"] = datetime.now().isoformat()
        metrics_doc["created_at"] = datetime.now()
        # Sanitize NaN values
        for h in metrics_doc.get("training_history", []):
            for k, v in h.items():
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    h[k] = None
        db.training_metrics.insert_one(metrics_doc)
        print(f"  ✓ Training metrics: 1 document")

    setup_sync_indexes(db)
    close_sync_db()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("\n✅ MongoDB population complete!\n")


if __name__ == "__main__":
    populate()
