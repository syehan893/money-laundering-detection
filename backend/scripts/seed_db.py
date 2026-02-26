"""
seed_db.py — Populate MongoDB from Processed Data (Legacy)
====================================================
Loads CSV, runs model inference on all edges, and batch-inserts
accounts + transactions + metrics into MongoDB.

NOTE: This is now a legacy script. The main flow uses
      train.py which auto-populates MongoDB after training.

Usage:
    python -m backend.scripts.seed_db
"""

import asyncio
import json
import os
import sys
import time

import certifi
import joblib
import numpy as np
import pandas as pd
import torch
from motor.motor_asyncio import AsyncIOMotorClient

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.ml.model import EdgeGATModel, get_risk_category
from backend.config import CSV_PATH, DATA_PATH, MODEL_PATH, ENCODERS_PATH, METRICS_PATH

MONGO_URL = os.getenv(
    "MONGO_URL",
    "mongodb+srv://syehanart:bSmsqyodB8crjqkv@cluster0.yo9ng.mongodb.net/?appName=Cluster0&tls=true&tlsInsecure=true",
)
DB_NAME = os.getenv("MONGO_DB", "aml_detection")
BATCH_SIZE = 5000  # Insert batch size


async def seed():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AML Detection — MongoDB Seeder                         ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Connect ──
    print("[1/6] Connecting to MongoDB ...")
    client = AsyncIOMotorClient(MONGO_URL)
    db = client[DB_NAME]
    info = await client.server_info()
    print(f"  ✓ Connected (MongoDB v{info.get('version', '?')})")

    # ── Load data ──
    print("\n[2/6] Loading data ...")
    df = pd.read_csv(CSV_PATH)
    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    print(f"  CSV: {len(df):,} rows")

    data = torch.load(DATA_PATH, weights_only=False, map_location="cpu")
    print(f"  Graph: {data.num_nodes:,} nodes, {data.edge_index.shape[1]:,} edges")

    artifacts = joblib.load(ENCODERS_PATH)
    node_map = artifacts["node_map"]
    node_map_inv = {v: k for k, v in node_map.items()}

    # ── Run model inference ──
    print("\n[3/6] Running model inference on all edges ...")
    model = EdgeGATModel(node_feat_dim=7, edge_feat_dim=8, hidden_dim=64, num_heads=4, dropout=0.0)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        preds = model(data.x, data.edge_index, data.edge_attr)
    preds_np = preds.cpu().numpy()
    print(f"  ✓ Predictions: {len(preds_np):,} edges")

    # ── Align predictions with CSV rows ──
    df_sorted = df.copy()
    df_sorted["_sort_dt"] = pd.to_datetime(
        df_sorted["Date"].astype(str) + " " + df_sorted["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S", errors="coerce",
    )
    df_sorted = df_sorted.sort_values("_sort_dt").reset_index(drop=True)

    num_preds = len(preds_np)
    num_rows = len(df_sorted)

    if num_preds <= num_rows:
        df_sorted = df_sorted.iloc[:num_preds].copy()
        df_sorted["prediction_probability"] = preds_np
    else:
        df_sorted["prediction_probability"] = preds_np[:num_rows]

    df_sorted["prediction_risk_category"] = df_sorted["prediction_probability"].apply(
        lambda p: get_risk_category(float(p))
    )

    print(f"  ✓ Aligned {len(df_sorted):,} transactions with predictions")

    # ── Build account profiles ──
    print("\n[4/6] Building account profiles ...")
    accounts = {}

    for acc, grp in df_sorted.groupby("Sender_account"):
        acc_str = str(acc)
        if acc_str not in accounts:
            accounts[acc_str] = {
                "account_id": acc_str,
                "total_sent": 0, "total_received": 0,
                "tx_count_sent": 0, "tx_count_received": 0,
                "unique_partners": set(),
                "foreign_tx": 0, "cross_border_tx": 0, "total_tx": 0,
                "max_risk_prob": 0.0, "risk_probs": [],
            }
        a = accounts[acc_str]
        a["total_sent"] += grp["Amount"].sum()
        a["tx_count_sent"] += len(grp)
        a["unique_partners"].update(grp["Receiver_account"].astype(str).unique())
        a["foreign_tx"] += int((grp["Payment_currency"] != grp["Received_currency"]).sum())
        a["cross_border_tx"] += int((grp["Sender_bank_location"] != grp["Receiver_bank_location"]).sum())
        a["total_tx"] += len(grp)
        a["risk_probs"].extend(grp["prediction_probability"].tolist())

    for acc, grp in df_sorted.groupby("Receiver_account"):
        acc_str = str(acc)
        if acc_str not in accounts:
            accounts[acc_str] = {
                "account_id": acc_str,
                "total_sent": 0, "total_received": 0,
                "tx_count_sent": 0, "tx_count_received": 0,
                "unique_partners": set(),
                "foreign_tx": 0, "cross_border_tx": 0, "total_tx": 0,
                "max_risk_prob": 0.0, "risk_probs": [],
            }
        a = accounts[acc_str]
        a["total_received"] += grp["Amount"].sum()
        a["tx_count_received"] += len(grp)
        a["unique_partners"].update(grp["Sender_account"].astype(str).unique())
        a["total_tx"] += len(grp)
        a["risk_probs"].extend(grp["prediction_probability"].tolist())

    account_docs = []
    for acc_id, a in accounts.items():
        total_tx = max(a["total_tx"], 1)
        avg_risk = float(np.mean(a["risk_probs"])) if a["risk_probs"] else 0.0
        max_risk = float(np.max(a["risk_probs"])) if a["risk_probs"] else 0.0

        account_docs.append({
            "account_id": acc_id,
            "risk_category": get_risk_category(max_risk),
            "risk_score": round(max_risk * 100, 2),
            "avg_risk_score": round(avg_risk * 100, 2),
            "total_sent": round(float(a["total_sent"]), 2),
            "total_received": round(float(a["total_received"]), 2),
            "tx_count_sent": int(a["tx_count_sent"]),
            "tx_count_received": int(a["tx_count_received"]),
            "unique_partners": len(a["unique_partners"]),
            "foreign_currency_ratio": round(a["foreign_tx"] / total_tx, 4),
            "cross_border_ratio": round(a["cross_border_tx"] / total_tx, 4),
        })

    print(f"  ✓ {len(account_docs):,} accounts profiled")

    cat_counts = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
    for doc in account_docs:
        cat_counts[doc["risk_category"]] += 1
    for cat, cnt in cat_counts.items():
        print(f"    {cat}: {cnt:,}")

    # ── Insert into MongoDB ──
    print("\n[5/6] Inserting into MongoDB ...")

    await db.accounts.drop()
    await db.transactions.drop()
    await db.model_metrics.drop()

    print(f"  Inserting {len(account_docs):,} accounts ...")
    for i in range(0, len(account_docs), BATCH_SIZE):
        batch = account_docs[i:i + BATCH_SIZE]
        await db.accounts.insert_many(batch)
    print(f"  ✓ accounts inserted")

    print(f"  Inserting {len(df_sorted):,} transactions ...")
    tx_docs = []
    for _, row in df_sorted.iterrows():
        tx_docs.append({
            "sender_account": str(row["Sender_account"]),
            "receiver_account": str(row["Receiver_account"]),
            "amount": float(row["Amount"]),
            "payment_currency": str(row["Payment_currency"]),
            "received_currency": str(row["Received_currency"]),
            "sender_bank_location": str(row["Sender_bank_location"]),
            "receiver_bank_location": str(row["Receiver_bank_location"]),
            "payment_type": str(row["Payment_type"]),
            "datetime": row["Datetime"].isoformat() if pd.notna(row["Datetime"]) else None,
            "date": str(row["Date"]),
            "time": str(row["Time"]),
            "is_laundering": int(row["Is_laundering"]),
            "laundering_type": str(row.get("Laundering_type", "Unknown")),
            "prediction_probability": round(float(row["prediction_probability"]), 6),
            "prediction_risk_category": str(row["prediction_risk_category"]),
        })

        if len(tx_docs) >= BATCH_SIZE:
            await db.transactions.insert_many(tx_docs)
            tx_docs = []

    if tx_docs:
        await db.transactions.insert_many(tx_docs)
    print(f"  ✓ transactions inserted")

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    metrics["_id"] = "latest"
    await db.model_metrics.insert_one(metrics)
    print(f"  ✓ model_metrics inserted")

    # ── Create indexes ──
    print("\n[6/6] Creating indexes ...")
    await db.accounts.create_index("account_id", unique=True)
    await db.accounts.create_index("risk_category")
    await db.accounts.create_index("risk_score")
    await db.transactions.create_index("sender_account")
    await db.transactions.create_index("receiver_account")
    await db.transactions.create_index("prediction_risk_category")
    print("  ✓ Indexes created")

    acc_count = await db.accounts.count_documents({})
    tx_count = await db.transactions.count_documents({})
    print(f"\n{'=' * 50}")
    print(f"  MongoDB Seeded Successfully!")
    print(f"  accounts:      {acc_count:,}")
    print(f"  transactions:  {tx_count:,}")
    print(f"  model_metrics: 1")
    print(f"{'=' * 50}")

    client.close()
    print("\n✅ Seeding complete!\n")


if __name__ == "__main__":
    asyncio.run(seed())
