"""
main.py — FastAPI Backend for AML Detection (No MongoDB)
=========================================================
REST API that loads data in-memory from local files (CSV, .pt, .json)
and serves predictions, account history, and summary endpoints.

Endpoints:
    POST /api/predict                 - Predict laundering risk
    GET  /api/metrics                 - Model performance metrics
    GET  /api/graph-stats             - Graph statistics
    GET  /api/health                  - Health check
    GET  /api/accounts/{account_id}   - Account history + graph data
    GET  /api/accounts                - List accounts by risk category
    GET  /api/summary                 - Training data summary

Usage:
    python main.py
"""

import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import EdgeGATModel, get_risk_category

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "training_metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "processed_data.pt")
CSV_PATH = os.path.join(BASE_DIR, "SAML-D.csv")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def sanitize_nan(obj):
    """Recursively replace NaN / Inf with None so JSON serialisation works."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    return obj


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AML Detection API",
    description="Money Laundering Detection using Graph Attention Networks",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class TransactionInput(BaseModel):
    Time: str = Field(..., example="10:35:19")
    Date: str = Field(..., example="2022-10-07")
    Sender_account: str = Field(..., example="8724731955")
    Receiver_account: str = Field(..., example="2769355426")
    Amount: float = Field(..., example=1459.15)
    Payment_currency: str = Field(..., example="UK pounds")
    Received_currency: str = Field(..., example="UK pounds")
    Sender_bank_location: str = Field(..., example="UK")
    Receiver_bank_location: str = Field(..., example="UK")
    Payment_type: str = Field(..., example="Cash Deposit")


class PredictRequest(BaseModel):
    transactions: List[TransactionInput]


class TransactionResult(BaseModel):
    sender: str
    receiver: str
    amount: float
    probability: float
    risk_category: str
    risk_score: float
    details: Dict


class PredictResponse(BaseModel):
    predictions: List[TransactionResult]
    summary: Dict


# ─── Global State (all in-memory) ─────────────────────────────────────────────
class AppState:
    model: Optional[EdgeGATModel] = None
    artifacts: Optional[Dict] = None
    metrics: Optional[Dict] = None
    graph_data = None
    device: torch.device = torch.device("cpu")

    # In-memory data (replaces MongoDB)
    accounts_df: Optional[pd.DataFrame] = None
    transactions_df: Optional[pd.DataFrame] = None
    accounts_dict: Dict = {}  # account_id → account profile


state = AppState()


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Load model + build in-memory data store from local files."""
    print("\n🔄 Starting AML Detection API (in-memory mode) ...")

    # Device
    state.device = torch.device("cpu")

    # Encoders
    if os.path.exists(ENCODERS_PATH):
        state.artifacts = joblib.load(ENCODERS_PATH)
        print(f"  ✓ Encoders loaded")

    # Model
    if os.path.exists(MODEL_PATH):
        state.model = EdgeGATModel(
            node_feat_dim=7, edge_feat_dim=8,
            hidden_dim=64, num_heads=4, dropout=0.0,
        )
        state.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=state.device, weights_only=True)
        )
        state.model.to(state.device)
        state.model.eval()
        print(f"  ✓ Model loaded")

    # Graph data
    if os.path.exists(DATA_PATH):
        state.graph_data = torch.load(DATA_PATH, map_location=state.device, weights_only=False)
        print(f"  ✓ Graph: {state.graph_data.num_nodes:,} nodes, {state.graph_data.edge_index.shape[1]:,} edges")

    # Metrics
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            state.metrics = sanitize_nan(json.load(f))
        print(f"  ✓ Metrics loaded")

    # ── Build in-memory store ──
    if os.path.exists(CSV_PATH) and state.model is not None and state.graph_data is not None:
        print(f"  🔄 Building in-memory data store ...")
        _build_in_memory_store()
        print(f"  ✓ Accounts: {len(state.accounts_dict):,}")
        print(f"  ✓ Transactions: {len(state.transactions_df):,}")

    print("✅ Server ready!\n")


def _build_in_memory_store():
    """Load CSV, run model inference, build account profiles — all in RAM."""
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S", errors="coerce",
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    # Run model inference
    data = state.graph_data
    with torch.no_grad():
        preds = state.model(data.x, data.edge_index, data.edge_attr)
    preds_np = preds.cpu().numpy()

    # Align predictions with rows
    num_preds = len(preds_np)
    df = df.iloc[:num_preds].copy()
    df["prediction_probability"] = preds_np
    df["prediction_risk_category"] = df["prediction_probability"].apply(
        lambda p: get_risk_category(float(p))
    )

    # Normalize column names for API consistency
    df["sender_account"] = df["Sender_account"].astype(str)
    df["receiver_account"] = df["Receiver_account"].astype(str)
    df["amount"] = df["Amount"]
    df["payment_currency"] = df["Payment_currency"]
    df["received_currency"] = df["Received_currency"]
    df["sender_bank_location"] = df["Sender_bank_location"]
    df["receiver_bank_location"] = df["Receiver_bank_location"]
    df["payment_type"] = df["Payment_type"]
    df["is_laundering"] = df["Is_laundering"].astype(int)

    state.transactions_df = df

    # Build account profiles
    accounts = {}

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

    # Finalize
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
        # Remove temp fields
        del a["risk_probs"]
        del a["foreign_tx"]
        del a["cross_border_tx"]
        del a["total_tx"]

    state.accounts_dict = accounts
    state.accounts_df = pd.DataFrame(list(accounts.values()))


def _new_account(acc_id: str) -> Dict:
    return {
        "account_id": acc_id,
        "total_sent": 0, "total_received": 0,
        "tx_count_sent": 0, "tx_count_received": 0,
        "unique_partners": set(),
        "foreign_tx": 0, "cross_border_tx": 0, "total_tx": 0,
        "risk_probs": [],
    }


# ─── Inference Helpers ─────────────────────────────────────────────────────────
def safe_label_transform(encoder, value: str) -> int:
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        return 0


def build_inference_features(transactions: List[TransactionInput]) -> Dict:
    artifacts = state.artifacts
    encoders = artifacts["encoders"]
    edge_scaler = artifacts["edge_scaler"]
    node_map = artifacts["node_map"]

    edge_features, edge_src, edge_dst = [], [], []
    unknown = set()
    ref_time = datetime(2022, 9, 1, 0, 0, 0)

    for tx in transactions:
        try:
            tx_dt = datetime.strptime(f"{tx.Date} {tx.Time}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            tx_dt = ref_time
        delta_t = (tx_dt - ref_time).total_seconds() / 60.0

        edge_features.append([
            tx.Amount,
            tx.Amount / (delta_t + 1.0),
            safe_label_transform(encoders["Payment_currency"], tx.Payment_currency),
            safe_label_transform(encoders["Received_currency"], tx.Received_currency),
            safe_label_transform(encoders["Sender_bank_location"], tx.Sender_bank_location),
            safe_label_transform(encoders["Receiver_bank_location"], tx.Receiver_bank_location),
            safe_label_transform(encoders["Payment_type"], tx.Payment_type),
            delta_t,
        ])

        s = node_map.get(tx.Sender_account, 0)
        r = node_map.get(tx.Receiver_account, 0)
        if tx.Sender_account not in node_map:
            unknown.add(tx.Sender_account)
        if tx.Receiver_account not in node_map:
            unknown.add(tx.Receiver_account)
        edge_src.append(s)
        edge_dst.append(r)

    edge_attr = torch.tensor(
        edge_scaler.transform(np.array(edge_features, dtype=np.float32)),
        dtype=torch.float32,
    )
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    return {"edge_attr": edge_attr, "edge_index": edge_index, "unknown": unknown}


@torch.no_grad()
def run_inference(transactions: List[TransactionInput]) -> List[Dict]:
    inf = build_inference_features(transactions)

    if state.graph_data is None:
        raise HTTPException(500, "Graph data not loaded")

    g = state.graph_data
    combined_ei = torch.cat([g.edge_index, inf["edge_index"]], dim=1)
    combined_ea = torch.cat([g.edge_attr, inf["edge_attr"]], dim=0)

    preds = state.model(g.x, combined_ei, combined_ea)
    new_preds = preds[g.edge_index.shape[1]:].cpu().numpy()

    results = []
    for i, tx in enumerate(transactions):
        prob = float(new_preds[i])
        results.append({
            "sender": tx.Sender_account,
            "receiver": tx.Receiver_account,
            "amount": tx.Amount,
            "probability": round(prob, 6),
            "risk_category": get_risk_category(prob),
            "risk_score": round(prob * 100, 2),
            "details": {
                "payment_currency": tx.Payment_currency,
                "received_currency": tx.Received_currency,
                "sender_location": tx.Sender_bank_location,
                "receiver_location": tx.Receiver_bank_location,
                "payment_type": tx.Payment_type,
                "is_cross_border": tx.Sender_bank_location != tx.Receiver_bank_location,
                "is_foreign_currency": tx.Payment_currency != tx.Received_currency,
            },
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "graph_loaded": state.graph_data is not None,
        "data_loaded": state.transactions_df is not None,
        "accounts_count": len(state.accounts_dict),
    }


# ─── Predict ──────────────────────────────────────────────────────────────────
@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict money laundering risk for transactions."""
    if state.model is None:
        raise HTTPException(503, "Model not loaded")
    if not request.transactions:
        raise HTTPException(400, "No transactions provided")
    if len(request.transactions) > 100:
        raise HTTPException(400, "Max 100 transactions per request")

    try:
        results = run_inference(request.transactions)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

    risk_counts = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
    for r in results:
        risk_counts[r["risk_category"]] += 1

    return PredictResponse(
        predictions=[TransactionResult(**r) for r in results],
        summary={
            "total_transactions": len(results),
            "risk_distribution": risk_counts,
            "flagged_count": risk_counts["High"] + risk_counts["Critical"],
            "avg_risk_score": round(
                sum(r["risk_score"] for r in results) / len(results), 2
            ),
        },
    )


# ─── Metrics ──────────────────────────────────────────────────────────────────
@app.get("/api/metrics")
async def get_metrics():
    """Return training metrics and model performance."""
    if state.metrics is None:
        raise HTTPException(404, "Metrics not available")

    return {
        "model_performance": {
            "best_epoch": state.metrics["best_epoch"],
            "best_val_f1": state.metrics["best_val_f1"],
            "optimal_threshold": state.metrics["optimal_threshold"],
            "test_metrics": state.metrics["test_metrics"],
            "confusion_matrix": state.metrics["confusion_matrix"],
            "risk_distribution": state.metrics["risk_distribution"],
        },
        "hyperparameters": state.metrics["hyperparameters"],
        "training_history": [
            {
                "epoch": h["epoch"],
                "train_loss": h["train_loss"],
                "val_loss": h["val_loss"],
                "val_f1": h["val_f1"],
            }
            for h in state.metrics.get("training_history", [])
        ],
    }


# ─── Graph Stats ──────────────────────────────────────────────────────────────
@app.get("/api/graph-stats")
async def get_graph_stats():
    if state.graph_data is None:
        raise HTTPException(404, "Graph data not loaded")

    d = state.graph_data
    num_edges = d.edge_index.shape[1]
    num_pos = int(d.y.sum().item())

    return {
        "num_nodes": d.num_nodes,
        "num_edges": num_edges,
        "node_feature_dim": d.x.shape[1],
        "edge_feature_dim": d.edge_attr.shape[1],
        "class_distribution": {
            "normal": num_edges - num_pos,
            "laundering": num_pos,
            "ratio": round((num_edges - num_pos) / max(num_pos, 1), 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ACCOUNT & SUMMARY ENDPOINTS (in-memory)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/accounts/{account_id}")
async def get_account(account_id: str):
    """
    Account profile + transaction history + graph neighbor data.
    """
    if state.accounts_dict is None:
        raise HTTPException(503, "Data not loaded")

    account = state.accounts_dict.get(account_id)
    if not account:
        raise HTTPException(404, f"Account {account_id} not found")

    df = state.transactions_df

    # Get transactions (sent + received)
    tx_sent = df[df["sender_account"] == account_id].sort_values("datetime", ascending=False)
    tx_recv = df[df["receiver_account"] == account_id].sort_values("datetime", ascending=False)

    sent_list = []
    for _, row in tx_sent.head(250).iterrows():
        sent_list.append(_tx_to_dict(row, "sent"))

    recv_list = []
    for _, row in tx_recv.head(250).iterrows():
        recv_list.append(_tx_to_dict(row, "received"))

    all_tx = sorted(sent_list + recv_list, key=lambda t: t.get("datetime", ""), reverse=True)

    # Build graph data for frontend
    neighbor_ids = set()
    graph_edges = []
    for tx in all_tx[:200]:
        neighbor_ids.add(tx["sender_account"])
        neighbor_ids.add(tx["receiver_account"])
        graph_edges.append({
            "source": tx["sender_account"],
            "target": tx["receiver_account"],
            "amount": tx["amount"],
            "risk_category": tx["prediction_risk_category"],
            "probability": tx["prediction_probability"],
            "payment_type": tx.get("payment_type", ""),
            "datetime": tx.get("datetime", ""),
        })

    # Neighbor nodes
    graph_nodes = []
    for nid in neighbor_ids:
        na = state.accounts_dict.get(nid, {})
        graph_nodes.append({
            "id": nid,
            "risk_category": na.get("risk_category", "Unknown"),
            "risk_score": na.get("risk_score", 0),
            "is_center": nid == account_id,
            "total_sent": na.get("total_sent", 0),
            "total_received": na.get("total_received", 0),
        })

    return {
        "account": account,
        "transaction_summary": {
            "total_transactions": len(all_tx),
            "total_sent_count": len(sent_list),
            "total_received_count": len(recv_list),
            "flagged_transactions": sum(
                1 for t in all_tx
                if t.get("prediction_risk_category") in ("High", "Critical")
            ),
        },
        "transactions": all_tx[:100],
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges[:200],
        },
    }


@app.get("/api/accounts")
async def list_accounts(
    category: Optional[str] = Query(None, description="Filter: Low, Moderate, High, Critical"),
    sort_by: str = Query("risk_score", description="Sort field"),
    order: str = Query("desc", description="asc or desc"),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search account ID"),
):
    """List accounts with optional risk category filter and pagination."""
    if state.accounts_df is None:
        raise HTTPException(503, "Data not loaded")

    df = state.accounts_df.copy()

    # Filter
    if category:
        if category not in ("Low", "Moderate", "High", "Critical"):
            raise HTTPException(400, "Invalid category")
        df = df[df["risk_category"] == category]
    if search:
        df = df[df["account_id"].str.contains(search, case=False, na=False)]

    # Category counts (before pagination)
    full_df = state.accounts_df
    cat_counts = {
        "Low": int((full_df["risk_category"] == "Low").sum()),
        "Moderate": int((full_df["risk_category"] == "Moderate").sum()),
        "High": int((full_df["risk_category"] == "High").sum()),
        "Critical": int((full_df["risk_category"] == "Critical").sum()),
    }

    # Sort
    ascending = order != "desc"
    allowed_sorts = ["risk_score", "total_sent", "total_received",
                     "tx_count_sent", "avg_risk_score", "unique_partners"]
    if sort_by not in allowed_sorts:
        sort_by = "risk_score"
    df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    # Paginate
    total = len(df)
    total_pages = math.ceil(total / limit) if total > 0 else 1
    start = (page - 1) * limit
    end = start + limit
    page_df = df.iloc[start:end]

    accounts = sanitize_nan(page_df.to_dict(orient="records"))

    return {
        "accounts": accounts,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
        },
        "category_counts": cat_counts,
    }


@app.get("/api/summary")
async def get_summary():
    """Comprehensive summary of training data and model results."""
    if state.transactions_df is None:
        raise HTTPException(503, "Data not loaded")

    df = state.transactions_df
    acc_df = state.accounts_df

    # Class distribution
    normal_count = int((df["is_laundering"] == 0).sum())
    launder_count = int((df["is_laundering"] == 1).sum())

    # Account risk distribution
    acc_risk = {
        "Low": int((acc_df["risk_category"] == "Low").sum()),
        "Moderate": int((acc_df["risk_category"] == "Moderate").sum()),
        "High": int((acc_df["risk_category"] == "High").sum()),
        "Critical": int((acc_df["risk_category"] == "Critical").sum()),
    }

    # Transaction risk distribution
    tx_risk = {
        "Low": int((df["prediction_risk_category"] == "Low").sum()),
        "Moderate": int((df["prediction_risk_category"] == "Moderate").sum()),
        "High": int((df["prediction_risk_category"] == "High").sum()),
        "Critical": int((df["prediction_risk_category"] == "Critical").sum()),
    }

    # Top 20 flagged accounts
    flagged = acc_df[acc_df["risk_category"].isin(["High", "Critical"])]
    top_flagged = sanitize_nan(flagged.nlargest(20, "risk_score").to_dict(orient="records"))

    # Currency stats
    curr_stats = (
        df.groupby("payment_currency")
        .agg(count=("amount", "size"), total_amount=("amount", "sum"))
        .sort_values("count", ascending=False)
        .head(10)
        .reset_index()
    )
    currency_stats = [
        {"currency": r["payment_currency"], "count": int(r["count"]),
         "total_amount": round(float(r["total_amount"]), 2)}
        for _, r in curr_stats.iterrows()
    ]

    # Location stats
    loc_stats = (
        df["sender_bank_location"].value_counts().head(15).reset_index()
    )
    loc_stats.columns = ["location", "count"]
    location_stats = [
        {"location": r["location"], "count": int(r["count"])}
        for _, r in loc_stats.iterrows()
    ]

    # Payment type stats
    ptype_stats = (
        df.groupby("payment_type")
        .agg(count=("amount", "size"), avg_amount=("amount", "mean"))
        .sort_values("count", ascending=False)
        .reset_index()
    )
    payment_types = [
        {"type": r["payment_type"], "count": int(r["count"]),
         "avg_amount": round(float(r["avg_amount"]), 2)}
        for _, r in ptype_stats.iterrows()
    ]

    # Model metrics
    model_metrics = None
    if state.metrics:
        model_metrics = {
            "best_epoch": state.metrics["best_epoch"],
            "test_metrics": state.metrics["test_metrics"],
            "confusion_matrix": state.metrics["confusion_matrix"],
            "hyperparameters": state.metrics["hyperparameters"],
        }

    return {
        "overview": {
            "total_accounts": len(state.accounts_dict),
            "total_transactions": len(df),
            "class_distribution": {
                "normal": normal_count,
                "laundering": launder_count,
            },
        },
        "risk_distribution": {
            "accounts": acc_risk,
            "transactions": tx_risk,
        },
        "top_flagged_accounts": top_flagged,
        "currency_stats": currency_stats,
        "location_stats": location_stats,
        "payment_type_stats": payment_types,
        "model_metrics": model_metrics,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _tx_to_dict(row, direction: str) -> Dict:
    return {
        "sender_account": str(row["sender_account"]),
        "receiver_account": str(row["receiver_account"]),
        "amount": float(row["amount"]),
        "payment_currency": str(row["payment_currency"]),
        "received_currency": str(row["received_currency"]),
        "sender_bank_location": str(row["sender_bank_location"]),
        "receiver_bank_location": str(row["receiver_bank_location"]),
        "payment_type": str(row["payment_type"]),
        "datetime": str(row["datetime"]) if pd.notna(row["datetime"]) else None,
        "is_laundering": int(row["is_laundering"]),
        "prediction_probability": round(float(row["prediction_probability"]), 6),
        "prediction_risk_category": str(row["prediction_risk_category"]),
        "direction": direction,
    }


# ─── Run Server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AML Detection — FastAPI Server v2.0 (In-Memory)         ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
