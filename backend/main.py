"""
main.py — FastAPI Backend for AML Detection (MongoDB)
======================================================
REST API that reads data from MongoDB and serves predictions,
account history, and summary endpoints.

MongoDB collections (populated by train.py):
    - transactions:      152K+ docs with predictions
    - accounts:          52K+ account profiles with risk scores
    - training_metrics:  Training results snapshot

Endpoints:
    POST /api/predict                 - Predict laundering risk (+ save to MongoDB)
    GET  /api/metrics                 - Model performance metrics
    GET  /api/graph-stats             - Graph statistics
    GET  /api/health                  - Health check
    GET  /api/accounts/{account_id}   - Account history + graph data
    GET  /api/accounts                - List accounts by risk category
    GET  /api/summary                 - Training data summary

Usage:
    python -m backend.main
"""

import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import connect_async_db, close_async_db, get_async_db
from backend.ml.model import EdgeGATModel, get_risk_category
from backend.config import MODEL_PATH, ENCODERS_PATH, DATA_PATH


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AML Detection API",
    description="Anti-Money Laundering Detection using Graph Attention Networks",
    version="3.0",
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
    Amount: float = Field(..., example=5000.00)
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


# ─── Global State (model only — data is in MongoDB) ──────────────────────────
class AppState:
    model: Optional[EdgeGATModel] = None
    artifacts: Optional[Dict] = None
    graph_data = None
    device: torch.device = torch.device("cpu")


state = AppState()


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Load model + connect to MongoDB."""
    print("\n🔄 Starting AML Detection API (MongoDB) ...")

    # Device
    state.device = torch.device("cpu")

    # Encoders (for /predict)
    if os.path.exists(ENCODERS_PATH):
        state.artifacts = joblib.load(ENCODERS_PATH)
        print(f"  ✓ Encoders loaded")

    # Model (for /predict)
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

    # Graph data (for /graph-stats only)
    if os.path.exists(DATA_PATH):
        state.graph_data = torch.load(DATA_PATH, map_location=state.device, weights_only=False)
        print(f"  ✓ Graph: {state.graph_data.num_nodes:,} nodes, {state.graph_data.edge_index.shape[1]:,} edges")

    # MongoDB
    await connect_async_db()

    print("✅ Server ready!\n")


@app.on_event("shutdown")
async def shutdown():
    await close_async_db()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def sanitize_nan(obj):
    """Recursively replace NaN / Inf with None so JSON serialisation works."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(i) for i in obj]
    return obj


def _clean_doc(doc):
    """Remove MongoDB _id field from a document for JSON response."""
    if doc and "_id" in doc:
        del doc["_id"]
    return doc


def _clean_docs(docs):
    """Remove _id from list of documents."""
    return [_clean_doc(d) for d in docs]


# ─── Inference Helpers ─────────────────────────────────────────────────────────
def safe_label_transform(encoder, value: str):
    try:
        return int(encoder.transform([value])[0])
    except (ValueError, KeyError):
        return 0


def build_inference_features(transactions: List[TransactionInput]):
    if not state.artifacts or not state.model:
        raise HTTPException(503, "Model not loaded")

    encoders = state.artifacts
    features = []
    for tx in transactions:
        pay_curr = safe_label_transform(encoders["Payment_currency"], tx.Payment_currency)
        rec_curr = safe_label_transform(encoders["Received_currency"], tx.Received_currency)
        s_loc = safe_label_transform(encoders["Sender_bank_location"], tx.Sender_bank_location)
        r_loc = safe_label_transform(encoders["Receiver_bank_location"], tx.Receiver_bank_location)
        p_type = safe_label_transform(encoders["Payment_type"], tx.Payment_type)

        feat = [pay_curr, rec_curr, s_loc, r_loc, p_type, tx.Amount, tx.Amount, 0]
        features.append(feat)

    return torch.tensor(features, dtype=torch.float32)


def run_inference(transactions: List[TransactionInput]):
    if state.graph_data is None or state.model is None:
        raise HTTPException(503, "Model or graph data not loaded")

    edge_features = build_inference_features(transactions)
    data = state.graph_data

    with torch.no_grad():
        all_preds = state.model(
            data.x.to(state.device),
            data.edge_index.to(state.device),
            data.edge_attr.to(state.device),
        )

    n_existing = data.edge_attr.shape[0]
    combined_edge_attr = torch.cat([data.edge_attr, edge_features], dim=0)
    combined_edge_index = torch.cat([
        data.edge_index,
        torch.zeros(2, len(transactions), dtype=torch.long),
    ], dim=1)

    with torch.no_grad():
        all_preds = state.model(
            data.x.to(state.device),
            combined_edge_index.to(state.device),
            combined_edge_attr.to(state.device),
        )

    new_preds = all_preds[n_existing:].cpu().numpy()
    return new_preds


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    db = get_async_db()
    db_connected = db is not None
    acc_count = 0
    tx_count = 0
    if db_connected:
        try:
            acc_count = await db.accounts.estimated_document_count()
            tx_count = await db.transactions.estimated_document_count()
        except Exception:
            db_connected = False

    return {
        "status": "healthy" if db_connected and state.model else "degraded",
        "model_loaded": state.model is not None,
        "database_connected": db_connected,
        "accounts_count": acc_count,
        "transactions_count": tx_count,
    }


# ─── Predict ──────────────────────────────────────────────────────────────────
@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict money laundering risk for transactions + save to MongoDB."""
    if not request.transactions:
        raise HTTPException(400, "No transactions provided")

    preds = run_inference(request.transactions)
    results = []

    for tx, prob in zip(request.transactions, preds):
        p = float(prob)
        cat = get_risk_category(p)
        results.append(TransactionResult(
            sender=tx.Sender_account,
            receiver=tx.Receiver_account,
            amount=tx.Amount,
            probability=round(p, 6),
            risk_category=cat,
            risk_score=round(p * 100, 2),
            details={
                "payment_type": tx.Payment_type,
                "cross_border": tx.Sender_bank_location != tx.Receiver_bank_location,
                "currency_mismatch": tx.Payment_currency != tx.Received_currency,
            },
        ))

    summary = {
        "total_analyzed": len(results),
        "high_risk_count": sum(1 for r in results if r.risk_category in ("High", "Critical")),
        "average_risk": round(float(np.mean([r.probability for r in results])), 4),
    }

    # Save predictions to MongoDB
    db = get_async_db()
    if db is not None:
        pred_docs = []
        for tx, r in zip(request.transactions, results):
            pred_docs.append({
                "sender_account": r.sender,
                "receiver_account": r.receiver,
                "amount": r.amount,
                "probability": r.probability,
                "risk_category": r.risk_category,
                "risk_score": r.risk_score,
                "payment_type": tx.Payment_type,
                "payment_currency": tx.Payment_currency,
                "received_currency": tx.Received_currency,
                "sender_bank_location": tx.Sender_bank_location,
                "receiver_bank_location": tx.Receiver_bank_location,
                "cross_border": tx.Sender_bank_location != tx.Receiver_bank_location,
                "currency_mismatch": tx.Payment_currency != tx.Received_currency,
                "created_at": datetime.now(),
            })
        try:
            await db.predictions.insert_many(pred_docs)
        except Exception as e:
            print(f"  ⚠ Failed to save predictions to MongoDB: {e}")

    return PredictResponse(predictions=results, summary=summary)


# ─── Metrics ──────────────────────────────────────────────────────────────────
@app.get("/api/metrics")
async def get_metrics():
    """Return training metrics and model performance from MongoDB."""
    db = get_async_db()
    if db is None:
        raise HTTPException(503, "Database not connected")

    # Get latest training metrics
    doc = await db.training_metrics.find_one(
        {}, sort=[("created_at", -1)]
    )
    if not doc:
        raise HTTPException(404, "No training metrics found")

    doc = _clean_doc(doc)
    doc = sanitize_nan(doc)

    return {
        "model_performance": {
            "best_epoch": doc.get("best_epoch"),
            "best_val_f1": doc.get("best_val_f1"),
            "optimal_threshold": doc.get("optimal_threshold"),
            "test_metrics": doc.get("test_metrics"),
            "confusion_matrix": doc.get("confusion_matrix"),
            "risk_distribution": doc.get("risk_distribution"),
        },
        "hyperparameters": doc.get("hyperparameters"),
        "training_history": [
            {
                "epoch": h.get("epoch"),
                "train_loss": h.get("train_loss"),
                "val_loss": h.get("val_loss"),
                "val_f1": h.get("val_f1"),
            }
            for h in doc.get("training_history", [])
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
#  ACCOUNT & SUMMARY ENDPOINTS (MongoDB)
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/accounts/{account_id}")
async def get_account(account_id: str):
    """Account profile + transaction history + graph neighbor data."""
    db = get_async_db()
    if db is None:
        raise HTTPException(503, "Database not connected")

    # Get account
    account = await db.accounts.find_one({"account_id": account_id})
    if not account:
        raise HTTPException(404, f"Account {account_id} not found")
    account = _clean_doc(account)

    # Get transactions (sent + received)
    sent_cursor = db.transactions.find(
        {"sender_account": account_id}
    ).sort("datetime", -1).limit(250)
    sent_txs = await sent_cursor.to_list(250)

    recv_cursor = db.transactions.find(
        {"receiver_account": account_id}
    ).sort("datetime", -1).limit(250)
    recv_txs = await recv_cursor.to_list(250)

    sent_list = [_tx_to_dict(t, "sent") for t in sent_txs]
    recv_list = [_tx_to_dict(t, "received") for t in recv_txs]

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

    # Fetch neighbor account info from MongoDB
    graph_nodes = []
    if neighbor_ids:
        neighbor_cursor = db.accounts.find(
            {"account_id": {"$in": list(neighbor_ids)}}
        )
        neighbor_accounts = {
            na["account_id"]: na
            async for na in neighbor_cursor
        }
        for nid in neighbor_ids:
            na = neighbor_accounts.get(nid, {})
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
    db = get_async_db()
    if db is None:
        raise HTTPException(503, "Database not connected")

    # Build query filter
    query = {}
    if category:
        if category not in ("Low", "Moderate", "High", "Critical"):
            raise HTTPException(400, "Invalid category")
        query["risk_category"] = category
    if search:
        query["account_id"] = {"$regex": search, "$options": "i"}

    # Category counts (before pagination)
    cat_counts = {}
    for cat in ["Low", "Moderate", "High", "Critical"]:
        cat_counts[cat] = await db.accounts.count_documents({"risk_category": cat})

    # Sort
    allowed_sorts = ["risk_score", "total_sent", "total_received",
                     "tx_count_sent", "avg_risk_score", "unique_partners"]
    if sort_by not in allowed_sorts:
        sort_by = "risk_score"
    sort_dir = -1 if order == "desc" else 1

    # Total count for pagination
    total = await db.accounts.count_documents(query)
    total_pages = math.ceil(total / limit) if total > 0 else 1

    # Fetch page
    skip = (page - 1) * limit
    cursor = db.accounts.find(query).sort(sort_by, sort_dir).skip(skip).limit(limit)
    accounts = _clean_docs(await cursor.to_list(limit))

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
    db = get_async_db()
    if db is None:
        raise HTTPException(503, "Database not connected")

    # Total counts
    total_accounts = await db.accounts.estimated_document_count()
    total_transactions = await db.transactions.estimated_document_count()

    # Class distribution
    normal_count = await db.transactions.count_documents({"is_laundering": 0})
    launder_count = await db.transactions.count_documents({"is_laundering": 1})

    # Account risk distribution
    acc_risk = {}
    for cat in ["Low", "Moderate", "High", "Critical"]:
        acc_risk[cat] = await db.accounts.count_documents({"risk_category": cat})

    # Transaction risk distribution
    tx_risk = {}
    for cat in ["Low", "Moderate", "High", "Critical"]:
        tx_risk[cat] = await db.transactions.count_documents({"prediction_risk_category": cat})

    # Top 20 flagged accounts
    flagged_cursor = db.accounts.find(
        {"risk_category": {"$in": ["High", "Critical"]}}
    ).sort("risk_score", -1).limit(20)
    top_flagged = _clean_docs(await flagged_cursor.to_list(20))

    # Currency stats (aggregation)
    curr_pipeline = [
        {"$group": {
            "_id": "$payment_currency",
            "count": {"$sum": 1},
            "total_amount": {"$sum": "$amount"},
        }},
        {"$sort": {"count": -1}},
        {"$limit": 10},
    ]
    curr_agg = await db.transactions.aggregate(curr_pipeline).to_list(10)
    currency_stats = [
        {"currency": r["_id"], "count": r["count"],
         "total_amount": round(r["total_amount"], 2)}
        for r in curr_agg
    ]

    # Location stats
    loc_pipeline = [
        {"$group": {"_id": "$sender_bank_location", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15},
    ]
    loc_agg = await db.transactions.aggregate(loc_pipeline).to_list(15)
    location_stats = [
        {"location": r["_id"], "count": r["count"]} for r in loc_agg
    ]

    # Payment type stats
    ptype_pipeline = [
        {"$group": {
            "_id": "$payment_type",
            "count": {"$sum": 1},
            "avg_amount": {"$avg": "$amount"},
        }},
        {"$sort": {"count": -1}},
    ]
    ptype_agg = await db.transactions.aggregate(ptype_pipeline).to_list(20)
    payment_types = [
        {"type": r["_id"], "count": r["count"],
         "avg_amount": round(r["avg_amount"], 2)}
        for r in ptype_agg
    ]

    # Model metrics (latest from training_metrics collection)
    model_metrics = None
    metrics_doc = await db.training_metrics.find_one({}, sort=[("created_at", -1)])
    if metrics_doc:
        model_metrics = {
            "best_epoch": metrics_doc.get("best_epoch"),
            "test_metrics": metrics_doc.get("test_metrics"),
            "confusion_matrix": metrics_doc.get("confusion_matrix"),
            "hyperparameters": metrics_doc.get("hyperparameters"),
        }

    return {
        "overview": {
            "total_accounts": total_accounts,
            "total_transactions": total_transactions,
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
def _tx_to_dict(doc, direction: str) -> Dict:
    return {
        "sender_account": str(doc.get("sender_account", "")),
        "receiver_account": str(doc.get("receiver_account", "")),
        "amount": float(doc.get("amount", 0)),
        "payment_currency": str(doc.get("payment_currency", "")),
        "received_currency": str(doc.get("received_currency", "")),
        "sender_bank_location": str(doc.get("sender_bank_location", "")),
        "receiver_bank_location": str(doc.get("receiver_bank_location", "")),
        "payment_type": str(doc.get("payment_type", "")),
        "datetime": str(doc.get("datetime", "")) if doc.get("datetime") else None,
        "is_laundering": int(doc.get("is_laundering", 0)),
        "prediction_probability": round(float(doc.get("prediction_probability", 0)), 6),
        "prediction_risk_category": str(doc.get("prediction_risk_category", "Low")),
        "direction": direction,
    }


# ─── Run Server ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AML Detection — FastAPI Server v3.0 (MongoDB)           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
