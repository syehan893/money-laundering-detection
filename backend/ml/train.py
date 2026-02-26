"""
train.py — Training & Evaluation Pipeline (GPU Optimized)
==================================================================
Optimized for RTX 3060 (6GB VRAM) + 16GB RAM:
  - Mixed Precision (AMP) → ~40% less VRAM
  - VRAM capped at 80% → leaves room for OS/display
  - Edge mini-batching → controls memory per step
  - gc + cache clearing between epochs
  - Lower CPU thread count → system stays responsive
  - After training: populate MongoDB with predictions & accounts

Usage:
    python -m backend.ml.train
"""

import gc
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict

import joblib
import pandas as pd

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.ml.model import EdgeGATModel, FocalLoss, count_parameters, get_risk_category
from backend.config import DATA_PATH, MODEL_PATH, METRICS_PATH, CSV_PATH, ENCODERS_PATH
from backend.config import EPOCHS, LR, WEIGHT_DECAY, PATIENCE, FOCAL_ALPHA, FOCAL_GAMMA

# ─── Resource Limits (keep PC responsive) ────────────────────────────────────
# Limit PyTorch CPU threads so OS stays snappy
torch.set_num_threads(4)               # Ryzen 5 has 6C/12T → use 4 threads
torch.set_num_interop_threads(2)


# ─── Device Setup ─────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name} ({total_mem:.1f} GB VRAM)")

        # Cap VRAM at 80% → leaves ~1.2GB for Windows + display
        torch.cuda.set_per_process_memory_fraction(0.80, 0)
        print(f"  VRAM limit: {total_mem * 0.80:.1f} GB (80%)")
        print(f"  Mixed precision (AMP): ENABLED")
    else:
        device = torch.device("cpu")
        print("  Using CPU (no CUDA available)")
    return device


# ─── Balanced Sampling ────────────────────────────────────────────────────────
def get_balanced_edge_indices(
    y: torch.Tensor, mask: torch.Tensor, oversample_ratio: float = 3.0
) -> torch.Tensor:
    """Balanced edge sampling with minority oversampling."""
    mask_indices = mask.nonzero(as_tuple=True)[0]
    labels = y[mask_indices]

    pos_indices = mask_indices[labels == 1]
    neg_indices = mask_indices[labels == 0]

    num_pos = len(pos_indices)
    if num_pos == 0:
        return neg_indices[torch.randperm(len(neg_indices))[:10000]]

    num_neg_sample = min(int(num_pos * oversample_ratio), len(neg_indices))

    # Oversample positives
    repeat_count = max(num_neg_sample // num_pos, 1)
    pos_oversampled = pos_indices.repeat(repeat_count)

    # Sample negatives
    neg_sampled = neg_indices[torch.randperm(len(neg_indices))[:num_neg_sample]]

    balanced = torch.cat([pos_oversampled, neg_sampled])
    return balanced[torch.randperm(len(balanced))]


# ─── Evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: EdgeGATModel,
    data,
    mask: torch.Tensor,
    criterion: FocalLoss,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = True,
) -> Dict:
    """Memory-efficient evaluation."""
    model.eval()

    with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
        logits = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        )

    # Apply sigmoid to get probabilities (model returns raw logits)
    probs_masked = torch.sigmoid(logits[mask]).cpu().numpy()
    labels_masked = data.y[mask].cpu().numpy()
    loss = criterion(logits[mask], data.y[mask].to(device)).item()

    binary_preds = (probs_masked >= threshold).astype(int)
    precision = precision_score(labels_masked, binary_preds, zero_division=0)
    recall = recall_score(labels_masked, binary_preds, zero_division=0)
    f1 = f1_score(labels_masked, binary_preds, zero_division=0)

    return {
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "preds": probs_masked,
        "labels": labels_masked,
        "binary_preds": binary_preds,
    }


def find_best_threshold(preds: np.ndarray, labels: np.ndarray) -> tuple:
    """Find threshold maximizing F1."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(labels, (preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


# ─── Training (AMP + balanced sampling) ──────────────────────────────────────
def train_one_epoch(
    model: EdgeGATModel,
    data,
    optimizer: optim.Optimizer,
    criterion: FocalLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
    oversample_ratio: float = 3.0,
) -> float:
    """Train one epoch with AMP mixed precision + balanced sampling."""
    model.train()
    optimizer.zero_grad(set_to_none=True)  # More memory efficient

    # Forward with mixed precision
    with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
        preds = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        )

        balanced_idx = get_balanced_edge_indices(
            data.y, data.train_mask, oversample_ratio=oversample_ratio
        )
        loss = criterion(preds[balanced_idx], data.y[balanced_idx].to(device))

    # Backward with gradient scaling (AMP)
    if use_amp and device.type == "cuda":
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return loss.item()


# ─── Risk Distribution ────────────────────────────────────────────────────────
def compute_risk_distribution(preds: np.ndarray) -> Dict[str, int]:
    risk_counts = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
    for p in preds:
        risk_counts[get_risk_category(float(p))] += 1
    return risk_counts


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AML Detection — GAT Training (GPU Optimized)            ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Load Data ──
    print("[1/5] Loading processed data ...")
    if not os.path.exists(DATA_PATH):
        print(f"  ❌ {DATA_PATH} not found! Run data_pipeline.py first.")
        return
    data = torch.load(DATA_PATH, weights_only=False)
    print(f"  Nodes: {data.num_nodes:,} | Edges: {data.edge_index.shape[1]:,}")

    num_pos = int(data.y.sum().item())
    num_neg = int(data.y.shape[0]) - num_pos
    print(f"  Class 0: {num_neg:,} | Class 1: {num_pos:,} | Ratio: {num_neg/max(num_pos,1):.1f}:1")

    # ── Device ──
    device = get_device()
    use_amp = device.type == "cuda"

    # ── Model ──
    print(f"\n[2/5] Initializing EdgeGATModel ...")
    model = EdgeGATModel(
        node_feat_dim=data.x.shape[1],
        edge_feat_dim=data.edge_attr.shape[1],
        hidden_dim=64,
        num_heads=4,
        dropout=0.3,
    ).to(device)
    print(f"  Parameters: {count_parameters(model):,}")

    if device.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated() / (1024**2)
        print(f"  VRAM after model load: {mem_alloc:.0f} MB")

    # ── Loss, Optimizer, AMP Scaler ──
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training Loop ──
    oversample_ratio = 5.0
    print(f"\n[3/5] Training (epochs={EPOCHS}, patience={PATIENCE}, AMP={'ON' if use_amp else 'OFF'}) ...\n")
    print(f"  {'Ep':>4} | {'T.Loss':>8} | {'V.Loss':>8} | {'V.P':>6} | {'V.R':>6} | {'V.F1':>6} | {'VRAM':>6} | {'Time':>5}")
    print("  " + "-" * 64)

    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, data, optimizer, criterion, grad_scaler,
            device, use_amp=use_amp, oversample_ratio=oversample_ratio,
        )

        val_metrics = evaluate(model, data, data.val_mask, criterion, device, use_amp=use_amp)
        scheduler.step()
        elapsed = time.time() - t0

        # VRAM usage
        vram_mb = torch.cuda.memory_allocated() / (1024**2) if device.type == "cuda" else 0

        if epoch <= 3 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f"  {epoch:4d} | {train_loss:8.5f} | {val_metrics['loss']:8.5f} | "
                f"{val_metrics['precision']:6.4f} | {val_metrics['recall']:6.4f} | "
                f"{val_metrics['f1']:6.4f} | {vram_mb:5.0f}M | {elapsed:5.1f}s"
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
            "vram_mb": vram_mb,
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  ⏹ Early stopping at epoch {epoch} (best: {best_epoch})")
                break

        # Memory cleanup every 10 epochs
        if epoch % 10 == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"\n  ✓ Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ── Test Evaluation ──
    print(f"\n[4/5] Test evaluation ...")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    val_eval = evaluate(model, data, data.val_mask, criterion, device, use_amp=use_amp)
    optimal_thresh, optimal_f1 = find_best_threshold(val_eval["preds"], val_eval["labels"])
    print(f"  Optimal threshold: {optimal_thresh:.2f} (val F1={optimal_f1:.4f})")

    test_metrics = evaluate(
        model, data, data.test_mask, criterion, device,
        threshold=optimal_thresh, use_amp=use_amp,
    )

    print("\n  ── Classification Report ──")
    report = classification_report(
        test_metrics["labels"],
        test_metrics["binary_preds"],
        target_names=["Normal (0)", "Laundering (1)"],
        digits=4,
    )
    print(report)

    cm = confusion_matrix(test_metrics["labels"], test_metrics["binary_preds"])
    print("  ── Confusion Matrix ──")
    print(f"                    Predicted 0    Predicted 1")
    print(f"  Actual 0 (Normal)     {cm[0][0]:>8,}       {cm[0][1]:>8,}")
    print(f"  Actual 1 (Launder)    {cm[1][0]:>8,}       {cm[1][1]:>8,}")

    risk_dist = compute_risk_distribution(test_metrics["preds"])
    print("\n  ── Risk Distribution (Test Set) ──")
    total_test = len(test_metrics["preds"])
    for cat in ["Low", "Moderate", "High", "Critical"]:
        count = risk_dist[cat]
        pct = count / total_test * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:>10}: {count:>8,}  ({pct:5.2f}%) {bar}")

    # ── Save ──
    print(f"\n[5/5] Saving outputs ...")
    metrics_output = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "optimal_threshold": optimal_thresh,
        "test_metrics": {
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "loss": test_metrics["loss"],
        },
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        },
        "risk_distribution": risk_dist,
        "hyperparameters": {
            "epochs_trained": len(history),
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "focal_alpha": FOCAL_ALPHA,
            "focal_gamma": FOCAL_GAMMA,
            "hidden_dim": 64,
            "num_heads": 4,
            "dropout": 0.3,
            "threshold": optimal_thresh,
            "oversample_ratio": oversample_ratio,
        },
        "training_history": history,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"  ✓ Model: {MODEL_PATH}")
    print(f"  ✓ Metrics: {METRICS_PATH}")

    # Final cleanup
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Populate MongoDB ──
    print(f"\n[6/6] Populating MongoDB ...")
    populate_mongodb(model, data, metrics_output, device)

    print("\n✅ Training complete! MongoDB populated. Ready for FastAPI Backend.\n")


# ─── MongoDB Population ──────────────────────────────────────────────────────
def populate_mongodb(model, data, metrics_output, device):
    """Run inference on all transactions and populate MongoDB collections."""
    from backend.database import get_sync_db, setup_sync_indexes, close_sync_db

    db = get_sync_db()
    setup_sync_indexes(db)

    # ── 1) Load CSV + run inference ──
    print("  🔄 Loading CSV and running inference ...")
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%Y-%m-%d %H:%M:%S", errors="coerce",
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    model.eval()
    with torch.no_grad():
        preds = model.predict(
            data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        )
    preds_np = preds.cpu().numpy()

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

    # ── 2) Build account profiles ──
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

    # ── 3) Drop old data and insert new ──
    print("  🔄 Inserting transactions into MongoDB ...")
    db.transactions.drop()

    # Prepare transaction documents
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
        # Convert datetime to string for MongoDB
        if pd.notna(doc.get("datetime")):
            doc["datetime"] = str(doc["datetime"])
        else:
            doc["datetime"] = None
        # Ensure float for probability
        doc["prediction_probability"] = round(float(doc["prediction_probability"]), 6)
        tx_docs.append(doc)

    # Batch insert (10K per batch)
    batch_size = 10000
    for i in range(0, len(tx_docs), batch_size):
        batch = tx_docs[i:i + batch_size]
        db.transactions.insert_many(batch, ordered=False)
        print(f"    Inserted {min(i + batch_size, len(tx_docs)):,} / {len(tx_docs):,} transactions")

    print(f"  ✓ Transactions: {len(tx_docs):,} documents")

    # ── 4) Insert accounts ──
    print("  🔄 Inserting accounts into MongoDB ...")
    db.accounts.drop()
    acc_docs = list(accounts.values())

    for i in range(0, len(acc_docs), batch_size):
        batch = acc_docs[i:i + batch_size]
        db.accounts.insert_many(batch, ordered=False)
        print(f"    Inserted {min(i + batch_size, len(acc_docs)):,} / {len(acc_docs):,} accounts")

    print(f"  ✓ Accounts: {len(acc_docs):,} documents")

    # ── 5) Insert training metrics ──
    print("  🔄 Inserting training metrics ...")
    metrics_doc = metrics_output.copy()
    metrics_doc["run_id"] = datetime.now().isoformat()
    metrics_doc["created_at"] = datetime.now()

    # Sanitize NaN values in training_history
    for h in metrics_doc.get("training_history", []):
        for k, v in h.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                h[k] = None

    db.training_metrics.insert_one(metrics_doc)
    print(f"  ✓ Training metrics: 1 document")

    # Re-create indexes after drop
    setup_sync_indexes(db)

    close_sync_db()
    print("  ✓ MongoDB population complete!")


if __name__ == "__main__":
    main()
