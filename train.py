"""
train.py — Step 2: Training & Evaluation Pipeline (GPU Optimized)
==================================================================
Optimized for RTX 3060 (6GB VRAM) + 16GB RAM:
  - Mixed Precision (AMP) → ~40% less VRAM
  - VRAM capped at 80% → leaves room for OS/display
  - Edge mini-batching → controls memory per step
  - gc + cache clearing between epochs
  - Lower CPU thread count → system stays responsive

Usage:
    python augment_data.py      # inject laundering patterns first
    python data_pipeline.py     # rebuild graph
    python train.py             # train model
"""

import gc
import json
import os
import time
from typing import Dict

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

from model import EdgeGATModel, FocalLoss, count_parameters, get_risk_category

# ─── Resource Limits (keep PC responsive) ────────────────────────────────────
# Limit PyTorch CPU threads so OS stays snappy
torch.set_num_threads(4)               # Ryzen 5 has 6C/12T → use 4 threads
torch.set_num_interop_threads(2)

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed_data.pt")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
METRICS_PATH = os.path.join(BASE_DIR, "training_metrics.json")

# Hyperparameters
EPOCHS = 150
LR = 0.0005
WEIGHT_DECAY = 1e-5
PATIENCE = 25
FOCAL_ALPHA = 0.90
FOCAL_GAMMA = 2.0


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
        preds = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        )

    preds_masked = preds[mask].cpu().numpy()
    labels_masked = data.y[mask].cpu().numpy()
    loss = criterion(preds[mask], data.y[mask].to(device)).item()

    binary_preds = (preds_masked >= threshold).astype(int)
    precision = precision_score(labels_masked, binary_preds, zero_division=0)
    recall = recall_score(labels_masked, binary_preds, zero_division=0)
    f1 = f1_score(labels_masked, binary_preds, zero_division=0)

    return {
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "preds": preds_masked,
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
    oversample_ratio = 3.0
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

    print("\n✅ Training complete! Ready for Step 3 (FastAPI Backend).\n")


if __name__ == "__main__":
    main()
