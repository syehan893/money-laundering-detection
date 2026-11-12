"""
train.py — GAT Training & Evaluation Pipeline (GPU Optimized)
==================================================================
Optimized for RTX 3060 (6GB VRAM) + 16GB RAM:
  - Mixed Precision (AMP) -> ~40% less VRAM
  - VRAM capped at 95% -> leaves room for OS/display
  - Edge mini-batching -> controls memory per step
  - gc + cache clearing between epochs
  - Lower CPU thread count -> system stays responsive

Improvements in this version:
  - Reproducible seeding via backend.ml.utils.set_seed
  - Cost-sensitive FocalLoss with FN cost weighting
  - Hard-example mining to focus on false negatives/positives
  - Constraint-based threshold selection (FPR/FNR < 0.1%)
  - Logging to logs/training.log and console
  - Checkpointing: best, last, and periodic checkpoints

Usage:
    python -m backend.ml.legacy.train
"""

import gc
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Reduce CUDA memory fragmentation (helpful for RTX 3060 with large graphs)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

from backend.ml.legacy.model import EdgeGATModel, FocalLoss, count_parameters, get_risk_category
from backend.ml.legacy.evaluate import (
    compute_confusion_matrix,
    compute_error_rates,
    compute_risk_distribution,
    evaluate_model,
    find_best_threshold,
    find_threshold_with_constraints,
    print_classification_report,
    print_confusion_matrix,
    save_metrics,
)
from backend.ml.utils import (
    format_time,
    get_device,
    get_memory_usage_mb,
    sanitize_for_json,
    set_seed,
)
from backend.config import (
    CSV_PATH,
    DATA_PATH,
    ENCODERS_PATH,
    METRICS_PATH,
    MODEL_PATH,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
    VAL_CSV_PATH,
    DROPOUT_TRAIN,
    EPOCHS,
    FOCAL_ALPHA,
    FOCAL_GAMMA,
    FN_COST,
    FP_COST,
    HARD_MINING_ENABLED,
    HARD_MINING_RATIO,
    HARD_MINING_START_EPOCH,
    HIDDEN_DIM,
    LABEL_SMOOTHING,
    LR,
    MAX_FNR,
    MAX_FPR,
    NODE_FEAT_DIM,
    EDGE_FEAT_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    PATIENCE,
    SEED,
    THRESHOLD_GRID_STEP,
    USE_LAYER_NORM,
    USE_RESIDUAL,
    WARM_START_FROM_CHECKPOINT,
    WEIGHT_DECAY,
)

# ─── Logging Setup ────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "training.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ─── Resource Limits ──────────────────────────────────────────────────────────
torch.set_num_threads(4)
torch.set_num_interop_threads(2)


# ─── Balanced Sampling & Hard Mining ──────────────────────────────────────────
def get_balanced_edge_indices(
    y: torch.Tensor,
    mask: torch.Tensor,
    oversample_ratio: float = 3.0,
) -> torch.Tensor:
    """
    Balanced edge sampling with minority oversampling.

    Args:
        y: Edge labels.
        mask: Boolean mask (e.g., train_mask).
        oversample_ratio: Number of negatives per positive.

    Returns:
        Tensor of sampled edge indices.
    """
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


def get_hard_example_indices(
    y: torch.Tensor,
    mask: torch.Tensor,
    probs: torch.Tensor,
    ratio: float = 0.3,
    top_k_per_class: Optional[int] = None,
) -> torch.Tensor:
    """
    Identify hard examples from a previous evaluation.

    Hard examples are:
      - False negatives: laundering edges with low predicted probability.
      - False positives: normal edges with high predicted probability.

    Args:
        y: Edge labels.
        mask: Mask selecting edges to consider.
        probs: Predicted probabilities.
        ratio: Fraction of positives/negatives to return.
        top_k_per_class: If set, cap the number of hard examples per class.

    Returns:
        Tensor of hard edge indices.
    """
    mask_indices = mask.nonzero(as_tuple=True)[0]
    labels = y[mask_indices].cpu().numpy()
    p = probs[mask_indices].cpu().numpy()

    # False negatives: label=1 but predicted low
    fn_mask = (labels == 1)
    fn_indices = mask_indices[fn_mask]
    fn_scores = p[fn_mask]
    if len(fn_scores) > 0:
        k_fn = max(1, int(len(fn_scores) * ratio))
        if top_k_per_class is not None:
            k_fn = min(k_fn, top_k_per_class)
        fn_hard = fn_indices[torch.from_numpy(np.argpartition(fn_scores, k_fn - 1)[:k_fn])]
    else:
        fn_hard = torch.tensor([], dtype=torch.long)

    # False positives: label=0 but predicted high
    fp_mask = (labels == 0)
    fp_indices = mask_indices[fp_mask]
    fp_scores = p[fp_mask]
    if len(fp_scores) > 0:
        k_fp = max(1, int(len(fp_scores) * ratio))
        if top_k_per_class is not None:
            k_fp = min(k_fp, top_k_per_class)
        fp_hard = fp_indices[torch.from_numpy(np.argpartition(-fp_scores, k_fp - 1)[:k_fp])]
    else:
        fp_hard = torch.tensor([], dtype=torch.long)

    hard = torch.cat([fn_hard, fp_hard])
    return hard[torch.randperm(len(hard))]


def mix_with_hard_examples(
    base_indices: torch.Tensor,
    hard_indices: torch.Tensor,
    hard_ratio: float = 0.3,
) -> torch.Tensor:
    """Replace a fraction of base indices with hard examples."""
    if len(hard_indices) == 0:
        return base_indices

    n_hard = min(int(len(base_indices) * hard_ratio), len(hard_indices))
    n_keep = len(base_indices) - n_hard

    keep_indices = base_indices[torch.randperm(len(base_indices))[:n_keep]]
    selected_hard = hard_indices[torch.randperm(len(hard_indices))[:n_hard]]

    mixed = torch.cat([keep_indices, selected_hard])
    return mixed[torch.randperm(len(mixed))]


# ─── Training Step ────────────────────────────────────────────────────────────
def train_one_epoch(
    model: EdgeGATModel,
    data,
    optimizer: optim.Optimizer,
    criterion: FocalLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool = True,
    oversample_ratio: float = 5.0,
    hard_indices: Optional[torch.Tensor] = None,
    hard_ratio: float = 0.0,
) -> Tuple[float, torch.Tensor]:
    """
    Train one epoch with AMP + balanced sampling + optional hard mining.

    Returns:
        (avg_loss, logits_full) where logits_full are predictions on all edges
        (needed for hard mining in next epoch).
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # Forward on full graph once to get logits for hard mining next epoch
    with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
        logits_full = model(
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
        )

        base_indices = get_balanced_edge_indices(
            data.y, data.train_mask, oversample_ratio=oversample_ratio
        )

        if hard_indices is not None and hard_ratio > 0 and len(hard_indices) > 0:
            train_indices = mix_with_hard_examples(base_indices, hard_indices, hard_ratio=hard_ratio)
        else:
            train_indices = base_indices

        loss = criterion(logits_full[train_indices], data.y[train_indices].to(device))

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

    return loss.item(), logits_full.detach()


# ─── Checkpointing ────────────────────────────────────────────────────────────
def save_checkpoint(
    model: EdgeGATModel,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    metrics: Dict,
    path: str,
) -> None:
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: EdgeGATModel,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Load a checkpoint and return the epoch to resume from."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint.get("epoch", 0)


# ─── Metric Scoring for Early Stopping ────────────────────────────────────────
def constraint_score(
    fpr: float,
    fnr: float,
    max_fpr: float = MAX_FPR,
    max_fnr: float = MAX_FNR,
) -> float:
    """
    Reward thresholds/models that satisfy both constraints.
    Returns a higher score when both FPR and FNR are below thresholds.
    """
    fpr_penalty = max(0.0, fpr - max_fpr)
    fnr_penalty = max(0.0, fnr - max_fnr)
    # Higher is better; base score = 1 - sum of normalized penalties
    return 1.0 - (fpr_penalty + fnr_penalty)


# ─── Main Training Loop ───────────────────────────────────────────────────────
def main():
    logger.info("=" * 60)
    logger.info("AML Detection — GAT Training (GPU Optimized)")
    logger.info("=" * 60)

    # Reproducibility
    set_seed(SEED)
    logger.info(f"Random seed set to {SEED}")

    # ── Load Data ──
    logger.info("[1/6] Loading processed data ...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"  {DATA_PATH} not found! Run data_pipeline.py first.")
        return

    data = torch.load(DATA_PATH, weights_only=False)
    logger.info(f"  Nodes: {data.num_nodes:,} | Edges: {data.edge_index.shape[1]:,}")

    num_pos = int(data.y.sum().item())
    num_neg = int(data.y.shape[0]) - num_pos
    logger.info(
        f"  Class 0: {num_neg:,} | Class 1: {num_pos:,} | Ratio: {num_neg/max(num_pos,1):.1f}:1"
    )

    train_sz = int(data.train_mask.sum().item())
    val_sz = int(data.val_mask.sum().item())
    test_sz = int(data.test_mask.sum().item())
    logger.info(f"  Split sizes: train={train_sz:,} | val={val_sz:,} | test={test_sz:,}")

    # ── Device ──
    device = get_device()
    use_amp = device.type == "cuda"

    # ── Model ──
    logger.info("\n[2/6] Initializing EdgeGATModel ...")
    model = EdgeGATModel(
        node_feat_dim=NODE_FEAT_DIM,
        edge_feat_dim=EDGE_FEAT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_TRAIN,
        use_layer_norm=USE_LAYER_NORM,
        use_residual=USE_RESIDUAL,
    ).to(device)
    logger.info(f"  Parameters: {count_parameters(model):,}")

    if device.type == "cuda":
        logger.info(f"  VRAM after model load: {get_memory_usage_mb(device):.0f} MB")
        torch.cuda.empty_cache()

    # ── Warm start from existing checkpoint (optional) ──
    if WARM_START_FROM_CHECKPOINT and os.path.exists(MODEL_PATH):
        try:
            logger.info(f"  Warm-starting from existing checkpoint: {MODEL_PATH}")
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            logger.info("  Checkpoint loaded successfully.")
        except RuntimeError as e:
            logger.warning(f"  Could not warm-start from checkpoint (architecture mismatch?): {e}")
            logger.warning("  Continuing with randomly initialized weights.")

    # ── Loss, Optimizer, Scheduler, Scaler ──
    criterion = FocalLoss(
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
        label_smoothing=LABEL_SMOOTHING,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Training Loop ──
    oversample_ratio = 5.0
    logger.info(
        f"\n[3/6] Training (epochs={EPOCHS}, patience={PATIENCE}, AMP={'ON' if use_amp else 'OFF'}) ..."
    )
    logger.info(
        f"  {'Ep':>4} | {'T.Loss':>8} | {'V.Loss':>8} | {'V.P':>6} | {'V.R':>6} | {'V.F1':>6} | {'V.FPR%':>7} | {'V.FNR%':>7} | {'VRAM':>6} | {'Time':>6}"
    )
    logger.info("  " + "-" * 80)

    best_score = -1.0
    best_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    history: List[Dict] = []
    hard_indices: Optional[torch.Tensor] = None

    # Checkpoint paths
    checkpoint_dir = os.path.join(os.path.dirname(MODEL_PATH), "checkpoints")
    best_ckpt = os.path.join(checkpoint_dir, "best_checkpoint.pt")
    last_ckpt = os.path.join(checkpoint_dir, "last_checkpoint.pt")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Determine if we use hard mining this epoch
        use_hard_mining = (
            HARD_MINING_ENABLED
            and epoch >= HARD_MINING_START_EPOCH
            and hard_indices is not None
            and len(hard_indices) > 0
        )

        train_loss, train_logits = train_one_epoch(
            model, data, optimizer, criterion, grad_scaler,
            device, use_amp=use_amp, oversample_ratio=oversample_ratio,
            hard_indices=hard_indices if use_hard_mining else None,
            hard_ratio=HARD_MINING_RATIO if use_hard_mining else 0.0,
        )

        # Validation
        val_metrics = evaluate_model(model, data, data.val_mask, criterion, device, use_amp=use_amp)
        scheduler.step()

        # Constraint-based threshold selection on validation
        val_threshold, val_constraint_metrics = find_threshold_with_constraints(
            val_metrics["preds"],
            val_metrics["labels"],
            max_fpr=MAX_FPR,
            max_fnr=MAX_FNR,
            step=THRESHOLD_GRID_STEP,
        )
        val_rates = val_constraint_metrics["rates"]
        val_score = constraint_score(val_rates["fpr"], val_rates["fnr"])

        elapsed = time.time() - t0
        vram_mb = get_memory_usage_mb(device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_fpr": val_rates["fpr"],
            "val_fnr": val_rates["fnr"],
            "val_threshold": val_threshold,
            "val_score": val_score,
            "lr": optimizer.param_groups[0]["lr"],
            "time": elapsed,
            "vram_mb": vram_mb,
        })

        if epoch <= 3 or epoch % 5 == 0 or epoch == EPOCHS or val_score >= 1.0:
            logger.info(
                f"  {epoch:4d} | {train_loss:8.5f} | {val_metrics['loss']:8.5f} | "
                f"{val_metrics['precision']:6.4f} | {val_metrics['recall']:6.4f} | "
                f"{val_metrics['f1']:6.4f} | {val_rates['fpr']*100:7.4f} | {val_rates['fnr']*100:7.4f} | "
                f"{vram_mb:5.0f}M | {format_time(elapsed)}"
            )

        # Prepare hard examples for next epoch (based on train logits)
        if HARD_MINING_ENABLED and epoch >= HARD_MINING_START_EPOCH - 1:
            train_probs = torch.sigmoid(train_logits).cpu()
            hard_indices = get_hard_example_indices(
                data.y, data.train_mask, train_probs, ratio=HARD_MINING_RATIO
            )

        # Save best model based on constraint score (then F1 tie-breaker)
        improved = False
        if val_score > best_score:
            improved = True
        elif abs(val_score - best_score) < 1e-6 and val_metrics["f1"] > best_val_f1:
            improved = True

        if improved:
            best_score = val_score
            best_epoch = epoch
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            save_checkpoint(
                model, optimizer, scheduler, grad_scaler, epoch,
                {"best_score": best_score, "best_epoch": best_epoch},
                best_ckpt,
            )
        else:
            patience_counter += 1

        # Always save last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, grad_scaler, epoch,
            {"best_score": best_score, "best_epoch": best_epoch},
            last_ckpt,
        )

        if patience_counter >= PATIENCE:
            logger.info(f"\n  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

        # Memory cleanup
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"\n  Best Val Constraint Score: {best_score:.6f} at epoch {best_epoch}")

    # ── Test Evaluation ──
    logger.info("\n[4/6] Test evaluation ...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

    # Use validation to select threshold, then apply to test
    val_eval = evaluate_model(model, data, data.val_mask, criterion, device, use_amp=use_amp)
    optimal_thresh, val_constraint = find_threshold_with_constraints(
        val_eval["preds"],
        val_eval["labels"],
        max_fpr=MAX_FPR,
        max_fnr=MAX_FNR,
        step=THRESHOLD_GRID_STEP,
    )
    logger.info(f"  Optimal threshold: {optimal_thresh:.2f} (val FPR={val_constraint['rates']['fpr']*100:.4f}%, FNR={val_constraint['rates']['fnr']*100:.4f}%)")

    test_metrics = evaluate_model(
        model, data, data.test_mask, criterion, device, use_amp=use_amp
    )
    test_preds = test_metrics["preds"]
    test_labels = test_metrics["labels"]
    test_binary = (test_preds >= optimal_thresh).astype(int)

    print_classification_report(test_labels, test_preds, optimal_thresh)
    cm = print_confusion_matrix(test_labels, test_preds, optimal_thresh, title="Test Confusion Matrix")
    rates = compute_error_rates(cm)
    logger.info(
        f"  Test FPR: {rates['fpr']*100:.4f}% | Test FNR: {rates['fnr']*100:.4f}% | "
        f"Accuracy: {rates['accuracy']*100:.4f}%"
    )

    risk_dist = compute_risk_distribution(test_preds)
    logger.info("\n  ── Risk Distribution (Test Set) ──")
    total_test = len(test_preds)
    for cat in ["Low", "Moderate", "High", "Critical"]:
        count = risk_dist[cat]
        pct = count / total_test * 100
        bar = "█" * int(pct / 2)
        logger.info(f"  {cat:>10}: {count:>8,}  ({pct:5.2f}%) {bar}")

    # ── Save ──
    logger.info("\n[5/6] Saving outputs ...")
    metrics_output = {
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "optimal_threshold": optimal_thresh,
        "threshold_selection": "constraint_based",
        "test_metrics": {
            "precision": float(precision_score(test_labels, test_binary, zero_division=0)),
            "recall": float(recall_score(test_labels, test_binary, zero_division=0)),
            "f1": float(f1_score(test_labels, test_binary, zero_division=0)),
            "loss": test_metrics["loss"],
        },
        "confusion_matrix": cm,
        "error_rates": {k: float(v) for k, v in rates.items()},
        "risk_distribution": risk_dist,
        "hyperparameters": {
            "epochs_trained": len(history),
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "focal_alpha": FOCAL_ALPHA,
            "focal_gamma": FOCAL_GAMMA,
            "focal_fn_cost": FN_COST,
            "focal_fp_cost": FP_COST,
            "label_smoothing": LABEL_SMOOTHING,
            "hidden_dim": HIDDEN_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT_TRAIN,
            "use_layer_norm": USE_LAYER_NORM,
            "use_residual": USE_RESIDUAL,
            "threshold": optimal_thresh,
            "oversample_ratio": oversample_ratio,
            "hard_mining_enabled": HARD_MINING_ENABLED,
            "hard_mining_ratio": HARD_MINING_RATIO,
            "seed": SEED,
        },
        "training_history": sanitize_for_json(history),
    }

    save_metrics(metrics_output, METRICS_PATH)
    logger.info(f"  Model: {MODEL_PATH}")
    logger.info(f"  Metrics: {METRICS_PATH}")
    logger.info(f"  Logs: {LOG_PATH}")

    # Final cleanup
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Populate MongoDB ──
    logger.info("\n[6/6] Populating MongoDB ...")
    try:
        populate_mongodb(model, data, metrics_output, device)
    except Exception as e:
        logger.error(f"  MongoDB population failed: {e}")
        logger.error("  Continuing without MongoDB population.")

    logger.info("\n[OK] Training complete!")
    return metrics_output


# ─── MongoDB Population ──────────────────────────────────────────────────────
def populate_mongodb(model, data, metrics_output, device):
    """Run inference on all transactions and populate MongoDB collections."""
    from backend.database import get_sync_db, setup_sync_indexes, close_sync_db

    db = get_sync_db()
    setup_sync_indexes(db)

    logger.info("  Loading CSV and running inference ...")
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
    else:
        dfs = []
        for split_path in [TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH]:
            if os.path.exists(split_path):
                dfs.append(pd.read_csv(split_path))
        df = pd.concat(dfs, ignore_index=True)

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

    df["sender_account"] = df["Sender_account"].astype(str)
    df["receiver_account"] = df["Receiver_account"].astype(str)
    df["amount"] = df["Amount"]
    df["payment_currency"] = df["Payment_currency"]
    df["received_currency"] = df["Received_currency"]
    df["sender_bank_location"] = df["Sender_bank_location"]
    df["receiver_bank_location"] = df["Receiver_bank_location"]
    df["payment_type"] = df["Payment_type"]
    df["is_laundering"] = df["Is_laundering"].astype(int)

    logger.info("  Building account profiles ...")
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

    logger.info("  Inserting transactions into MongoDB ...")
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
        logger.info(f"    Inserted {min(i + batch_size, len(tx_docs)):,} / {len(tx_docs):,} transactions")

    logger.info(f"  Transactions: {len(tx_docs):,} documents")

    logger.info("  Inserting accounts into MongoDB ...")
    db.accounts.drop()
    acc_docs = list(accounts.values())

    for i in range(0, len(acc_docs), batch_size):
        batch = acc_docs[i:i + batch_size]
        db.accounts.insert_many(batch, ordered=False)
        logger.info(f"    Inserted {min(i + batch_size, len(acc_docs)):,} / {len(acc_docs):,} accounts")

    logger.info(f"  Accounts: {len(acc_docs):,} documents")

    logger.info("  Inserting training metrics ...")
    metrics_doc = metrics_output.copy()
    metrics_doc["run_id"] = datetime.now().isoformat()
    metrics_doc["created_at"] = datetime.now()

    for h in metrics_doc.get("training_history", []):
        for k, v in h.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                h[k] = None

    db.training_metrics.insert_one(metrics_doc)
    logger.info("  Training metrics: 1 document")

    setup_sync_indexes(db)
    close_sync_db()
    logger.info("  MongoDB population complete!")


if __name__ == "__main__":
    main()
