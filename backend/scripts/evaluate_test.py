"""
evaluate_test.py — Standalone Test Set Evaluation
==================================================
Load trained model and evaluate ONLY on the test set.
Useful after training to get final test metrics without retraining.

Usage:
    python -m backend.scripts.evaluate_test
"""

import gc
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.ml.model import EdgeGATModel, FocalLoss, get_risk_category
from backend.config import DATA_PATH, MODEL_PATH, METRICS_PATH, ENCODERS_PATH
from backend.config import FOCAL_ALPHA, FOCAL_GAMMA


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.set_per_process_memory_fraction(0.95, 0)
    else:
        device = torch.device("cpu")
        print("  Using CPU")
    return device


def evaluate_test():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  AML Detection — Test Set Evaluation                     ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    # ── Load data ──
    print("[1/3] Loading processed data ...")
    if not os.path.exists(DATA_PATH):
        print(f"  ❌ {DATA_PATH} not found! Run data_pipeline.py first.")
        return
    data = torch.load(DATA_PATH, weights_only=False)
    print(f"  Nodes: {data.num_nodes:,} | Edges: {data.edge_index.shape[1]:,}")

    test_sz = int(data.test_mask.sum().item())
    print(f"  Test set size: {test_sz:,} edges")

    # ── Load model ──
    print("\n[2/3] Loading model ...")
    device = get_device()
    model = EdgeGATModel(
        node_feat_dim=data.x.shape[1],
        edge_feat_dim=data.edge_attr.shape[1],
        hidden_dim=64,
        num_heads=4,
        dropout=0.0,  # inference: no dropout
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"  ❌ {MODEL_PATH} not found! Train the model first.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    print(f"  ✓ Model loaded from {MODEL_PATH}")

    # ── Evaluate on test set ──
    print("\n[3/3] Evaluating on test set ...")
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(
                data.x.to(device),
                data.edge_index.to(device),
                data.edge_attr.to(device),
            )

    probs = torch.sigmoid(logits[data.test_mask]).cpu().numpy()
    labels = data.y[data.test_mask].cpu().numpy()
    loss = criterion(logits[data.test_mask], data.y[data.test_mask].to(device)).item()

    # Find optimal threshold on validation, then apply to test
    val_probs = torch.sigmoid(logits[data.val_mask]).cpu().numpy()
    val_labels = data.y[data.val_mask].cpu().numpy()

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(val_labels, (val_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"  Optimal threshold (from val): {best_t:.2f} (val F1={best_f1:.4f})")

    binary_preds = (probs >= best_t).astype(int)
    precision = precision_score(labels, binary_preds, zero_division=0)
    recall = recall_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)

    print("\n  ── Test Set Metrics ──")
    print(f"  Loss      : {loss:.5f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    print("\n  ── Classification Report ──")
    print(classification_report(labels, binary_preds, target_names=["Normal (0)", "Laundering (1)"], digits=4))

    cm = confusion_matrix(labels, binary_preds)
    print("  ── Confusion Matrix ──")
    print(f"                    Predicted 0    Predicted 1")
    print(f"  Actual 0 (Normal)     {cm[0][0]:>8,}       {cm[0][1]:>8,}")
    print(f"  Actual 1 (Launder)    {cm[1][0]:>8,}       {cm[1][1]:>8,}")

    # ── Risk category distribution ──
    print("\n  ── Risk Distribution (Test Set) ──")
    risk_counts = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
    for p in probs:
        risk_counts[get_risk_category(float(p))] += 1
    total_test = len(probs)
    for cat in ["Low", "Moderate", "High", "Critical"]:
        count = risk_counts[cat]
        pct = count / total_test * 100
        bar = "█" * int(pct / 2)
        print(f"  {cat:>10}: {count:>8,}  ({pct:5.2f}%) {bar}")

    # Save test metrics
    test_output = {
        "threshold": float(best_t),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_loss": float(loss),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        },
    }

    out_path = os.path.join(os.path.dirname(METRICS_PATH), "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(test_output, f, indent=2)
    print(f"\n  ✓ Test metrics saved to: {out_path}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print("\n✅ Test evaluation complete!\n")


if __name__ == "__main__":
    evaluate_test()
