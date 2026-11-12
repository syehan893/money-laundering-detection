"""
evaluate.py — Model evaluation, threshold selection, and metrics persistence.
================================================================================
Provides memory-efficient evaluation and threshold tuning strategies,
including constraint-based selection for FPR/FNR targets.
"""

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from backend.ml.legacy.model import EdgeGATModel, FocalLoss, get_risk_category


def evaluate_model(
    model: EdgeGATModel,
    data,
    mask: torch.Tensor,
    criterion: Optional[FocalLoss] = None,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = True,
) -> Dict:
    """
    Memory-efficient evaluation on a subset of edges.

    Args:
        model: The GAT model.
        data: PyTorch Geometric Data object.
        mask: Boolean mask selecting edges to evaluate.
        criterion: Optional FocalLoss instance to compute loss.
        device: torch.device.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Dictionary with loss, precision, recall, f1, preds (probabilities),
        binary_preds, and labels.
    """
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits = model(
                data.x.to(device),
                data.edge_index.to(device),
                data.edge_attr.to(device),
            )

        probs_masked = torch.sigmoid(logits[mask]).cpu().numpy()
        labels_masked = data.y[mask].cpu().numpy()

        loss = 0.0
        if criterion is not None:
            loss = criterion(logits[mask], data.y[mask].to(device)).item()

    binary_preds = (probs_masked >= 0.5).astype(int)
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


def compute_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """
    Compute binary confusion matrix components.

    Returns:
        Dict with keys 'tn', 'fp', 'fn', 'tp'.
    """
    binary_preds = (preds >= threshold).astype(int)
    cm = confusion_matrix(labels, binary_preds)
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_error_rates(cm: Dict[str, int]) -> Dict[str, float]:
    """
    Compute FPR, FNR, TPR, TNR from confusion matrix components.
    """
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    fpr = fp / max((tn + fp), 1)
    fnr = fn / max((tp + fn), 1)
    tpr = tp / max((tp + fn), 1)
    tnr = tn / max((tn + fp), 1)
    return {
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "tnr": tnr,
        "accuracy": (tp + tn) / max((tp + tn + fp + fn), 1),
    }


def find_best_threshold(
    preds: np.ndarray,
    labels: np.ndarray,
    step: float = 0.05,
) -> Tuple[float, float]:
    """Find threshold that maximizes F1 score."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.95, step):
        f1 = f1_score(labels, (preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t), float(best_f1)


def find_threshold_with_constraints(
    preds: np.ndarray,
    labels: np.ndarray,
    max_fpr: float = 0.001,
    max_fnr: float = 0.001,
    step: float = 0.01,
) -> Tuple[float, Dict]:
    """
    Select the best threshold satisfying both FPR <= max_fpr and FNR <= max_fnr.
    If no threshold satisfies both, choose the one minimizing total errors.

    Returns:
        (threshold, metrics_dict)
    """
    best_constraint_candidate = None
    best_constraint_metrics = None
    best_fallback_candidate = None
    best_fallback_metrics = None

    # Fine grid search from step to just below 1.0
    thresholds = np.arange(step, 1.0, step)

    for t in thresholds:
        cm = compute_confusion_matrix(labels, preds, threshold=t)
        rates = compute_error_rates(cm)

        metrics = {
            "threshold": float(t),
            "cm": cm,
            "rates": rates,
            "total_errors": cm["fp"] + cm["fn"],
        }

        # Track thresholds meeting both constraints
        if rates["fpr"] <= max_fpr and rates["fnr"] <= max_fnr:
            if (
                best_constraint_candidate is None
                or metrics["total_errors"] < best_constraint_metrics["total_errors"]
            ):
                best_constraint_candidate = t
                best_constraint_metrics = metrics
        else:
            # Track fallback with minimal total error
            if (
                best_fallback_candidate is None
                or metrics["total_errors"] < best_fallback_metrics["total_errors"]
            ):
                best_fallback_candidate = t
                best_fallback_metrics = metrics

    # Prefer constraint-satisfying threshold, fallback to minimal-error threshold
    if best_constraint_candidate is not None:
        return float(best_constraint_candidate), best_constraint_metrics

    if best_fallback_candidate is not None:
        return float(best_fallback_candidate), best_fallback_metrics

    # Ultimate fallback
    default_metrics = {
        "threshold": 0.5,
        "cm": compute_confusion_matrix(labels, preds, threshold=0.5),
        "rates": compute_error_rates(compute_confusion_matrix(labels, preds, threshold=0.5)),
        "total_errors": int(labels.shape[0]),
    }
    return 0.5, default_metrics


def compute_risk_distribution(preds: np.ndarray) -> Dict[str, int]:
    """Bucket probabilities into Low / Moderate / High / Critical."""
    risk_counts = {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
    for p in preds:
        risk_counts[get_risk_category(float(p))] += 1
    return risk_counts


def print_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    threshold: float,
    title: str = "Confusion Matrix",
) -> Dict[str, int]:
    """Pretty-print confusion matrix and error rates."""
    cm = compute_confusion_matrix(labels, preds, threshold)
    rates = compute_error_rates(cm)

    print(f"\n  ── {title} (threshold={threshold:.2f}) ──")
    print(f"                    Predicted 0    Predicted 1")
    print(f"  Actual 0 (Normal)     {cm['tn']:>8,}       {cm['fp']:>8,}")
    print(f"  Actual 1 (Launder)    {cm['fn']:>8,}       {cm['tp']:>8,}")
    print(f"  FPR: {rates['fpr']*100:.4f}% | FNR: {rates['fnr']*100:.4f}% | Acc: {rates['accuracy']*100:.4f}%")
    return cm


def print_classification_report(
    labels: np.ndarray,
    preds: np.ndarray,
    threshold: float,
) -> None:
    """Print sklearn classification report."""
    binary_preds = (preds >= threshold).astype(int)
    report = classification_report(
        labels,
        binary_preds,
        target_names=["Normal (0)", "Laundering (1)"],
        digits=4,
    )
    print("\n  ── Classification Report ──")
    print(report)


def save_metrics(
    metrics_output: Dict,
    path: str,
) -> None:
    """Save metrics dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_output, f, indent=2)
