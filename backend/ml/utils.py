"""
utils.py — Reusable helpers for AML GAT training & inference.
===============================================================
Includes reproducibility seeding, device selection, parameter counting,
and small formatting utilities.
"""

import os
import random
import time
from typing import Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across NumPy, Python random,
    and PyTorch (CPU & CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can hurt performance; keep False by default.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(memory_fraction: float = 0.95) -> torch.device:
    """
    Select CUDA if available, otherwise CPU. Optionally cap VRAM usage.

    Args:
        memory_fraction: Max fraction of GPU memory this process may allocate.

    Returns:
        torch.device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name} ({total_mem:.1f} GB VRAM)")
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
        print(f"  VRAM limit: {total_mem * memory_fraction:.1f} GB ({memory_fraction*100:.0f}%)")
        print("  Mixed precision (AMP): ENABLED")
    else:
        device = torch.device("cpu")
        print("  Using CPU (no CUDA available)")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds as mm:ss or hh:mm:ss."""
    seconds = int(seconds)
    if seconds < 3600:
        return time.strftime("%M:%S", time.gmtime(seconds))
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def get_memory_usage_mb(device: torch.device) -> float:
    """Return current GPU memory allocated in MB, or 0 for CPU."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024**2)
    return 0.0


def sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


def compute_class_weights(labels: torch.Tensor) -> Tuple[float, float]:
    """
    Compute inverse-frequency class weights for binary labels.

    Returns:
        (weight_for_class_0, weight_for_class_1)
    """
    n_pos = int(labels.sum().item())
    n_neg = labels.numel() - n_pos
    total = n_pos + n_neg
    w_pos = total / (2.0 * max(n_pos, 1))
    w_neg = total / (2.0 * max(n_neg, 1))
    return w_neg, w_pos
