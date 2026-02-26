"""
config.py — Centralized Configuration
=======================================
All paths, constants, and environment variables in one place.
"""

import os

# ─── Project Roots ────────────────────────────────────────────────────────────
# BASE_DIR = money-laundering-detection/ (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
DATA_DIR = os.path.join(BASE_DIR, "data")


def _resolve(filename: str) -> str:
    """Resolve data file path: prefer data/ dir, fallback to project root."""
    data_path = os.path.join(DATA_DIR, filename)
    root_path = os.path.join(BASE_DIR, filename)
    if os.path.exists(data_path):
        return data_path
    if os.path.exists(root_path):
        return root_path
    # Default to data/ for new files
    return data_path


# ─── Data Paths ───────────────────────────────────────────────────────────────
CSV_PATH = _resolve("SAML-D.csv")
DATA_PATH = _resolve("processed_data.pt")
MODEL_PATH = _resolve("best_model.pt")
ENCODERS_PATH = _resolve("encoders.pkl")
METRICS_PATH = _resolve("training_metrics.json")

# ─── Training Hyperparameters ─────────────────────────────────────────────────
EPOCHS = 150
LR = 0.0005
WEIGHT_DECAY = 1e-5
PATIENCE = 40
FOCAL_ALPHA = 0.90
FOCAL_GAMMA = 2.0

# ─── Model Architecture ──────────────────────────────────────────────────────
NODE_FEAT_DIM = 7
EDGE_FEAT_DIM = 8
HIDDEN_DIM = 64
NUM_HEADS = 4
DROPOUT_TRAIN = 0.3
DROPOUT_INFERENCE = 0.0
