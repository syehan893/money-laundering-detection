"""
scripts/retrain_gat.py — One-command GAT retraining pipeline.
===============================================================
This script runs the full AML detection training pipeline:
    1. Data preprocessing & graph construction (if needed).
    2. GAT model training with cost-sensitive focal loss + hard mining.
    3. Test evaluation and metrics saving.

Usage:
    python scripts/retrain_gat.py [--force-pipeline]

Options:
    --force-pipeline   Re-run data_pipeline.py even if processed_data.pt exists.
"""

import argparse
import os
import sys

# Add project root to Python path
def _find_root(start: str) -> str:
    """Akar proyek = direktori terdekat di atas berkas ini yang memuat backend/config.py."""
    d = os.path.dirname(os.path.abspath(start))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "backend", "config.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"Akar proyek tidak ditemukan dari {start}")


PROJECT_ROOT = _find_root(__file__)
sys.path.insert(0, PROJECT_ROOT)

from backend.config import DATA_PATH
from backend.ml.legacy.data_pipeline import main as run_data_pipeline
from backend.ml.legacy.train import main as run_training


def main():
    parser = argparse.ArgumentParser(
        description="Retrain the AML GAT model end-to-end."
    )
    parser.add_argument(
        "--force-pipeline",
        action="store_true",
        help="Re-run data pipeline even if processed_data.pt exists.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AML Guard — End-to-End GAT Retraining")
    print("=" * 60)

    # Step 1: Data pipeline
    if not os.path.exists(DATA_PATH) or args.force_pipeline:
        print("\n[Step 1/3] Running data pipeline ...")
        run_data_pipeline()
    else:
        print(f"\n[Step 1/3] Skipping data pipeline (found {DATA_PATH}).")
        print("            Use --force-pipeline to regenerate.")

    # Step 2: Training
    print("\n[Step 2/3] Running GAT training ...")
    metrics = run_training()

    # Step 3: Summary
    print("\n[Step 3/3] Final test metrics summary:")
    cm = metrics.get("confusion_matrix", {})
    rates = metrics.get("error_rates", {})
    print(f"  Threshold: {metrics.get('optimal_threshold', 0.5):.2f}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {cm.get('tn', 0):,}  FP: {cm.get('fp', 0):,}")
    print(f"    FN: {cm.get('fn', 0):,}  TP: {cm.get('tp', 0):,}")
    print(f"  FPR: {rates.get('fpr', 0.0) * 100:.4f}%")
    print(f"  FNR: {rates.get('fnr', 0.0) * 100:.4f}%")
    print(f"  Precision: {metrics.get('test_metrics', {}).get('precision', 0.0):.4f}")
    print(f"  Recall:    {metrics.get('test_metrics', {}).get('recall', 0.0):.4f}")
    print(f"  F1:        {metrics.get('test_metrics', {}).get('f1', 0.0):.4f}")

    target_met = (
        rates.get("fpr", 1.0) < 0.001 and rates.get("fnr", 1.0) < 0.001
    )
    if target_met:
        print("\n[OK] Target achieved: FPR < 0.1% and FNR < 0.1% per class.")
    else:
        print("\n[WARN] Target not yet achieved. Consider tuning FN_COST, FOCAL_ALPHA, or epochs.")

    print("\n[OK] Retraining pipeline complete.")


if __name__ == "__main__":
    main()
