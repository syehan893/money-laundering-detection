"""
explore_data.py — Quick verification that the pipeline output is valid.

Run AFTER data_pipeline.py:
    python -m backend.ml.data_pipeline
    python -m backend.scripts.explore_data
"""

import os
import sys

import torch
import joblib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import DATA_PATH, ENCODERS_PATH


def verify():
    print("=" * 50)
    print("  Verifying processed_data.pt")
    print("=" * 50)

    if not os.path.exists(DATA_PATH):
        print("❌ processed_data.pt not found! Run data_pipeline.py first.")
        return

    data = torch.load(DATA_PATH, weights_only=False)
    print(f"  ✓ Loaded Data object")
    print(f"    x (node features)    : {data.x.shape}")
    print(f"    edge_index           : {data.edge_index.shape}")
    print(f"    edge_attr            : {data.edge_attr.shape}")
    print(f"    y (labels)           : {data.y.shape}")
    print(f"    num_nodes            : {data.num_nodes}")
    print(f"    train_mask sum       : {data.train_mask.sum().item()}")
    print(f"    val_mask sum         : {data.val_mask.sum().item()}")
    print(f"    test_mask sum        : {data.test_mask.sum().item()}")
    print()

    # Sanity checks
    assert data.edge_index.shape[0] == 2, "edge_index should be 2 x num_edges"
    assert data.edge_index.shape[1] == data.y.shape[0], "edge count mismatch"
    assert data.edge_attr.shape[0] == data.y.shape[0], "edge_attr count mismatch"
    assert data.x.shape[0] == data.num_nodes, "node count mismatch"
    assert not torch.isnan(data.x).any(), "NaN in node features"
    assert not torch.isnan(data.edge_attr).any(), "NaN in edge features"
    print("  ✓ All sanity checks passed!")
    print()

    # Verify encoders
    if os.path.exists(ENCODERS_PATH):
        artifacts = joblib.load(ENCODERS_PATH)
        print(f"  ✓ Loaded encoders.pkl")
        print(f"    Keys: {list(artifacts.keys())}")
        print(f"    Node map size: {len(artifacts['node_map'])}")
        print(f"    Encoder keys: {list(artifacts['encoders'].keys())}")
    else:
        print("❌ encoders.pkl not found!")

    print("\n✅ Verification complete!")


if __name__ == "__main__":
    verify()
