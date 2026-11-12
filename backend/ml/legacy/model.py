"""
model.py — GAT Model for Edge-Level AML Classification
===============================================================
Implements EdgeGATModel (Graph Attention Network with edge-aware
classification) and FocalLoss for handling class imbalance.

Improvements in this version:
  - Configurable depth (num_layers >= 2).
  - Optional LayerNorm and residual connections.
  - Cost-sensitive FocalLoss with label smoothing.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.

    Supports:
      - Standard focal modulation (alpha, gamma).
      - Cost-sensitive weighting (fn_cost, fp_cost) to penalize
        false negatives more heavily.
      - Label smoothing for regularization.

    FL(p_t) = -alpha_t * cost_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 2.0,
        fn_cost: float = 1.0,
        fp_cost: float = 1.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.fn_cost = fn_cost
        self.fp_cost = fp_cost
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw logits (before sigmoid), shape (N,).
            targets: Ground truth labels (0 or 1), shape (N,).
        """
        # Apply label smoothing to hard targets
        if self.label_smoothing > 0:
            targets = targets.clamp(0.0, 1.0)
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Numerically stable BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p = torch.sigmoid(logits)
        p_t = targets * p + (1 - targets) * (1 - p)

        # Class weight (alpha)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Cost-sensitive weighting: FN cost on positives, FP cost on negatives
        cost_t = targets * self.fn_cost + (1 - targets) * self.fp_cost

        focal_weight = alpha_t * cost_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class EdgeGATModel(nn.Module):
    """
    Graph Attention Network for edge-level binary classification.

    Architecture:
        1. GAT Encoder (configurable depth) -> node embeddings
        2. Edge Feature Transform -> edge feature projection
        3. Edge Classifier MLP -> per-edge probability

    Args:
        node_feat_dim: Number of node input features.
        edge_feat_dim: Number of edge input features.
        hidden_dim: Hidden dimension per GAT head.
        num_heads: Number of attention heads in first GAT layer.
        num_layers: Number of GAT encoder layers (>= 2).
        dropout: Dropout rate.
        use_layer_norm: Use LayerNorm instead of BatchNorm between GAT layers.
        use_residual: Add residual connections between GAT layers.
    """

    def __init__(
        self,
        node_feat_dim: int = 7,
        edge_feat_dim: int = 8,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # ── GAT Encoder ──
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()

        # First layer: node_feat_dim -> hidden_dim * num_heads
        self.gat_layers.append(
            GATConv(
                in_channels=node_feat_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True,
            )
        )

        # Hidden layers: use single-head to keep memory manageable
        for i in range(1, num_layers - 1):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    heads=1,
                    dropout=dropout,
                    concat=False,
                )
            )

        # Last layer: hidden_dim -> hidden_dim (single head)
        last_in_channels = hidden_dim * num_heads if num_layers == 2 else hidden_dim
        self.gat_layers.append(
            GATConv(
                in_channels=last_in_channels,
                out_channels=hidden_dim,
                heads=1,
                dropout=dropout,
                concat=False,
            )
        )

        for i in range(num_layers):
            out_dim = hidden_dim * num_heads if i == 0 else hidden_dim
            if use_layer_norm:
                self.gat_norms.append(nn.LayerNorm(out_dim))
            else:
                self.gat_norms.append(nn.BatchNorm1d(out_dim))

        node_emb_dim = hidden_dim

        # ── Edge Feature Transform ──
        edge_proj_dim = 32
        self.edge_transform = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_proj_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # ── Edge Classifier MLP ──
        classifier_input_dim = node_emb_dim * 2 + edge_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.edge_chunk_size = 200_000

    def _gat_forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply GAT encoder layers with optional residual connections."""
        h = x
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            h_out = gat(h, edge_index)
            h_out = norm(h_out)
            h_out = F.elu(h_out)
            h_out = F.dropout(h_out, p=self.dropout, training=self.training)

            # Residual connection if dimensions match
            if self.use_residual and i > 0 and h.shape == h_out.shape:
                h_out = h_out + h

            h = h_out
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, node_feat_dim).
            edge_index: Edge indices (2, num_edges).
            edge_attr: Edge features (num_edges, edge_feat_dim).

        Returns:
            predictions: Per-edge raw logits (num_edges,).
        """
        h = self._gat_forward(x, edge_index)

        src, dst = edge_index[0], edge_index[1]
        num_edges = src.shape[0]
        chunk = self.edge_chunk_size
        all_logits = []

        for i in range(0, num_edges, chunk):
            end = min(i + chunk, num_edges)
            src_emb = h[src[i:end]]
            dst_emb = h[dst[i:end]]
            edge_feat = self.edge_transform(edge_attr[i:end])
            edge_repr = torch.cat([src_emb, dst_emb, edge_feat], dim=1)
            logits = self.classifier(edge_repr).squeeze(-1)
            all_logits.append(logits)

        return torch.cat(all_logits)

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Inference mode — returns probabilities (0-1)."""
        logits = self.forward(x, edge_index, edge_attr)
        return torch.sigmoid(logits)

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Get node embeddings only (for visualization / transfer learning)."""
        with torch.no_grad():
            return self._gat_forward(x, edge_index)


def get_risk_category(prob: float) -> str:
    """Classify a probability into a risk category."""
    if prob < 0.30:
        return "Low"
    elif prob < 0.70:
        return "Moderate"
    elif prob < 0.90:
        return "High"
    else:
        return "Critical"


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_confusion_matrix_rates(
    tn: int, fp: int, fn: int, tp: int
) -> dict:
    """Compute FPR, FNR, TPR, TNR from confusion matrix counts."""
    fpr = fp / max((tn + fp), 1)
    fnr = fn / max((tp + fn), 1)
    tpr = tp / max((tp + fn), 1)
    tnr = tn / max((tn + fp), 1)
    return {
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "tnr": tnr,
        "precision": tp / max((tp + fp), 1),
        "recall": tpr,
        "accuracy": (tp + tn) / max((tp + tn + fp + fn), 1),
    }


if __name__ == "__main__":
    print("EdgeGATModel — Architecture Summary")
    print("=" * 50)
    model = EdgeGATModel(node_feat_dim=7, edge_feat_dim=8, num_layers=3)
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    x = torch.randn(100, 7)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 8)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        probs = model.predict(x, edge_index, edge_attr)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"Probs shape: {probs.shape}")
    print(f"Probs range: [{probs.min():.4f}, {probs.max():.4f}]")
    print("\n[OK] Model sanity check passed!")
