"""
model.py — GAT Model for Edge-Level AML Classification
===============================================================
Implements EdgeGATModel (Graph Attention Network with edge-aware
classification) and FocalLoss for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the positive class (0-1).
        gamma: Focusing parameter — higher gamma = more focus on hard examples.
        reduction: 'mean' | 'sum' | 'none'.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predicted probabilities (after sigmoid), shape (N,).
            targets: Ground truth labels (0 or 1), shape (N,).
        """
        # Clamp for numerical stability
        p = inputs.clamp(min=1e-7, max=1 - 1e-7)

        # Binary cross entropy per sample
        bce = -targets * torch.log(p) - (1 - targets) * torch.log(1 - p)

        # p_t: probability of correct class
        p_t = targets * p + (1 - targets) * (1 - p)

        # Alpha weighting
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Focal modulation
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

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
        1. GAT Encoder (2 layers) → node embeddings
        2. Edge Feature Transform → edge feature projection
        3. Edge Classifier MLP → per-edge probability

    Args:
        node_feat_dim: Number of node input features (7).
        edge_feat_dim: Number of edge input features (8).
        hidden_dim: Hidden dimension per GAT head (default: 64).
        num_heads: Number of attention heads (default: 4).
        dropout: Dropout rate (default: 0.3).
    """

    def __init__(
        self,
        node_feat_dim: int = 7,
        edge_feat_dim: int = 8,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.dropout = dropout

        # ── GAT Encoder ──
        # Layer 1: node_feat_dim → hidden_dim * num_heads
        self.gat1 = GATConv(
            in_channels=node_feat_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,  # Output: hidden_dim * num_heads
        )

        # Layer 2: hidden_dim * num_heads → hidden_dim (single head output)
        self.gat2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False,  # Output: hidden_dim (no concat)
        )

        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        node_emb_dim = hidden_dim  # Single-head output

        # ── Edge Feature Transform ──
        edge_proj_dim = 32
        self.edge_transform = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_proj_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # ── Edge Classifier MLP ──
        # Input: src_emb + dst_emb + edge_feat
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

        # Chunk size for edge processing (memory control)
        self.edge_chunk_size = 200000

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
            predictions: Per-edge probabilities (num_edges,).
        """
        # ── GAT Encoder ──
        h = self.gat1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.gat2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ── Process edges in chunks to save VRAM ──
        src, dst = edge_index[0], edge_index[1]
        num_edges = src.shape[0]
        chunk = self.edge_chunk_size
        all_preds = []

        for i in range(0, num_edges, chunk):
            end = min(i + chunk, num_edges)
            src_emb = h[src[i:end]]
            dst_emb = h[dst[i:end]]
            edge_feat = self.edge_transform(edge_attr[i:end])
            edge_repr = torch.cat([src_emb, dst_emb, edge_feat], dim=1)
            logits = self.classifier(edge_repr).squeeze(-1)
            all_preds.append(torch.sigmoid(logits))

        return torch.cat(all_preds)

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Get node embeddings only (for visualization)."""
        with torch.no_grad():
            h = self.gat1(x, edge_index)
            h = self.bn1(h)
            h = F.elu(h)

            h = self.gat2(h, edge_index)
            h = self.bn2(h)
            h = F.elu(h)

        return h


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


if __name__ == "__main__":
    # Quick sanity check
    print("EdgeGATModel — Architecture Summary")
    print("=" * 50)
    model = EdgeGATModel(node_feat_dim=7, edge_feat_dim=8)
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Dummy forward pass
    x = torch.randn(100, 7)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 8)

    model.eval()
    with torch.no_grad():
        preds = model(x, edge_index, edge_attr)
    print(f"Output shape: {preds.shape}")
    print(f"Output range: [{preds.min():.4f}, {preds.max():.4f}]")
    print("\n✅ Model sanity check passed!")
