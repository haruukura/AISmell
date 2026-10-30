"""
model.py — Message Passing Neural Network with Principal Odor Map (MPNN-POM).

Architecture (per the Science 2023 paper):
  1. Atom & bond embedding layers (linear projections)
  2. 5 rounds of message passing (neighbor aggregation + GRU update)
  3. Graph-level attention pooling → fixed-length molecular vector
  4. Dense 256-dim Principal Odor Map (POM) layer with dropout=0.12
  5. Multi-label classification head → 138 sigmoid logits

The 256-dim POM embedding is the "principal odor map" where nearby
molecules smell alike.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
# Building blocks
# ===================================================================

class EdgeNetwork(nn.Module):
    """Maps edge features into a weight matrix for message transformation.

    edge_feats → (hidden_dim × hidden_dim) matrix used to transform
    neighbour node embeddings during message passing.
    """

    def __init__(self, edge_feat_dim: int, hidden_dim: int):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """(E, edge_feat_dim) → (E, hidden_dim, hidden_dim)"""
        return self.nn(edge_feats).view(-1, self.hidden_dim, self.hidden_dim)


class MessagePassingLayer(nn.Module):
    """One round of MPNN message passing.

    For each atom v:
        m_v = Σ_{u ∈ N(v)}  W(e_uv) · h_u
        h_v = GRU(h_v, m_v)
    """

    def __init__(self, hidden_dim: int, edge_feat_dim: int):
        super().__init__()
        self.edge_network = EdgeNetwork(edge_feat_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        node_feats: torch.Tensor,    # (N, hidden_dim)
        edge_index: torch.Tensor,    # (2, E)
        edge_feats: torch.Tensor,    # (E, edge_feat_dim)
    ) -> torch.Tensor:
        """Returns updated node features (N, hidden_dim)."""
        src, dst = edge_index  # dst receives messages from src
        num_nodes = node_feats.size(0)

        if edge_feats.size(0) == 0:
            # No edges — skip message passing
            return node_feats

        # Compute edge weight matrices
        W = self.edge_network(edge_feats)  # (E, H, H)

        # Gather source node embeddings and transform
        h_src = node_feats[src]  # (E, H)
        messages = torch.bmm(W, h_src.unsqueeze(-1)).squeeze(-1)  # (E, H)

        # Aggregate messages at destination nodes (sum pooling)
        agg = torch.zeros_like(node_feats)
        agg.index_add_(0, dst, messages)

        # GRU update
        updated = self.gru(agg, node_feats)
        return updated


class GlobalAttentionPooling(nn.Module):
    """Attention-weighted graph-level readout.

    Learns a gating vector per node, then computes
        h_graph = Σ_v  softmax(gate(h_v)) · transform(h_v)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.transform_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        node_feats: torch.Tensor,   # (total_nodes, H)
        batch_idx: torch.Tensor,    # (total_nodes,) graph membership
        num_graphs: int,
    ) -> torch.Tensor:
        """Returns (num_graphs, hidden_dim)."""
        gate_scores = self.gate_nn(node_feats)  # (N, 1)
        transformed = self.transform_nn(node_feats)  # (N, H)

        # Per-graph softmax of gate scores
        # Subtract max for numerical stability before exp
        gate_max = torch.zeros(num_graphs, 1, device=gate_scores.device)
        gate_max.scatter_reduce_(0, batch_idx.unsqueeze(1), gate_scores, reduce="amax")
        gate_scores = gate_scores - gate_max[batch_idx]
        gate_exp = gate_scores.exp()  # (N, 1)

        # Normalization denominator per graph
        gate_sum = torch.zeros(num_graphs, 1, device=gate_exp.device)
        gate_sum.scatter_add_(0, batch_idx.unsqueeze(1), gate_exp)
        gate_sum = gate_sum.clamp(min=1e-8)

        attention = gate_exp / gate_sum[batch_idx]  # (N, 1)

        weighted = attention * transformed  # (N, H)

        # Sum per graph
        out = torch.zeros(num_graphs, node_feats.size(1), device=node_feats.device)
        out.scatter_add_(0, batch_idx.unsqueeze(1).expand_as(weighted), weighted)
        return out


# ===================================================================
# Full Model
# ===================================================================

class MPNNPOMModel(nn.Module):
    """Message Passing Neural Network with Principal Odor Map.

    Parameters
    ----------
    node_feat_dim : int
        Dimension of input atom feature vectors.
    edge_feat_dim : int
        Dimension of input bond feature vectors.
    hidden_dim : int
        Hidden dimension for node embeddings (default: 128).
    num_message_passing : int
        Number of message passing rounds (default: 5, per paper).
    pom_dim : int
        Dimension of the Principal Odor Map embedding (default: 256).
    num_tasks : int
        Number of odor descriptor labels (default: 138).
    dropout : float
        Dropout rate on POM layer (default: 0.12).
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int = 128,
        num_message_passing: int = 5,
        pom_dim: int = 256,
        num_tasks: int = 138,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pom_dim = pom_dim
        self.num_tasks = num_tasks

        # Embedding projections
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)

        # Message passing layers
        self.mp_layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim, hidden_dim) for _ in range(num_message_passing)]
        )

        # Graph-level readout
        self.pooling = GlobalAttentionPooling(hidden_dim)

        # Principal Odor Map (POM) dense embedding
        self.pom_layer = nn.Sequential(
            nn.Linear(hidden_dim, pom_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Linear(pom_dim, num_tasks)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch_idx: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Forward pass returning raw logits (before sigmoid).

        Parameters
        ----------
        node_feats : (total_atoms, node_feat_dim)
        edge_index : (2, total_edges)
        edge_feats : (total_edges, edge_feat_dim)
        batch_idx  : (total_atoms,) — graph index per atom
        num_graphs : int

        Returns
        -------
        logits : (num_graphs, num_tasks)
        """
        # Encode
        h = self.node_encoder(node_feats)               # (N, H)
        e = self.edge_encoder(edge_feats) if edge_feats.size(0) > 0 else edge_feats

        # Message passing
        for mp in self.mp_layers:
            h = mp(h, edge_index, e)

        # Pool to graph-level
        graph_repr = self.pooling(h, batch_idx, num_graphs)  # (B, H)

        # POM embedding + classification
        pom = self.pom_layer(graph_repr)                      # (B, pom_dim)
        logits = self.classifier(pom)                         # (B, num_tasks)
        return logits

    def extract_pom(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        batch_idx: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Extract the 256-dim POM vector (no classification head).

        Returns (num_graphs, pom_dim).
        """
        h = self.node_encoder(node_feats)
        e = self.edge_encoder(edge_feats) if edge_feats.size(0) > 0 else edge_feats
        for mp in self.mp_layers:
            h = mp(h, edge_index, e)
        graph_repr = self.pooling(h, batch_idx, num_graphs)
        pom = self.pom_layer(graph_repr)
        return pom

    def forward_from_batch(self, batch: dict) -> torch.Tensor:
        """Convenience: forward pass from a collated batch dict."""
        return self.forward(
            node_feats=batch["node_feats"],
            edge_index=batch["edge_index"],
            edge_feats=batch["edge_feats"],
            batch_idx=batch["batch_idx"],
            num_graphs=batch["num_graphs"],
        )

    def pom_from_batch(self, batch: dict) -> torch.Tensor:
        """Convenience: POM extraction from a collated batch dict."""
        return self.extract_pom(
            node_feats=batch["node_feats"],
            edge_index=batch["edge_index"],
            edge_feats=batch["edge_feats"],
            batch_idx=batch["batch_idx"],
            num_graphs=batch["num_graphs"],
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke test with synthetic data
    N, H_node, H_edge = 10, 45, 8
    node_feats = torch.randn(N, H_node)
    edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,1],
                                [1,0,3,2,5,4,7,6,9,8]], dtype=torch.long)
    edge_feats = torch.randn(10, H_edge)
    batch_idx = torch.tensor([0]*5 + [1]*5, dtype=torch.long)

    model = MPNNPOMModel(node_feat_dim=H_node, edge_feat_dim=H_edge)
    logits = model(node_feats, edge_index, edge_feats, batch_idx, num_graphs=2)
    pom = model.extract_pom(node_feats, edge_index, edge_feats, batch_idx, num_graphs=2)
    print(f"Logits shape: {logits.shape}")   # (2, 138)
    print(f"POM shape:    {pom.shape}")      # (2, 256)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
