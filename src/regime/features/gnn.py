"""
Lightweight Graph Neural Network (GNN) Feature Engineering for Regime Detection.
Models cross-index / sector correlation structure as dynamic graphs and generates graph embeddings.
Obeys strict zero look-ahead bias: at time t, only correlations up to t are observed.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LightweightGNN(nn.Module):
    """
    Lightweight 2-layer Graph Convolutional Network (GCN) for market correlation graphs.
    Computes graph representation without external torch_geometric dependencies.
    """
    def __init__(self, in_features: int = 4, hidden_dim: int = 8, out_dim: int = 4, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.W0 = nn.Linear(in_features, hidden_dim, bias=False)
        self.W1 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
            A_hat = D^{-1/2} (A + I) D^{-1/2}
            H1 = ReLU(A_hat X W0)
            H2 = A_hat H1 W1
            graph_embed = mean(H2, dim=0)
        """
        N = A.shape[0]
        # Add self-loops
        A_tilde = A + torch.eye(N, dtype=A.dtype, device=A.device)
        # Degree normalization
        deg = torch.sum(A_tilde, dim=1)
        deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-6), -0.5)
        D_norm = torch.diag(deg_inv_sqrt)
        A_hat = torch.mm(torch.mm(D_norm, A_tilde), D_norm)

        H1 = F.relu(torch.mm(A_hat, self.W0(X)))
        H2 = torch.mm(A_hat, self.W1(H1))

        graph_embed = torch.mean(H2, dim=0)
        return H2, graph_embed


def compute_rolling_gnn_features(
    asset_returns_df: pd.DataFrame,
    window: int = 63,
    correlation_threshold: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    r"""
    Compute causal rolling correlation graph embeddings using Lightweight GNN.

    For each rolling causal window of length $W$ up to date $t$:
    1. Form correlation matrix $C_t \in \mathbb{R}^{K \times K}$ between assets.
    2. Threshold correlation matrix to form adjacency $A_t = |C_t| \cdot \mathbb{I}(|C_t| > \theta)$.
    3. Construct node features $X_t \in \mathbb{R}^{K \times 4}$ (mean, vol, skew, Sharpe within window).
    4. Pass through Graph Convolutional layers:
       - `gnn_embed_l2_norm`: magnitude of graph embedding (inter-connected market stress).
       - `gnn_graph_energy`: spectral energy $\sum \lambda_i^2$ of adjacency matrix.
       - `gnn_spectral_gap`: leading eigenvalue gap $\lambda_1 - \lambda_2$ (market co-movement coherence).
       - `gnn_mean_centrality`: average degree centrality across nodes.

    Strict Zero Look-Ahead Discipline:
        Observation at index $i$ strictly uses rows $[i - W + 1 : i + 1]$.
    """
    T, K = asset_returns_df.shape
    dates = asset_returns_df.index
    arr = asset_returns_df.values

    cols = [
        "gnn_embed_l2_norm",
        "gnn_graph_energy",
        "gnn_spectral_gap",
        "gnn_mean_centrality",
    ]
    records = []

    model = LightweightGNN(in_features=4, hidden_dim=8, out_dim=4, seed=seed)
    model.eval()

    for i in range(T):
        if i < window - 1:
            records.append({c: np.nan for c in cols})
            continue

        w_data = arr[i - window + 1 : i + 1]

        # 1. Rolling Pearson correlation
        corr = np.corrcoef(w_data, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        # 2. Adjacency with thresholding
        adj = np.where(np.abs(corr) >= correlation_threshold, np.abs(corr), 0.0)
        np.fill_diagonal(adj, 0.0)

        # Spectral features
        eigvals = np.linalg.eigvalsh(adj)
        eigvals_sorted = np.sort(eigvals)[::-1]
        spectral_gap = float(eigvals_sorted[0] - eigvals_sorted[1]) if len(eigvals_sorted) > 1 else 0.0
        graph_energy = float(np.sum(eigvals ** 2))
        mean_centrality = float(np.mean(np.sum(adj, axis=1)))

        # 3. Node features: [mean, std, skew, sharpe]
        mu = np.mean(w_data, axis=0)
        sd = np.std(w_data, axis=0) + 1e-8
        sk = np.mean(((w_data - mu) / sd) ** 3, axis=0)
        sh = mu / sd
        node_feats = np.column_stack([mu, sd, sk, sh])

        # 4. GNN Forward
        with torch.no_grad():
            x_t = torch.tensor(node_feats, dtype=torch.float32)
            a_t = torch.tensor(adj, dtype=torch.float32)
            _, g_embed = model(x_t, a_t)
            embed_norm = float(torch.norm(g_embed).item())

        records.append({
            "gnn_embed_l2_norm": embed_norm,
            "gnn_graph_energy": graph_energy,
            "gnn_spectral_gap": spectral_gap,
            "gnn_mean_centrality": mean_centrality,
        })

    return pd.DataFrame(records, index=dates)
