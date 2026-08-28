from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


class GINClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, batch) -> torch.Tensor:
        graph = batch.to_homogeneous(node_attrs=["x"])
        center_mask = graph.x[:, 0] > 0.5
        x = F.relu(self.input_lin(graph.x))
        for conv in self.convs:
            x = conv(x, graph.edge_index)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        centers = x[center_mask]
        expected = int(batch.y.view(-1).numel())
        if centers.size(0) != expected:
            raise ValueError(f"Expected {expected} center nodes, found {centers.size(0)}")
        return self.classifier(centers)
