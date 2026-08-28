from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

from e3prep.models.utils import center_process_embeddings


class HGTClassifier(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels: int = 6,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.metadata = metadata
        self.dropout = dropout
        self.input_lin = nn.ModuleDict(
            {node_type: nn.Linear(in_channels, hidden_channels) for node_type in metadata[0]}
        )
        self.convs = nn.ModuleList(
            [HGTConv(hidden_channels, hidden_channels, metadata, heads=heads) for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, batch) -> torch.Tensor:
        x_dict = {
            node_type: F.relu(self.input_lin[node_type](x))
            for node_type, x in batch.x_dict.items()
            if node_type in self.input_lin
        }
        for conv in self.convs:
            updated = conv(x_dict, batch.edge_index_dict)
            x_dict = {
                node_type: F.dropout(F.relu(updated.get(node_type, x)), p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }
        centers = center_process_embeddings(batch, x_dict["process"])
        return self.classifier(centers)
