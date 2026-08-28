from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, HGTConv

from e3prep.models.utils import center_process_embeddings


EdgeType = tuple[str, str, str]


def normalize_node_features(x: torch.Tensor) -> torch.Tensor:
    if x.size(-1) < 3:
        return x
    x = x.clone()
    x[:, 1:3] = torch.log1p(x[:, 1:3].clamp_min(0))
    return x


class HGTNodeEncoder(nn.Module):
    def __init__(
        self,
        metadata: tuple[list[str], list[EdgeType]],
        in_channels: int = 6,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        self.input_lin = nn.ModuleDict(
            {node_type: nn.Linear(in_channels, hidden_channels) for node_type in metadata[0]}
        )
        self.convs = nn.ModuleList(
            [HGTConv(hidden_channels, hidden_channels, metadata, heads=heads) for _ in range(num_layers)]
        )

    def forward(self, batch) -> dict[str, torch.Tensor]:
        x_dict = {
            node_type: F.relu(self.input_lin[node_type](normalize_node_features(x)))
            for node_type, x in batch.x_dict.items()
            if node_type in self.input_lin
        }
        for conv in self.convs:
            updated = conv(x_dict, batch.edge_index_dict)
            x_dict = {
                node_type: F.dropout(F.relu(updated.get(node_type, x)), p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }
        return x_dict


class GINNodeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
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

    def forward(self, batch) -> dict[str, torch.Tensor]:
        graph = batch.to_homogeneous(node_attrs=["x"])
        x = F.relu(self.input_lin(normalize_node_features(graph.x)))
        for conv in self.convs:
            x = conv(x, graph.edge_index)
            x = F.dropout(F.relu(x), p=self.dropout, training=self.training)

        x_dict: dict[str, torch.Tensor] = {}
        for type_id, node_type in enumerate(batch.node_types):
            x_dict[node_type] = x[graph.node_type == type_id]
        return x_dict


class EdgeTypePredictor(nn.Module):
    """Self-supervised relation predictor.

    The training target is the observed edge relation key, not the attack label
    stored in ``batch.y``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        edge_types: Sequence[EdgeType],
        hidden_channels: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = encoder
        self.edge_types = list(edge_types)
        self.relation_to_id = {edge_type: idx for idx, edge_type in enumerate(self.edge_types)}
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 4, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, len(self.edge_types)),
        )

    def encode_nodes(self, batch) -> dict[str, torch.Tensor]:
        return self.encoder(batch)

    def encode_centers(self, batch) -> torch.Tensor:
        x_dict = self.encode_nodes(batch)
        return center_process_embeddings(batch, x_dict["process"])

    def forward(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_dict = self.encode_nodes(batch)
        return edge_type_logits(batch, x_dict, self.relation_to_id, self.edge_mlp)


def make_edge_type_predictor(
    model_name: str,
    metadata: tuple[list[str], list[EdgeType]],
    in_channels: int = 6,
    hidden_channels: int = 64,
    num_layers: int = 2,
    heads: int = 2,
    dropout: float = 0.2,
) -> EdgeTypePredictor:
    if not metadata[1]:
        raise ValueError("Cannot train edge-type prediction without edge types")
    if model_name == "gin":
        encoder = GINNodeEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif model_name == "hgt":
        if hidden_channels % heads != 0:
            raise ValueError("--hidden-channels must be divisible by --heads for HGT")
        encoder = HGTNodeEncoder(
            metadata=metadata,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return EdgeTypePredictor(
        encoder=encoder,
        edge_types=metadata[1],
        hidden_channels=hidden_channels,
        dropout=dropout,
    )


def edge_type_logits(
    batch,
    x_dict: dict[str, torch.Tensor],
    relation_to_id: dict[EdgeType, int],
    edge_mlp: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_parts = []
    target_parts = []
    graph_id_parts = []

    for edge_type in sorted(batch.edge_index_dict.keys()):
        if edge_type not in relation_to_id:
            continue
        edge_index = batch.edge_index_dict[edge_type]
        if edge_index.numel() == 0:
            continue
        src_type, _relation, dst_type = edge_type
        src_z = x_dict[src_type][edge_index[0]]
        dst_z = x_dict[dst_type][edge_index[1]]
        edge_features = torch.cat([src_z, dst_z, torch.abs(src_z - dst_z), src_z * dst_z], dim=-1)
        logits_parts.append(edge_mlp(edge_features))
        target_parts.append(
            torch.full(
                (edge_index.size(1),),
                relation_to_id[edge_type],
                dtype=torch.long,
                device=edge_index.device,
            )
        )
        src_store = batch[src_type]
        if hasattr(src_store, "batch"):
            graph_id_parts.append(src_store.batch[edge_index[0]])
        else:
            graph_id_parts.append(torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device))

    if not logits_parts:
        device = next(edge_mlp.parameters()).device
        out_features = next(module for module in reversed(edge_mlp) if isinstance(module, nn.Linear)).out_features
        return (
            torch.empty((0, out_features), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    return torch.cat(logits_parts, dim=0), torch.cat(target_parts, dim=0), torch.cat(graph_id_parts, dim=0)


def graph_scores_from_edge_losses(
    losses: torch.Tensor,
    graph_ids: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros(num_graphs, dtype=losses.dtype, device=losses.device)
    counts = torch.zeros(num_graphs, dtype=losses.dtype, device=losses.device)
    if losses.numel():
        sums.scatter_add_(0, graph_ids, losses)
        counts.scatter_add_(0, graph_ids, torch.ones_like(losses))
    return sums / counts.clamp_min(1), counts.long()


def edge_prediction_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum()
    return F.cross_entropy(logits, target)
