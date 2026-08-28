from __future__ import annotations

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from e3prep.models.dataset import metadata_from_graphs
from e3prep.models.edge_prediction import (
    edge_prediction_loss,
    graph_scores_from_edge_losses,
    make_edge_type_predictor,
)


def make_graph(label: int) -> HeteroData:
    data = HeteroData()
    data["process"].x = torch.tensor(
        [
            [1.0, 0.0, 2.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float,
    )
    data["file"].x = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float,
    )
    data["socket"].x = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float)
    data["process", "read", "file"].edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    data["process", "connect", "socket"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data.y = torch.tensor([label], dtype=torch.long)
    data.center_type = "process"
    data.center_index = 0
    data.sample_id = f"sample_{label}"
    return data


def test_edge_type_predictor_trains_on_edges_not_graph_labels() -> None:
    graphs = [make_graph(0), make_graph(1)]
    batch = next(iter(DataLoader(graphs, batch_size=2)))
    metadata = metadata_from_graphs(graphs)
    model = make_edge_type_predictor("gin", metadata, hidden_channels=16, num_layers=1)

    logits, target, graph_ids = model(batch)
    loss = edge_prediction_loss(logits, target)

    assert logits.shape == (6, 2)
    assert sorted([target.tolist().count(0), target.tolist().count(1)]) == [2, 4]
    assert graph_ids.tolist().count(0) == 3
    assert graph_ids.tolist().count(1) == 3
    assert batch.y.tolist() == [0, 1]
    assert loss.requires_grad


def test_graph_scores_from_edge_losses_are_per_graph_means() -> None:
    losses = torch.tensor([1.0, 3.0, 10.0, 14.0])
    graph_ids = torch.tensor([0, 0, 1, 1])

    scores, counts = graph_scores_from_edge_losses(losses, graph_ids, num_graphs=3)

    assert scores.tolist() == [2.0, 12.0, 0.0]
    assert counts.tolist() == [2, 2, 0]


def test_hgt_edge_type_predictor_forward() -> None:
    graphs = [make_graph(0), make_graph(1)]
    batch = next(iter(DataLoader(graphs, batch_size=2)))
    metadata = metadata_from_graphs(graphs)
    model = make_edge_type_predictor("hgt", metadata, hidden_channels=16, num_layers=1, heads=2)

    logits, target, graph_ids = model(batch)

    assert logits.shape == (6, 2)
    assert target.shape == (6,)
    assert graph_ids.shape == (6,)
