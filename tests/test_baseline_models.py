from __future__ import annotations

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from e3prep.models.dataset import metadata_from_graphs, split_graphs_stratified
from e3prep.models.gin import GINClassifier
from e3prep.models.hgt import HGTClassifier


def make_graph(label: int, with_pipe: bool = False) -> HeteroData:
    data = HeteroData()
    data["process"].x = torch.tensor(
        [
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    data["file"].x = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    data["process", "write", "file"].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    if with_pipe:
        data["pipe"].x = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
        data["process", "create_object", "pipe"].edge_index = torch.tensor([[1], [0]], dtype=torch.long)
    data.y = torch.tensor([label], dtype=torch.long)
    data.center_type = "process"
    data.center_index = 0
    data.sample_id = f"s{label}"
    return data


def test_baseline_models_forward_on_heterodata_batch() -> None:
    graphs = [make_graph(0), make_graph(1, with_pipe=True)]
    batch = next(iter(DataLoader(graphs, batch_size=2)))
    metadata = metadata_from_graphs(graphs)

    hgt = HGTClassifier(metadata=metadata, hidden_channels=16, num_layers=1, heads=2)
    gin = GINClassifier(hidden_channels=16, num_layers=1)

    assert hgt(batch).shape == (2, 2)
    assert gin(batch).shape == (2, 2)


def test_stratified_split_preserves_classes() -> None:
    graphs = [make_graph(label) for label in [0, 0, 0, 0, 1, 1, 1, 1]]

    train, val, test = split_graphs_stratified(graphs, val_fraction=0.25, test_fraction=0.25, seed=1)

    assert len(train) == 4
    assert len(val) == 2
    assert len(test) == 2
    assert {int(graph.y.item()) for graph in train} == {0, 1}
    assert {int(graph.y.item()) for graph in val} == {0, 1}
    assert {int(graph.y.item()) for graph in test} == {0, 1}
