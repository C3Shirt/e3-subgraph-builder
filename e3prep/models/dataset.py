from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


class GraphListDataset(Dataset):
    def __init__(self, graphs: Sequence):
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int):
        return self.graphs[index]

    @property
    def labels(self) -> list[int]:
        return [int(graph.y.view(-1)[0].item()) for graph in self.graphs]


def graph_shards(subgraph_dir: Path) -> list[Path]:
    graph_dir = subgraph_dir / "graphs"
    if not graph_dir.exists():
        raise FileNotFoundError(f"Missing graph shard directory: {graph_dir}")
    shards = sorted(graph_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt files found in {graph_dir}")
    return shards


def load_graphs_from_dir(
    subgraph_dir: Path,
    max_samples: int | None = None,
    labels: set[int] | None = None,
    require_positive_node_count: str = "any",
) -> list:
    graphs = []
    for shard in graph_shards(subgraph_dir):
        for graph in torch.load(shard, weights_only=False):
            label = int(graph.y.view(-1)[0].item())
            if labels is not None and label not in labels:
                continue
            positive_node_count = int(getattr(graph, "positive_node_count", -1))
            if require_positive_node_count == "zero" and positive_node_count not in (0, -1):
                continue
            if require_positive_node_count == "nonzero" and positive_node_count <= 0:
                continue
            graphs.append(graph)
            if max_samples is not None and len(graphs) >= max_samples:
                return graphs
    return graphs


def limit_per_class(graphs: Sequence, max_per_class: int | None, seed: int) -> list:
    if max_per_class is None:
        return list(graphs)
    by_label: dict[int, list] = {}
    for graph in graphs:
        label = int(graph.y.view(-1)[0].item())
        by_label.setdefault(label, []).append(graph)
    rng = random.Random(seed)
    selected = []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        selected.extend(items[:max_per_class])
    rng.shuffle(selected)
    return selected


def class_counts(graphs: Iterable) -> dict[int, int]:
    counts: dict[int, int] = {}
    for graph in graphs:
        label = int(graph.y.view(-1)[0].item())
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def split_graphs_stratified(
    graphs: Sequence,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list, list, list]:
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("val_fraction and test_fraction must be in [0, 1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    by_label: dict[int, list] = {}
    for graph in graphs:
        label = int(graph.y.view(-1)[0].item())
        by_label.setdefault(label, []).append(graph)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for _label, items in sorted(by_label.items()):
        items = list(items)
        rng.shuffle(items)
        n_total = len(items)
        n_test = int(round(n_total * test_fraction))
        n_val = int(round(n_total * val_fraction))
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])

    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def metadata_from_graphs(graphs: Sequence) -> tuple[list[str], list[tuple[str, str, str]]]:
    node_types: set[str] = set()
    edge_types: set[tuple[str, str, str]] = set()
    for graph in graphs:
        node_types.update(graph.node_types)
        edge_types.update(graph.edge_types)
    return sorted(node_types), sorted(edge_types)
