from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from e3prep.io import write_parquet_records
from scripts.build_balanced_dataset import main as build_balanced_main


def make_graph(sample_id: str, label: int) -> HeteroData:
    graph = HeteroData()
    graph["process"].x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    graph.y = torch.tensor([label], dtype=torch.long)
    graph.center_type = "process"
    graph.center_index = 0
    graph.sample_id = sample_id
    return graph


def write_source(source_dir: Path) -> None:
    graph_dir = source_dir / "graphs"
    graph_dir.mkdir(parents=True)
    torch.save(
        [
            make_graph("p1", 1),
            make_graph("p2", 1),
            make_graph("n1", 0),
            make_graph("n2", 0),
        ],
        graph_dir / "shard_0000.pt",
    )
    rows = [
        {
            "sample_id": "p1",
            "dataset": "cadets",
            "split": "train",
            "center_uuid": "P1",
            "label": 1,
            "label_source": "test",
            "label_confidence": "subgraph_contains_labeled_node",
            "label_strategy": "subgraph_any_positive",
            "t_start_ns": 1,
            "t_end_ns": 1,
            "context_start_ns": 0,
            "context_end_ns": 2,
            "n_nodes": 2,
            "n_edges": 1,
            "positive_node_count": 1,
        },
        {
            "sample_id": "p2",
            "dataset": "cadets",
            "split": "train",
            "center_uuid": "P2",
            "label": 1,
            "label_source": "test",
            "label_confidence": "subgraph_contains_labeled_node",
            "label_strategy": "subgraph_any_positive",
            "t_start_ns": 1,
            "t_end_ns": 1,
            "context_start_ns": 0,
            "context_end_ns": 2,
            "n_nodes": 2,
            "n_edges": 1,
            "positive_node_count": 0,
        },
        {
            "sample_id": "n1",
            "dataset": "cadets",
            "split": "train",
            "center_uuid": "N1",
            "label": 0,
            "label_source": None,
            "label_confidence": None,
            "label_strategy": "subgraph_any_positive",
            "t_start_ns": 1,
            "t_end_ns": 1,
            "context_start_ns": 0,
            "context_end_ns": 2,
            "n_nodes": 2,
            "n_edges": 1,
            "positive_node_count": 0,
        },
        {
            "sample_id": "n2",
            "dataset": "cadets",
            "split": "train",
            "center_uuid": "N2",
            "label": 0,
            "label_source": None,
            "label_confidence": None,
            "label_strategy": "subgraph_any_positive",
            "t_start_ns": 1,
            "t_end_ns": 1,
            "context_start_ns": 0,
            "context_end_ns": 2,
            "n_nodes": 2,
            "n_edges": 1,
            "positive_node_count": 1,
        },
    ]
    write_parquet_records(rows, source_dir / "metadata.parquet", "sample_metadata")


def test_build_balanced_dataset_filters_by_positive_node_count(tmp_path: Path, monkeypatch) -> None:
    source_dir = tmp_path / "source"
    out_dir = tmp_path / "balanced"
    write_source(source_dir)
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_balanced_dataset.py",
            "--source-dir",
            str(source_dir),
            "--out-dir",
            str(out_dir),
            "--samples-per-class",
            "2",
            "--samples-per-shard",
            "2",
        ],
    )

    build_balanced_main()

    metadata = pd.read_parquet(out_dir / "metadata.parquet")
    assert len(metadata) == 2
    assert metadata["label"].value_counts().sort_index().to_dict() == {0: 1, 1: 1}
    assert set(metadata["sample_id"]) == {"p1", "n1"}
