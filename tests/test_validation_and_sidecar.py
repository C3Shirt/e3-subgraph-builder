from __future__ import annotations

from pathlib import Path

import pandas as pd

from e3prep.export.shards import write_pyg_shards
from e3prep.io import write_parquet_records
from e3prep.schema.samples import SubgraphSample
from e3prep.validation.coverage import label_coverage_report
from e3prep.validation.leakage import leakage_report, overlap_total
from scripts.validate import event_leakage_failures, resolve_subgraph_dirs


def test_write_sidecar_and_label_coverage(tmp_path: Path) -> None:
    entities = pd.DataFrame(
        [
            {"uuid": "P1", "node_type": "PROCESS", "name": "bash", "path": None, "cmdline": "bash -c x", "ip": None, "port": None},
            {"uuid": "F1", "node_type": "FILE", "name": "/tmp/x", "path": "/tmp/x", "cmdline": None, "ip": None, "port": None},
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "uuid": "P1",
                "dataset": "cadets",
                "label": 1,
                "label_source": "test",
                "attack_id": None,
                "confidence": "entity_level",
                "start_time_ns": None,
                "end_time_ns": None,
            }
        ]
    )
    sample = SubgraphSample(
        sample_id="s1",
        dataset="cadets",
        split="train",
        center_uuid="P1",
        label=1,
        label_source="test",
        label_confidence="entity_level",
        label_strategy="center",
        t_start_ns=10,
        t_end_ns=10,
        context_start_ns=0,
        context_end_ns=20,
        nodes={"P1": "PROCESS", "F1": "FILE"},
        edges=[
            {
                "event_uuid": "E1",
                "event_type": "EVENT_WRITE",
                "timestamp_ns": 10,
                "actor_uuid": "P1",
                "actor_type": "PROCESS",
                "object_uuid": "F1",
                "object_type": "FILE",
                "object_path": "/tmp/x",
                "flow_src_uuid": "P1",
                "flow_src_type": "PROCESS",
                "flow_dst_uuid": "F1",
                "flow_dst_type": "FILE",
                "object_role": "predicateObject",
                "_hop": 1,
                "_direction": "forward",
                "source_file": "train.json",
            }
        ],
        positive_node_count=1,
    )

    store_dir = tmp_path / "store"
    write_parquet_records(entities.to_dict("records"), store_dir / "entities.parquet", "entities")
    write_parquet_records(sample.edges, store_dir / "events.parquet", "events")
    write_parquet_records(labels.to_dict("records"), store_dir / "labels.parquet", "labels")
    out = write_pyg_shards([sample], entities, tmp_path / "subgraphs" / "train", labels=labels, write_sidecar=True)

    assert out["samples"] == 1
    assert (tmp_path / "subgraphs" / "train" / "nodes.parquet").exists()
    assert (tmp_path / "subgraphs" / "train" / "edges.parquet").exists()

    coverage = label_coverage_report(store_dir, tmp_path / "subgraphs" / "train")
    assert coverage["labels_in_entities"] == 1
    assert coverage["labels_in_event_actors"] == 1
    assert coverage["labels_in_subgraph_centers"] == 1
    assert coverage["labels_in_subgraph_nodes"] == 1

    leakage = leakage_report(store_dir, tmp_path / "subgraphs" / "train")
    assert leakage["shared_node_uuids"] == {}


def test_leakage_report_flags_shared_event_edge_ids(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    rows = []
    for split in ("train", "test"):
        rows.append(
            {
                "event_edge_id": "EE_SHARED",
                "event_uuid": "E1",
                "dataset": "cadets",
                "host": "H",
                "split": split,
                "actor_uuid": f"P_{split}",
                "actor_type": "PROCESS",
                "object_uuid": "F1",
                "object_type": "FILE",
                "object_path": "/tmp/x",
                "event_type": "EVENT_WRITE",
                "timestamp_ns": 10,
                "sequence": 1,
                "flow_src_uuid": f"P_{split}",
                "flow_src_type": "PROCESS",
                "flow_dst_uuid": "F1",
                "flow_dst_type": "FILE",
                "object_role": "predicateObject",
                "source_file": f"{split}.json",
            }
        )
    write_parquet_records(rows, store_dir / "events.parquet", "events")

    leakage = leakage_report(store_dir)

    assert overlap_total(leakage["shared_event_edge_ids"]) == 1
    assert event_leakage_failures(leakage) == ["shared_event_edge_ids=1"]


def test_leakage_report_reads_train_val_test_sidecars(tmp_path: Path) -> None:
    store_dir = tmp_path / "store"
    write_parquet_records([], store_dir / "events.parquet", "events")
    subgraph_root = tmp_path / "subgraphs"
    for split, event_edge_id in [("train", "EE_SHARED"), ("val", "EE_SHARED"), ("test", "EE_TEST")]:
        split_dir = subgraph_root / split
        write_parquet_records(
            [
                {
                    "sample_id": f"s_{split}",
                    "dataset": "cadets",
                    "split": split,
                    "center_uuid": f"P_{split}",
                    "label": 0,
                    "label_source": None,
                    "label_confidence": None,
                    "label_strategy": "center",
                    "t_start_ns": 1,
                    "t_end_ns": 1,
                    "context_start_ns": 0,
                    "context_end_ns": 2,
                    "n_nodes": 2,
                    "n_edges": 1,
                    "positive_node_count": 0,
                }
            ],
            split_dir / "metadata.parquet",
            "sample_metadata",
        )
        write_parquet_records(
            [
                {
                    "sample_id": f"s_{split}",
                    "dataset": "cadets",
                    "split": split,
                    "event_edge_id": event_edge_id,
                    "event_uuid": f"E_{event_edge_id}",
                    "event_type": "EVENT_WRITE",
                    "timestamp_ns": 1,
                    "actor_uuid": f"P_{split}",
                    "actor_type": "PROCESS",
                    "object_uuid": "F1",
                    "object_type": "FILE",
                    "object_path": "/tmp/x",
                    "flow_src_uuid": f"P_{split}",
                    "flow_src_type": "PROCESS",
                    "flow_dst_uuid": "F1",
                    "flow_dst_type": "FILE",
                    "object_role": "predicateObject",
                    "hop": 1,
                    "direction": "forward",
                    "source_file": f"{split}.json",
                }
            ],
            split_dir / "edges.parquet",
            "sample_edges",
        )

    leakage = leakage_report(
        store_dir,
        {
            "train": subgraph_root / "train",
            "val": subgraph_root / "val",
            "test": subgraph_root / "test",
        },
    )

    assert overlap_total(leakage["shared_subgraph_event_edge_ids"]) == 1
    assert "shared_subgraph_event_edge_ids=1" in event_leakage_failures(leakage)


def test_validate_subgraph_root_requires_formal_splits(tmp_path: Path) -> None:
    subgraph_root = tmp_path / "subgraphs"
    for split in ("train", "val", "test"):
        write_parquet_records([], subgraph_root / split / "metadata.parquet", "sample_metadata")

    class Args:
        pass

    args = Args()
    args.subgraph_root = subgraph_root
    args.required_subgraph_split = None
    args.subgraph_dir = []

    subgraph_dirs = resolve_subgraph_dirs(args)

    assert set(subgraph_dirs) == {"train", "val", "test"}
