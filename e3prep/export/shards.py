from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from e3prep.export.pyg import sample_to_heterodata
from e3prep.io import write_parquet_records
from e3prep.schema.samples import SubgraphSample


def entity_mapping(entities: pd.DataFrame) -> dict[str, dict]:
    return entities.set_index("uuid").to_dict("index")


def positive_label_set(labels: pd.DataFrame | None) -> set[str]:
    if labels is None or labels.empty:
        return set()
    rows = labels[labels["label"] == 1]
    return set(rows["uuid"].dropna().astype(str).str.upper().tolist())


def sample_node_rows(
    sample: SubgraphSample,
    entity_by_uuid: dict[str, dict],
    positive_uuids: set[str],
) -> Iterable[dict]:
    for uuid, node_type in sorted(sample.nodes.items()):
        entity = entity_by_uuid.get(uuid, {})
        port = entity.get("port")
        if pd.isna(port):
            port = None
        yield {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "split": sample.split,
            "uuid": uuid,
            "node_type": node_type,
            "is_center": int(uuid == sample.center_uuid),
            "is_labeled_positive": int(str(uuid).upper() in positive_uuids),
            "name": entity.get("name"),
            "path": entity.get("path"),
            "cmdline": entity.get("cmdline"),
            "ip": entity.get("ip"),
            "port": int(port) if port is not None else None,
        }


def sample_edge_rows(sample: SubgraphSample) -> Iterable[dict]:
    for edge in sample.edges:
        yield {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "split": sample.split,
            "event_edge_id": edge.get("event_edge_id"),
            "event_uuid": edge.get("event_uuid"),
            "event_type": edge.get("event_type"),
            "timestamp_ns": edge.get("timestamp_ns"),
            "actor_uuid": edge.get("actor_uuid"),
            "actor_type": edge.get("actor_type"),
            "object_uuid": edge.get("object_uuid"),
            "object_type": edge.get("object_type"),
            "object_path": edge.get("object_path"),
            "flow_src_uuid": edge.get("flow_src_uuid"),
            "flow_src_type": edge.get("flow_src_type"),
            "flow_dst_uuid": edge.get("flow_dst_uuid"),
            "flow_dst_type": edge.get("flow_dst_type"),
            "object_role": edge.get("object_role"),
            "hop": edge.get("_hop"),
            "direction": edge.get("_direction"),
            "source_file": edge.get("source_file"),
        }


def write_pyg_shards(
    samples: Iterable[SubgraphSample],
    entities: pd.DataFrame,
    out_dir: Path,
    samples_per_shard: int = 1000,
    labels: pd.DataFrame | None = None,
    write_sidecar: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_dir = out_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    entity_by_uuid = entity_mapping(entities)
    positive_uuids = positive_label_set(labels)

    shard: list = []
    metadata_rows: list[dict] = []
    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    shard_index = 0
    total_samples = 0

    def flush() -> None:
        nonlocal shard, shard_index
        if not shard:
            return
        path = graph_dir / f"shard_{shard_index:04d}.pt"
        torch.save(shard, path)
        shard = []
        shard_index += 1

    for sample in samples:
        shard.append(sample_to_heterodata(sample, entity_by_uuid))
        metadata_rows.append(sample.metadata())
        if write_sidecar:
            node_rows.extend(sample_node_rows(sample, entity_by_uuid, positive_uuids))
            edge_rows.extend(sample_edge_rows(sample))
        total_samples += 1
        if len(shard) >= samples_per_shard:
            flush()
    flush()
    write_parquet_records(metadata_rows, out_dir / "metadata.parquet", "sample_metadata")
    result = {
        "samples": total_samples,
        "shards": shard_index,
        "metadata": str(out_dir / "metadata.parquet"),
        "graph_dir": str(graph_dir),
    }
    if write_sidecar:
        write_parquet_records(node_rows, out_dir / "nodes.parquet", "sample_nodes")
        write_parquet_records(edge_rows, out_dir / "edges.parquet", "sample_edges")
        result["nodes"] = str(out_dir / "nodes.parquet")
        result["edges"] = str(out_dir / "edges.parquet")
    return result
