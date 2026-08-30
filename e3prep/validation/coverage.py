from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow.parquet as pq

from e3prep.io import parquet_columns, read_parquet
from e3prep.validation.leakage import read_subgraph_parquet_frames, subgraph_dirs_mapping


def _uuid_key_series(series: pd.Series) -> pd.Series:
    return series.dropna().astype(str).str.upper()


def _uuid_set(df: pd.DataFrame, column: str) -> set[str]:
    if df.empty or column not in df.columns:
        return set()
    return set(_uuid_key_series(df[column]).tolist())


def _sample_values(values: set[str], limit: int = 20) -> list[str]:
    return sorted(values)[:limit]


def _by_type_for_labels(labels: set[str], entities: pd.DataFrame) -> dict[str, int]:
    if entities.empty:
        return {}
    rows = entities.assign(_uuid_key=_uuid_key_series(entities["uuid"]))
    rows = rows[rows["_uuid_key"].isin(labels)]
    if rows.empty:
        return {}
    return rows["node_type"].fillna("UNKNOWN").value_counts().sort_index().astype(int).to_dict()


def _event_label_hits(events_path: Path, positive_labels: set[str], batch_size: int = 250_000) -> dict:
    if not events_path.exists() or not positive_labels:
        return {
            "actor_hits": set(),
            "object_hits": set(),
            "endpoint_hits": set(),
            "endpoint_hits_by_split": {},
        }
    available = set(parquet_columns(events_path))
    columns = [column for column in ("split", "actor_uuid", "object_uuid") if column in available]
    if "actor_uuid" not in columns and "object_uuid" not in columns:
        return {
            "actor_hits": set(),
            "object_hits": set(),
            "endpoint_hits": set(),
            "endpoint_hits_by_split": {},
        }

    actor_hits: set[str] = set()
    object_hits: set[str] = set()
    endpoint_hits_by_split: dict[str, set[str]] = {}
    parquet_file = pq.ParquetFile(events_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        events = batch.to_pandas()
        if "actor_uuid" in events.columns:
            actor_keys = set(_uuid_key_series(events["actor_uuid"]))
            actor_hits.update(actor_keys & positive_labels)
        else:
            actor_keys = set()
        if "object_uuid" in events.columns:
            object_keys = set(_uuid_key_series(events["object_uuid"]))
            object_hits.update(object_keys & positive_labels)
        else:
            object_keys = set()
        endpoint_hits = (actor_keys | object_keys) & positive_labels
        if endpoint_hits and "split" in events.columns:
            for split, group in events.groupby(events["split"].fillna("unknown").astype(str)):
                split_actor = set(_uuid_key_series(group["actor_uuid"])) if "actor_uuid" in group.columns else set()
                split_object = set(_uuid_key_series(group["object_uuid"])) if "object_uuid" in group.columns else set()
                split_hits = (split_actor | split_object) & positive_labels
                if split_hits:
                    endpoint_hits_by_split.setdefault(str(split), set()).update(split_hits)
    return {
        "actor_hits": actor_hits,
        "object_hits": object_hits,
        "endpoint_hits": actor_hits | object_hits,
        "endpoint_hits_by_split": endpoint_hits_by_split,
    }


def _subgraph_node_label_hits(
    subgraph_dirs: Mapping[str, Path],
    positive_labels: set[str],
    batch_size: int = 250_000,
) -> tuple[set[str], dict[str, set[str]], bool]:
    if not subgraph_dirs or not positive_labels:
        return set(), {}, False

    all_hits: set[str] = set()
    hits_by_split: dict[str, set[str]] = {}
    nodes_found = False
    for default_split, subgraph_dir in sorted(subgraph_dirs.items()):
        nodes_path = subgraph_dir / "nodes.parquet"
        if not nodes_path.exists() or "uuid" not in parquet_columns(nodes_path):
            continue
        nodes_found = True
        available = set(parquet_columns(nodes_path))
        columns = ["uuid"]
        if "split" in available:
            columns.insert(0, "split")
        parquet_file = pq.ParquetFile(nodes_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            nodes = batch.to_pandas()
            if nodes.empty:
                continue
            uuid_keys = _uuid_key_series(nodes["uuid"])
            hits = set(uuid_keys.tolist()) & positive_labels
            if not hits:
                continue
            all_hits.update(hits)
            if "split" not in nodes.columns:
                hits_by_split.setdefault(str(default_split), set()).update(hits)
                continue
            split_values = nodes["split"].fillna(default_split).astype(str)
            for split, group in nodes.groupby(split_values):
                split_hits = set(_uuid_key_series(group["uuid"]).tolist()) & positive_labels
                if split_hits:
                    hits_by_split.setdefault(str(split), set()).update(split_hits)
    return all_hits, hits_by_split, nodes_found


def label_coverage_report(store_dir: Path, subgraph_dir: Path | Mapping[str, Path] | None = None) -> dict:
    labels_path = store_dir / "labels.parquet"
    entities_path = store_dir / "entities.parquet"
    events_path = store_dir / "events.parquet"

    labels = read_parquet(labels_path, columns=["uuid", "label"]) if labels_path.exists() else pd.DataFrame()
    entities = read_parquet(entities_path, columns=["uuid", "node_type"]) if entities_path.exists() else pd.DataFrame()
    subgraph_dirs = subgraph_dirs_mapping(subgraph_dir)
    metadata = read_subgraph_parquet_frames(subgraph_dirs, "metadata.parquet", ["center_uuid", "label", "split"])

    positive_labels = set()
    if not labels.empty:
        positive_rows = labels[labels["label"] == 1]
        positive_labels = _uuid_set(positive_rows, "uuid")

    entity_uuids = _uuid_set(entities, "uuid")
    event_hits = _event_label_hits(events_path, positive_labels)
    event_actor_uuids = event_hits["actor_hits"]
    event_object_uuids = event_hits["object_hits"]
    event_endpoint_uuids = event_hits["endpoint_hits"]
    center_uuids = _uuid_set(metadata, "center_uuid")
    subgraph_node_uuids, subgraph_node_uuids_by_split, nodes_found = _subgraph_node_label_hits(
        subgraph_dirs,
        positive_labels,
    )

    in_entities = positive_labels & entity_uuids
    in_events = positive_labels & event_endpoint_uuids
    in_centers = positive_labels & center_uuids
    in_subgraph_nodes = positive_labels & subgraph_node_uuids if nodes_found else set()

    report = {
        "labels_path": str(labels_path) if labels_path.exists() else None,
        "labels_total": int(len(labels)),
        "positive_label_uuids": int(len(positive_labels)),
        "labels_in_entities": int(len(in_entities)),
        "labels_not_in_entities": int(len(positive_labels - entity_uuids)),
        "labels_in_event_actors": int(len(positive_labels & event_actor_uuids)),
        "labels_in_event_objects": int(len(positive_labels & event_object_uuids)),
        "labels_in_event_endpoints": int(len(in_events)),
        "labels_not_in_event_endpoints": int(len(positive_labels - event_endpoint_uuids)),
        "labels_in_subgraph_centers": int(len(in_centers)),
        "labels_in_subgraph_nodes": int(len(in_subgraph_nodes)) if nodes_found else None,
        "labeled_entity_types": _by_type_for_labels(in_entities, entities),
        "sample_missing_entity_labels": _sample_values(positive_labels - entity_uuids),
        "sample_missing_event_labels": _sample_values(positive_labels - event_endpoint_uuids),
    }

    if not metadata.empty:
        report["positive_subgraphs"] = int((metadata["label"] == 1).sum())
        report["negative_subgraphs"] = int((metadata["label"] == 0).sum())
        report["positive_subgraphs_by_split"] = (
            metadata[metadata["label"] == 1]["split"].value_counts().sort_index().astype(int).to_dict()
        )
        report["negative_subgraphs_by_split"] = (
            metadata[metadata["label"] == 0]["split"].value_counts().sort_index().astype(int).to_dict()
        )
        report["labels_in_subgraph_centers_by_split"] = {
            str(split): int(len(positive_labels & _uuid_set(group, "center_uuid")))
            for split, group in metadata.groupby("split")
        }
    if nodes_found:
        report["labels_in_subgraph_nodes_by_split"] = {
            split: int(len(values))
            for split, values in sorted(subgraph_node_uuids_by_split.items())
        }
    if event_hits["endpoint_hits_by_split"]:
        report["labels_in_event_endpoints_by_split"] = {
            split: int(len(values))
            for split, values in sorted(event_hits["endpoint_hits_by_split"].items())
        }
    return report
