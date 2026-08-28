from __future__ import annotations

from pathlib import Path

import pandas as pd

from e3prep.io import read_parquet


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


def label_coverage_report(store_dir: Path, subgraph_dir: Path | None = None) -> dict:
    labels_path = store_dir / "labels.parquet"
    entities_path = store_dir / "entities.parquet"
    events_path = store_dir / "events.parquet"
    metadata_path = subgraph_dir / "metadata.parquet" if subgraph_dir else None
    nodes_path = subgraph_dir / "nodes.parquet" if subgraph_dir else None

    labels = read_parquet(labels_path, columns=["uuid", "label"]) if labels_path.exists() else pd.DataFrame()
    entities = read_parquet(entities_path, columns=["uuid", "node_type"]) if entities_path.exists() else pd.DataFrame()
    events = (
        read_parquet(events_path, columns=["split", "actor_uuid", "object_uuid"])
        if events_path.exists()
        else pd.DataFrame()
    )
    metadata = (
        read_parquet(metadata_path, columns=["center_uuid", "label"])
        if metadata_path and metadata_path.exists()
        else pd.DataFrame()
    )
    nodes = read_parquet(nodes_path, columns=["uuid"]) if nodes_path and nodes_path.exists() else pd.DataFrame()

    positive_labels = set()
    if not labels.empty:
        positive_rows = labels[labels["label"] == 1]
        positive_labels = _uuid_set(positive_rows, "uuid")

    entity_uuids = _uuid_set(entities, "uuid")
    event_actor_uuids = _uuid_set(events, "actor_uuid")
    event_object_uuids = _uuid_set(events, "object_uuid")
    event_endpoint_uuids = event_actor_uuids | event_object_uuids
    center_uuids = _uuid_set(metadata, "center_uuid")
    subgraph_node_uuids = _uuid_set(nodes, "uuid")

    in_entities = positive_labels & entity_uuids
    in_events = positive_labels & event_endpoint_uuids
    in_centers = positive_labels & center_uuids
    in_subgraph_nodes = positive_labels & subgraph_node_uuids if subgraph_node_uuids else set()

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
        "labels_in_subgraph_nodes": int(len(in_subgraph_nodes)) if subgraph_node_uuids else None,
        "labeled_entity_types": _by_type_for_labels(in_entities, entities),
        "sample_missing_entity_labels": _sample_values(positive_labels - entity_uuids),
        "sample_missing_event_labels": _sample_values(positive_labels - event_endpoint_uuids),
    }

    if not metadata.empty:
        report["positive_subgraphs"] = int((metadata["label"] == 1).sum())
        report["negative_subgraphs"] = int((metadata["label"] == 0).sum())
    if not events.empty and "split" in events.columns:
        per_split = {}
        for split, group in events.groupby("split"):
            split_endpoints = _uuid_set(group, "actor_uuid") | _uuid_set(group, "object_uuid")
            per_split[str(split)] = int(len(positive_labels & split_endpoints))
        report["labels_in_event_endpoints_by_split"] = per_split
    return report
