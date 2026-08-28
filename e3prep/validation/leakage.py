from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from e3prep.io import parquet_columns, read_parquet


def pairwise_overlap(df: pd.DataFrame, key: str, split_col: str = "split") -> dict:
    if df.empty or key not in df.columns or split_col not in df.columns:
        return {}
    by_split = {
        split: set(group[key].dropna().astype(str).tolist())
        for split, group in df.groupby(split_col)
    }
    return {
        f"{left}/{right}": len(by_split[left].intersection(by_split[right]))
        for left, right in combinations(sorted(by_split), 2)
    }


def pairwise_jaccard(df: pd.DataFrame, key: str, split_col: str = "split") -> dict:
    if df.empty or key not in df.columns or split_col not in df.columns:
        return {}
    by_split = {
        split: set(group[key].dropna().astype(str).tolist())
        for split, group in df.groupby(split_col)
    }
    result = {}
    for left, right in combinations(sorted(by_split), 2):
        union = by_split[left].union(by_split[right])
        result[f"{left}/{right}"] = 0.0 if not union else len(by_split[left].intersection(by_split[right])) / len(union)
    return result


def leakage_report(store_dir: Path, subgraph_dir: Path | None = None) -> dict:
    events_path = store_dir / "events.parquet"
    events = pd.DataFrame()
    if events_path.exists():
        columns = ["event_uuid", "object_role", "actor_uuid", "object_uuid", "split"]
        if "event_edge_id" in parquet_columns(events_path):
            columns.insert(0, "event_edge_id")
        events = read_parquet(events_path, columns=columns)
    report = {"shared_event_ids": pairwise_overlap(events, "event_uuid")}

    if not events.empty:
        if "event_edge_id" in events.columns:
            report["shared_event_edge_ids"] = pairwise_overlap(events, "event_edge_id")
        else:
            report["shared_event_edge_ids"] = None
        edge_keys = events.assign(
            edge_key=(
                events["event_uuid"].astype(str)
                + "|"
                + events["object_role"].astype(str)
                + "|"
                + events["actor_uuid"].astype(str)
                + "|"
                + events["object_uuid"].astype(str)
            )
        )
        report["shared_event_edge_keys"] = pairwise_overlap(edge_keys, "edge_key")

    metadata_path = subgraph_dir / "metadata.parquet" if subgraph_dir else None
    if metadata_path and metadata_path.exists():
        metadata = read_parquet(metadata_path, columns=["sample_id", "center_uuid", "split"])
        report["shared_sample_ids"] = pairwise_overlap(metadata, "sample_id")
        report["shared_center_uuids"] = pairwise_overlap(metadata, "center_uuid")
    nodes_path = subgraph_dir / "nodes.parquet" if subgraph_dir else None
    if nodes_path and nodes_path.exists():
        nodes = read_parquet(nodes_path, columns=["uuid", "split"])
        report["shared_node_uuids"] = pairwise_overlap(nodes, "uuid")
        report["shared_node_uuid_jaccard"] = pairwise_jaccard(nodes, "uuid")
    return report
