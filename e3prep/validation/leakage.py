from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Mapping

import pandas as pd

from e3prep.graph.index import ensure_event_edge_ids
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


def subgraph_dirs_mapping(subgraph_dirs: Path | Mapping[str, Path] | None) -> dict[str, Path]:
    if subgraph_dirs is None:
        return {}
    if isinstance(subgraph_dirs, Path):
        return {subgraph_dirs.name: subgraph_dirs}
    return {str(split): Path(path) for split, path in subgraph_dirs.items()}


def read_subgraph_parquet_frames(subgraph_dirs: Mapping[str, Path], filename: str, columns: list[str]) -> pd.DataFrame:
    frames = []
    for split, subgraph_dir in sorted(subgraph_dirs.items()):
        path = subgraph_dir / filename
        if not path.exists():
            continue
        df = read_parquet(path, columns=columns, allow_missing_columns=True)
        if "split" in columns:
            df["split"] = df["split"].fillna(split)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def leakage_report(store_dir: Path, subgraph_dir: Path | Mapping[str, Path] | None = None) -> dict:
    events_path = store_dir / "events.parquet"
    events = pd.DataFrame()
    if events_path.exists():
        columns = [
            "event_edge_id",
            "dataset",
            "event_uuid",
            "object_role",
            "actor_uuid",
            "object_uuid",
            "event_type",
            "timestamp_ns",
            "sequence",
            "split",
        ]
        available = set(parquet_columns(events_path))
        events = read_parquet(events_path, columns=columns, allow_missing_columns=True)
        if "event_edge_id" not in available or events["event_edge_id"].isna().any():
            events = ensure_event_edge_ids(events)
    report = {"shared_event_ids": pairwise_overlap(events, "event_uuid")}

    if not events.empty:
        report["shared_event_edge_ids"] = pairwise_overlap(events, "event_edge_id")
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

    subgraph_dirs = subgraph_dirs_mapping(subgraph_dir)
    if subgraph_dirs:
        report["subgraph_dirs"] = {split: str(path) for split, path in sorted(subgraph_dirs.items())}

    metadata = read_subgraph_parquet_frames(subgraph_dirs, "metadata.parquet", ["sample_id", "center_uuid", "split"])
    if not metadata.empty:
        report["shared_sample_ids"] = pairwise_overlap(metadata, "sample_id")
        report["shared_center_uuids"] = pairwise_overlap(metadata, "center_uuid")

    nodes = read_subgraph_parquet_frames(subgraph_dirs, "nodes.parquet", ["uuid", "split"])
    if not nodes.empty:
        report["shared_node_uuids"] = pairwise_overlap(nodes, "uuid")
        report["shared_node_uuid_jaccard"] = pairwise_jaccard(nodes, "uuid")

    subgraph_edges = read_subgraph_parquet_frames(subgraph_dirs, "edges.parquet", ["event_edge_id", "event_uuid", "split"])
    if not subgraph_edges.empty:
        report["shared_subgraph_event_edge_ids"] = pairwise_overlap(subgraph_edges, "event_edge_id")
        report["shared_subgraph_event_ids"] = pairwise_overlap(subgraph_edges, "event_uuid")
    return report


def overlap_total(overlap: dict | None) -> int | None:
    if overlap is None:
        return None
    return int(sum(int(value) for value in overlap.values()))
