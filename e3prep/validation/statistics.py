from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from e3prep.io import read_parquet


def describe_series(series: pd.Series) -> dict:
    if series.empty:
        return {"min": 0, "median": 0, "p95": 0, "max": 0}
    return {
        "min": int(series.min()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "max": int(series.max()),
    }


def _subgraph_dirs_mapping(subgraph_dirs: Path | Mapping[str, Path] | None) -> dict[str, Path]:
    if subgraph_dirs is None:
        return {}
    if isinstance(subgraph_dirs, Path):
        return {subgraph_dirs.name: subgraph_dirs}
    return {str(split): Path(path) for split, path in subgraph_dirs.items()}


def _read_metadata_frames(subgraph_dirs: Mapping[str, Path]) -> pd.DataFrame:
    frames = []
    columns = ["sample_id", "split", "label", "n_nodes", "n_edges"]
    for split, subgraph_dir in sorted(subgraph_dirs.items()):
        metadata_path = subgraph_dir / "metadata.parquet"
        if not metadata_path.exists():
            continue
        frame = read_parquet(metadata_path, columns=columns, allow_missing_columns=True)
        frame["split"] = frame["split"].fillna(split)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _subgraph_stats(metadata: pd.DataFrame) -> dict:
    return {
        "subgraphs_total": int(len(metadata)),
        "positive_subgraphs": int((metadata["label"] == 1).sum()),
        "negative_subgraphs": int((metadata["label"] == 0).sum()),
        "nodes_per_subgraph": describe_series(metadata["n_nodes"]),
        "edges_per_subgraph": describe_series(metadata["n_edges"]),
    }


def build_dataset_report(store_dir: Path, subgraph_dir: Path | Mapping[str, Path] | None = None) -> dict:
    entities_path = store_dir / "entities.parquet"
    events_path = store_dir / "events.parquet"
    labels_path = store_dir / "labels.parquet"

    report: dict = {}
    entities = read_parquet(entities_path, columns=["node_type"]) if entities_path.exists() else pd.DataFrame()
    events = (
        read_parquet(events_path, columns=["split", "event_type", "actor_type", "object_type", "timestamp_ns"])
        if events_path.exists()
        else pd.DataFrame()
    )
    labels = read_parquet(labels_path, columns=["label"]) if labels_path.exists() else pd.DataFrame()

    report["entities_total"] = int(len(entities))
    report["entities_by_type"] = (
        entities["node_type"].value_counts().sort_index().astype(int).to_dict() if not entities.empty else {}
    )
    report["events_total"] = int(len(events))
    report["events_by_split"] = events["split"].value_counts().sort_index().astype(int).to_dict() if not events.empty else {}
    report["events_by_type"] = (
        events["event_type"].value_counts().sort_index().astype(int).to_dict() if not events.empty else {}
    )
    if not events.empty:
        report["missing_actor_type"] = int((events["actor_type"] == "UNKNOWN").sum())
        report["missing_object_type"] = int((events["object_type"] == "UNKNOWN").sum())
        report["timestamp_range_ns"] = {
            "min": int(events["timestamp_ns"].dropna().min()) if events["timestamp_ns"].notna().any() else None,
            "max": int(events["timestamp_ns"].dropna().max()) if events["timestamp_ns"].notna().any() else None,
        }
    else:
        report["missing_actor_type"] = 0
        report["missing_object_type"] = 0
        report["timestamp_range_ns"] = {"min": None, "max": None}

    report["malicious_entity_count"] = int((labels["label"] == 1).sum()) if not labels.empty else 0

    subgraph_dirs = _subgraph_dirs_mapping(subgraph_dir)
    if subgraph_dirs:
        report["subgraph_dirs"] = {split: str(path) for split, path in sorted(subgraph_dirs.items())}
    metadata = _read_metadata_frames(subgraph_dirs)
    if not metadata.empty:
        report.update(_subgraph_stats(metadata))
        report["subgraphs_by_split"] = metadata["split"].value_counts().sort_index().astype(int).to_dict()
        report["subgraph_stats_by_split"] = {
            str(split): _subgraph_stats(group)
            for split, group in metadata.groupby("split")
        }
    return report
