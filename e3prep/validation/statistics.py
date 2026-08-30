from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

import pandas as pd
import pyarrow.parquet as pq

from e3prep.io import parquet_columns, read_parquet


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


def _event_store_stats(events_path: Path, batch_size: int = 250_000) -> dict:
    if not events_path.exists():
        return {
            "events_total": 0,
            "events_by_split": {},
            "events_by_type": {},
            "missing_actor_type": 0,
            "missing_object_type": 0,
            "timestamp_range_ns": {"min": None, "max": None},
        }

    available = set(parquet_columns(events_path))
    columns = [
        column
        for column in ("split", "event_type", "actor_type", "object_type", "timestamp_ns")
        if column in available
    ]
    total = 0
    split_counts: Counter[str] = Counter()
    event_type_counts: Counter[str] = Counter()
    missing_actor_type = 0
    missing_object_type = 0
    timestamp_min: int | None = None
    timestamp_max: int | None = None

    parquet_file = pq.ParquetFile(events_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        events = batch.to_pandas()
        total += len(events)
        if events.empty:
            continue

        if "split" in events.columns:
            split_counts.update(events["split"].fillna("unknown").astype(str).tolist())
        if "event_type" in events.columns:
            event_type_counts.update(events["event_type"].fillna("UNKNOWN").astype(str).tolist())
        if "actor_type" in events.columns:
            actor_type = events["actor_type"].fillna("UNKNOWN").astype(str).str.upper()
            missing_actor_type += int((actor_type == "UNKNOWN").sum())
        if "object_type" in events.columns:
            object_type = events["object_type"].fillna("UNKNOWN").astype(str).str.upper()
            missing_object_type += int((object_type == "UNKNOWN").sum())
        if "timestamp_ns" in events.columns and events["timestamp_ns"].notna().any():
            timestamps = events["timestamp_ns"].dropna().astype("int64")
            batch_min = int(timestamps.min())
            batch_max = int(timestamps.max())
            timestamp_min = batch_min if timestamp_min is None else min(timestamp_min, batch_min)
            timestamp_max = batch_max if timestamp_max is None else max(timestamp_max, batch_max)

    return {
        "events_total": int(total),
        "events_by_split": dict(sorted((key, int(value)) for key, value in split_counts.items())),
        "events_by_type": dict(sorted((key, int(value)) for key, value in event_type_counts.items())),
        "missing_actor_type": int(missing_actor_type),
        "missing_object_type": int(missing_object_type),
        "timestamp_range_ns": {
            "min": timestamp_min,
            "max": timestamp_max,
        },
    }


def build_dataset_report(store_dir: Path, subgraph_dir: Path | Mapping[str, Path] | None = None) -> dict:
    entities_path = store_dir / "entities.parquet"
    events_path = store_dir / "events.parquet"
    labels_path = store_dir / "labels.parquet"

    report: dict = {}
    entities = read_parquet(entities_path, columns=["node_type"]) if entities_path.exists() else pd.DataFrame()
    labels = read_parquet(labels_path, columns=["label"]) if labels_path.exists() else pd.DataFrame()

    report["entities_total"] = int(len(entities))
    report["entities_by_type"] = (
        entities["node_type"].value_counts().sort_index().astype(int).to_dict() if not entities.empty else {}
    )
    report.update(_event_store_stats(events_path))

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
