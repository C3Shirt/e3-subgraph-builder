from __future__ import annotations

from itertools import combinations
from pathlib import Path
import tempfile
from typing import Mapping

import pandas as pd
import pyarrow.parquet as pq

from e3prep.graph.index import ensure_event_edge_ids
from e3prep.io import parquet_columns, read_parquet


COLUMNAR_OVERLAP_ROW_THRESHOLD = 1_000_000
OVERLAP_BUCKETS = 256


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


def parquet_num_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _append_bucket_rows(
    df: pd.DataFrame,
    key: str,
    split_col: str,
    bucket_dir: Path,
    num_buckets: int,
) -> None:
    if df.empty:
        return
    df = df[[split_col, key]].dropna(subset=[split_col, key]).copy()
    if df.empty:
        return
    df[split_col] = df[split_col].astype(str)
    df[key] = df[key].astype(str)
    buckets = pd.util.hash_pandas_object(df[key], index=False).mod(num_buckets)
    df["_bucket"] = buckets.astype("int64")
    for bucket, group in df.groupby("_bucket", sort=False):
        bucket_path = bucket_dir / f"bucket_{int(bucket):04d}.tsv"
        group[[split_col, key]].drop_duplicates().to_csv(
            bucket_path,
            sep="\t",
            header=False,
            index=False,
            mode="a",
            encoding="utf-8",
        )


def _bucket_pairwise_counts(bucket_dir: Path, key: str, split_col: str) -> dict:
    counts: dict[str, int] = {}
    for bucket_path in sorted(bucket_dir.glob("bucket_*.tsv")):
        df = pd.read_csv(
            bucket_path,
            sep="\t",
            names=[split_col, key],
            dtype={split_col: "string", key: "string"},
        ).drop_duplicates()
        if df.empty:
            continue
        pairs = df.merge(df, on=key, how="inner", suffixes=("_left", "_right"))
        pairs = pairs[pairs[f"{split_col}_left"] < pairs[f"{split_col}_right"]]
        if pairs.empty:
            continue
        pair_counts = (
            pairs.groupby([f"{split_col}_left", f"{split_col}_right"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        for _, row in pair_counts.iterrows():
            pair = f"{row[f'{split_col}_left']}/{row[f'{split_col}_right']}"
            counts[pair] = counts.get(pair, 0) + int(row["count"])
    return counts


def pairwise_overlap_parquet_sources(
    sources: Mapping[str, Path],
    key: str,
    split_col: str = "split",
    tmp_parent: Path | None = None,
    batch_size: int = 250_000,
    num_buckets: int = OVERLAP_BUCKETS,
) -> dict | None:
    if not sources:
        return {}
    for path in sources.values():
        if not path.exists() or key not in parquet_columns(path):
            return None

    with tempfile.TemporaryDirectory(prefix=".leakage_", dir=tmp_parent) as tmp_text:
        bucket_dir = Path(tmp_text)
        for default_split, path in sorted(sources.items()):
            available = set(parquet_columns(path))
            columns = [key]
            if split_col in available:
                columns.insert(0, split_col)
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
                df = batch.to_pandas()
                if split_col not in df.columns:
                    df[split_col] = default_split
                else:
                    df[split_col] = df[split_col].fillna(default_split)
                _append_bucket_rows(df, key, split_col, bucket_dir, num_buckets)
        return _bucket_pairwise_counts(bucket_dir, key, split_col)


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
    report: dict = {
        "event_identity_policy": {
            "formal_gate": "event_edge_id",
            "source_event_uuid": "diagnostic_only; CDM event_uuid is not assumed globally unique",
        }
    }
    if events_path.exists():
        use_columnar_events = parquet_num_rows(events_path) > COLUMNAR_OVERLAP_ROW_THRESHOLD
        if use_columnar_events:
            report["shared_event_ids"] = {"skipped": "large_event_store; formal_gate_uses_event_edge_id"}
            report["shared_event_edge_ids"] = pairwise_overlap_parquet_sources(
                {"events": events_path},
                "event_edge_id",
                tmp_parent=store_dir,
            )
            report["shared_event_edge_keys"] = {"skipped": "columnar_validation_uses_event_edge_id"}
            events = pd.DataFrame()
        else:
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
    if "shared_event_ids" not in report:
        report["shared_event_ids"] = pairwise_overlap(events, "event_uuid")

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

    subgraph_node_sources = {
        split: path / "nodes.parquet"
        for split, path in subgraph_dirs.items()
        if (path / "nodes.parquet").exists()
    }
    if subgraph_node_sources:
        total_subgraph_nodes = sum(parquet_num_rows(path) for path in subgraph_node_sources.values())
        if total_subgraph_nodes > COLUMNAR_OVERLAP_ROW_THRESHOLD:
            report["shared_node_uuids"] = pairwise_overlap_parquet_sources(
                subgraph_node_sources,
                "uuid",
                tmp_parent=store_dir,
            )
            report["shared_node_uuid_jaccard"] = {"skipped": "large_subgraph_sidecars"}
        else:
            nodes = read_subgraph_parquet_frames(subgraph_dirs, "nodes.parquet", ["uuid", "split"])
            if not nodes.empty:
                report["shared_node_uuids"] = pairwise_overlap(nodes, "uuid")
                report["shared_node_uuid_jaccard"] = pairwise_jaccard(nodes, "uuid")

    subgraph_edge_sources = {
        split: path / "edges.parquet"
        for split, path in subgraph_dirs.items()
        if (path / "edges.parquet").exists()
    }
    if subgraph_edge_sources:
        total_subgraph_edges = sum(parquet_num_rows(path) for path in subgraph_edge_sources.values())
        if total_subgraph_edges > COLUMNAR_OVERLAP_ROW_THRESHOLD:
            report["shared_subgraph_event_edge_ids"] = pairwise_overlap_parquet_sources(
                subgraph_edge_sources,
                "event_edge_id",
                tmp_parent=store_dir,
            )
            report["shared_subgraph_event_ids"] = {"skipped": "large_subgraph_sidecars; formal_gate_uses_event_edge_id"}
        else:
            subgraph_edges = read_subgraph_parquet_frames(
                subgraph_dirs,
                "edges.parquet",
                ["event_edge_id", "event_uuid", "split"],
            )
            if not subgraph_edges.empty:
                report["shared_subgraph_event_edge_ids"] = pairwise_overlap(subgraph_edges, "event_edge_id")
                report["shared_subgraph_event_ids"] = pairwise_overlap(subgraph_edges, "event_uuid")
                report["shared_subgraph_event_ids_note"] = (
                    "diagnostic_only; use shared_subgraph_event_edge_ids for formal leakage gating"
                )
    return report


def overlap_total(overlap: dict | None) -> int | None:
    if overlap is None:
        return None
    total = 0
    for value in overlap.values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            return None
    return total
