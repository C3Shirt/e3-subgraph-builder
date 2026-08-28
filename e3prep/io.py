from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PARQUET_SCHEMAS = {
    "entities": pa.schema(
        [
            ("uuid", pa.string()),
            ("dataset", pa.string()),
            ("host", pa.string()),
            ("node_type", pa.string()),
            ("name", pa.string()),
            ("path", pa.string()),
            ("cmdline", pa.string()),
            ("ip", pa.string()),
            ("port", pa.int64()),
            ("raw_subtype", pa.string()),
            ("raw_type", pa.string()),
            ("source_file", pa.string()),
        ]
    ),
    "events": pa.schema(
        [
            ("event_edge_id", pa.string()),
            ("event_uuid", pa.string()),
            ("dataset", pa.string()),
            ("host", pa.string()),
            ("split", pa.string()),
            ("actor_uuid", pa.string()),
            ("actor_type", pa.string()),
            ("object_uuid", pa.string()),
            ("object_type", pa.string()),
            ("object_path", pa.string()),
            ("event_type", pa.string()),
            ("timestamp_ns", pa.int64()),
            ("sequence", pa.int64()),
            ("flow_src_uuid", pa.string()),
            ("flow_src_type", pa.string()),
            ("flow_dst_uuid", pa.string()),
            ("flow_dst_type", pa.string()),
            ("object_role", pa.string()),
            ("source_file", pa.string()),
        ]
    ),
    "labels": pa.schema(
        [
            ("uuid", pa.string()),
            ("dataset", pa.string()),
            ("label", pa.int64()),
            ("label_source", pa.string()),
            ("attack_id", pa.string()),
            ("confidence", pa.string()),
            ("start_time_ns", pa.int64()),
            ("end_time_ns", pa.int64()),
        ]
    ),
    "sample_metadata": pa.schema(
        [
            ("sample_id", pa.string()),
            ("dataset", pa.string()),
            ("split", pa.string()),
            ("center_uuid", pa.string()),
            ("label", pa.int64()),
            ("label_source", pa.string()),
            ("label_confidence", pa.string()),
            ("label_strategy", pa.string()),
            ("t_start_ns", pa.int64()),
            ("t_end_ns", pa.int64()),
            ("context_start_ns", pa.int64()),
            ("context_end_ns", pa.int64()),
            ("n_nodes", pa.int64()),
            ("n_edges", pa.int64()),
            ("positive_node_count", pa.int64()),
        ]
    ),
    "sample_nodes": pa.schema(
        [
            ("sample_id", pa.string()),
            ("dataset", pa.string()),
            ("split", pa.string()),
            ("uuid", pa.string()),
            ("node_type", pa.string()),
            ("is_center", pa.int64()),
            ("is_labeled_positive", pa.int64()),
            ("name", pa.string()),
            ("path", pa.string()),
            ("cmdline", pa.string()),
            ("ip", pa.string()),
            ("port", pa.int64()),
        ]
    ),
    "sample_edges": pa.schema(
        [
            ("sample_id", pa.string()),
            ("dataset", pa.string()),
            ("split", pa.string()),
            ("event_edge_id", pa.string()),
            ("event_uuid", pa.string()),
            ("event_type", pa.string()),
            ("timestamp_ns", pa.int64()),
            ("actor_uuid", pa.string()),
            ("actor_type", pa.string()),
            ("object_uuid", pa.string()),
            ("object_type", pa.string()),
            ("object_path", pa.string()),
            ("flow_src_uuid", pa.string()),
            ("flow_src_type", pa.string()),
            ("flow_dst_uuid", pa.string()),
            ("flow_dst_type", pa.string()),
            ("object_role", pa.string()),
            ("hop", pa.int64()),
            ("direction", pa.string()),
            ("source_file", pa.string()),
        ]
    ),
}


def iter_jsonl_records(
    paths: Sequence[Path],
    include_patterns: Sequence[str] | None = None,
) -> Iterator[tuple[Path, int, dict]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                if include_patterns and not any(pattern in line for pattern in include_patterns):
                    continue
                try:
                    yield path, line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def write_parquet_records(
    rows: Iterable[dict],
    path: Path,
    schema_name: str,
    chunk_size: int = 100_000,
) -> int:
    schema = PARQUET_SCHEMAS[schema_name]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    count = 0
    buffer: list[dict] = []

    def flush() -> None:
        nonlocal writer, count, buffer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(path, schema)
        writer.write_table(table)
        count += len(buffer)
        buffer = []

    try:
        for row in rows:
            buffer.append({name: clean_parquet_value(row.get(name)) for name in schema.names})
            if len(buffer) >= chunk_size:
                flush()
        flush()
        if writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=schema), path)
    finally:
        if writer is not None:
            writer.close()
    return count


def clean_parquet_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value


def read_parquet(
    path: Path,
    columns: Sequence[str] | None = None,
    filters=None,
    allow_missing_columns: bool = False,
) -> pd.DataFrame:
    if not allow_missing_columns or columns is None:
        return pd.read_parquet(path, columns=columns, filters=filters)
    available = set(parquet_columns(path))
    present_columns = [column for column in columns if column in available]
    df = pd.read_parquet(path, columns=present_columns, filters=filters)
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df.loc[:, list(columns)]


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema.names


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
