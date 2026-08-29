from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from e3prep.graph.index import IndexedEvent, ensure_event_edge_ids
from e3prep.io import clean_parquet_value, parquet_columns


SQLITE_INDEX_VERSION = 1
SQLITE_EVENT_COLUMNS = [
    "event_edge_id",
    "event_uuid",
    "dataset",
    "host",
    "split",
    "actor_uuid",
    "actor_type",
    "object_uuid",
    "object_type",
    "object_path",
    "event_type",
    "timestamp_ns",
    "sequence",
    "flow_src_uuid",
    "flow_src_type",
    "flow_dst_uuid",
    "flow_dst_type",
    "object_role",
    "source_file",
]


class SqliteTemporalGraphIndex:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row

    def incoming_edges(self, node_uuid: str, start_ns: int, end_ns: int) -> Iterable[IndexedEvent]:
        return self._query_edges(
            "SELECT * FROM events WHERE flow_dst_uuid = ? AND timestamp_ns BETWEEN ? AND ? "
            "ORDER BY timestamp_ns, COALESCE(sequence, 0), event_edge_id",
            (node_uuid, int(start_ns), int(end_ns)),
        )

    def outgoing_edges(self, node_uuid: str, start_ns: int, end_ns: int) -> Iterable[IndexedEvent]:
        return self._query_edges(
            "SELECT * FROM events WHERE flow_src_uuid = ? AND timestamp_ns BETWEEN ? AND ? "
            "ORDER BY timestamp_ns, COALESCE(sequence, 0), event_edge_id",
            (node_uuid, int(start_ns), int(end_ns)),
        )

    def incident_edges(self, node_uuid: str) -> list[IndexedEvent]:
        return list(
            self._query_edges(
                "SELECT * FROM events WHERE actor_uuid = ? OR object_uuid = ? "
                "ORDER BY timestamp_ns, COALESCE(sequence, 0), event_edge_id",
                (node_uuid, node_uuid),
            )
        )

    def _query_edges(self, query: str, params: tuple) -> Iterable[IndexedEvent]:
        for row in self.conn.execute(query, params):
            yield sqlite_row_to_event(row)

    def close(self) -> None:
        self.conn.close()


def sqlite_row_to_event(row: sqlite3.Row) -> IndexedEvent:
    def text(name: str, default: str = "UNKNOWN") -> str:
        value = row[name]
        return default if value is None else str(value)

    def nullable_text(name: str) -> str | None:
        value = row[name]
        return None if value is None else str(value)

    sequence = row["sequence"]
    return IndexedEvent(
        row_id=int(row["id"]),
        event_edge_id=text("event_edge_id"),
        event_uuid=text("event_uuid"),
        actor_uuid=text("actor_uuid"),
        actor_type=text("actor_type"),
        object_uuid=text("object_uuid"),
        object_type=text("object_type"),
        event_type=text("event_type"),
        timestamp_ns=int(row["timestamp_ns"]),
        sequence=None if sequence is None else int(sequence),
        flow_src_uuid=text("flow_src_uuid"),
        flow_src_type=text("flow_src_type"),
        flow_dst_uuid=text("flow_dst_uuid"),
        flow_dst_type=text("flow_dst_type"),
        split=text("split", "unknown"),
        object_path=nullable_text("object_path"),
        object_role=text("object_role", "predicateObject"),
        source_file=nullable_text("source_file"),
    )


def sqlite_cache_key(events_path: Path, split: str) -> dict[str, str]:
    stat = events_path.stat()
    return {
        "version": str(SQLITE_INDEX_VERSION),
        "events_path": str(events_path.resolve()),
        "events_size": str(stat.st_size),
        "events_mtime_ns": str(stat.st_mtime_ns),
        "split": split,
    }


def read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    except sqlite3.Error:
        return {}
    return {str(row[0]): str(row[1]) for row in rows}


def sqlite_index_is_current(path: Path, expected_key: dict[str, str]) -> bool:
    if not path.exists():
        return False
    conn = sqlite3.connect(path)
    try:
        meta = read_meta(conn)
        if meta != expected_key:
            return False
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return int(count) >= 0
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def initialize_sqlite_index(path: Path, expected_key: dict[str, str]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", sorted(expected_key.items()))
    conn.execute(
        """
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            event_edge_id TEXT NOT NULL,
            event_uuid TEXT NOT NULL,
            dataset TEXT,
            host TEXT,
            split TEXT NOT NULL,
            actor_uuid TEXT NOT NULL,
            actor_type TEXT,
            object_uuid TEXT NOT NULL,
            object_type TEXT,
            object_path TEXT,
            event_type TEXT NOT NULL,
            timestamp_ns INTEGER NOT NULL,
            sequence INTEGER,
            flow_src_uuid TEXT NOT NULL,
            flow_src_type TEXT,
            flow_dst_uuid TEXT NOT NULL,
            flow_dst_type TEXT,
            object_role TEXT,
            source_file TEXT
        )
        """
    )
    conn.commit()
    return conn


def create_sqlite_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX idx_events_flow_src_time ON events(flow_src_uuid, timestamp_ns)")
    conn.execute("CREATE INDEX idx_events_flow_dst_time ON events(flow_dst_uuid, timestamp_ns)")
    conn.execute("CREATE INDEX idx_events_actor_time ON events(actor_uuid, timestamp_ns)")
    conn.execute("CREATE INDEX idx_events_object_time ON events(object_uuid, timestamp_ns)")
    conn.commit()


def _prepare_batch(df: pd.DataFrame, split: str, available_columns: set[str]) -> pd.DataFrame:
    if "split" not in df.columns:
        df["split"] = split
    df = df[df["split"].fillna("").astype(str) == split].copy()
    if df.empty:
        return df
    for column in SQLITE_EVENT_COLUMNS:
        if column not in df.columns:
            df[column] = None
    if "event_edge_id" not in available_columns or df["event_edge_id"].isna().any():
        df = ensure_event_edge_ids(df)
    df = df.dropna(subset=["event_edge_id", "event_uuid", "actor_uuid", "object_uuid", "event_type", "timestamp_ns"])
    return df.loc[:, SQLITE_EVENT_COLUMNS]


def _batch_records(df: pd.DataFrame) -> list[tuple]:
    records = []
    for row in df.itertuples(index=False, name=None):
        records.append(tuple(clean_parquet_value(value) for value in row))
    return records


def build_sqlite_temporal_index(
    events_path: Path,
    split: str,
    index_path: Path,
    rebuild: bool = False,
    batch_size: int = 250_000,
) -> tuple[SqliteTemporalGraphIndex, str, int]:
    expected_key = sqlite_cache_key(events_path, split)
    if not rebuild and sqlite_index_is_current(index_path, expected_key):
        conn = sqlite3.connect(index_path)
        try:
            rows = int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        finally:
            conn.close()
        return SqliteTemporalGraphIndex(index_path), "loaded", rows

    available = set(parquet_columns(events_path))
    columns = [column for column in SQLITE_EVENT_COLUMNS if column in available]
    conn = initialize_sqlite_index(index_path, expected_key)
    inserted = 0
    placeholders = ", ".join("?" for _ in SQLITE_EVENT_COLUMNS)
    insert_sql = f"INSERT INTO events({', '.join(SQLITE_EVENT_COLUMNS)}) VALUES ({placeholders})"
    try:
        parquet_file = pq.ParquetFile(events_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            df = _prepare_batch(batch.to_pandas(), split, available)
            if df.empty:
                continue
            records = _batch_records(df)
            conn.executemany(insert_sql, records)
            inserted += len(records)
            conn.commit()
        create_sqlite_indexes(conn)
    finally:
        conn.close()
    return SqliteTemporalGraphIndex(index_path), "built", inserted
