from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Iterable

import pandas as pd

from e3prep.graph.identity import stable_event_edge_id, stable_event_edge_id_from_values


INDEX_CACHE_VERSION = 1


@dataclass(frozen=True)
class IndexedEvent:
    row_id: int
    event_edge_id: str
    event_uuid: str
    actor_uuid: str
    actor_type: str
    object_uuid: str
    object_type: str
    event_type: str
    timestamp_ns: int
    sequence: int | None
    flow_src_uuid: str
    flow_src_type: str
    flow_dst_uuid: str
    flow_dst_type: str
    split: str
    object_path: str | None
    object_role: str
    source_file: str | None

    @classmethod
    def from_series(cls, row_id: int, row: pd.Series) -> "IndexedEvent":
        event_edge_id = row.get("event_edge_id")
        if event_edge_id is None or pd.isna(event_edge_id):
            event_edge_id = stable_event_edge_id(row)
        return cls(
            row_id=row_id,
            event_edge_id=str(event_edge_id),
            event_uuid=str(row["event_uuid"]),
            actor_uuid=str(row["actor_uuid"]),
            actor_type=str(row.get("actor_type", "UNKNOWN")),
            object_uuid=str(row["object_uuid"]),
            object_type=str(row.get("object_type", "UNKNOWN")),
            event_type=str(row["event_type"]),
            timestamp_ns=int(row["timestamp_ns"]),
            sequence=None if pd.isna(row.get("sequence")) else int(row["sequence"]),
            flow_src_uuid=str(row["flow_src_uuid"]),
            flow_src_type=str(row.get("flow_src_type", "UNKNOWN")),
            flow_dst_uuid=str(row["flow_dst_uuid"]),
            flow_dst_type=str(row.get("flow_dst_type", "UNKNOWN")),
            split=str(row.get("split", "unknown")),
            object_path=None if pd.isna(row.get("object_path")) else str(row.get("object_path")),
            object_role=str(row.get("object_role", "predicateObject")),
            source_file=None if pd.isna(row.get("source_file")) else str(row.get("source_file")),
        )

    @classmethod
    def from_tuple(cls, row_id: int, row) -> "IndexedEvent":
        def value(name: str, default=None):
            return getattr(row, name, default)

        def nullable_str(name: str) -> str | None:
            raw = value(name)
            return None if raw is None or pd.isna(raw) else str(raw)

        def required_str(name: str, default: str = "UNKNOWN") -> str:
            raw = value(name, default)
            return default if raw is None or pd.isna(raw) else str(raw)

        sequence = value("sequence")
        event_edge_id = value("event_edge_id")
        if event_edge_id is None or pd.isna(event_edge_id):
            event_edge_id = stable_event_edge_id_from_values(
                value("dataset"),
                value("event_uuid"),
                value("object_role"),
                value("actor_uuid"),
                value("object_uuid"),
                value("event_type"),
                value("timestamp_ns"),
                sequence,
            )
        return cls(
            row_id=row_id,
            event_edge_id=str(event_edge_id),
            event_uuid=required_str("event_uuid"),
            actor_uuid=required_str("actor_uuid"),
            actor_type=required_str("actor_type"),
            object_uuid=required_str("object_uuid"),
            object_type=required_str("object_type"),
            event_type=required_str("event_type"),
            timestamp_ns=int(value("timestamp_ns")),
            sequence=None if sequence is None or pd.isna(sequence) else int(sequence),
            flow_src_uuid=required_str("flow_src_uuid"),
            flow_src_type=required_str("flow_src_type"),
            flow_dst_uuid=required_str("flow_dst_uuid"),
            flow_dst_type=required_str("flow_dst_type"),
            split=required_str("split", "unknown"),
            object_path=nullable_str("object_path"),
            object_role=required_str("object_role", "predicateObject"),
            source_file=nullable_str("source_file"),
        )

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "event_edge_id": self.event_edge_id,
            "event_uuid": self.event_uuid,
            "actor_uuid": self.actor_uuid,
            "actor_type": self.actor_type,
            "object_uuid": self.object_uuid,
            "object_type": self.object_type,
            "event_type": self.event_type,
            "timestamp_ns": self.timestamp_ns,
            "sequence": self.sequence,
            "flow_src_uuid": self.flow_src_uuid,
            "flow_src_type": self.flow_src_type,
            "flow_dst_uuid": self.flow_dst_uuid,
            "flow_dst_type": self.flow_dst_type,
            "split": self.split,
            "object_path": self.object_path,
            "object_role": self.object_role,
            "source_file": self.source_file,
        }


class TemporalGraphIndex:
    def __init__(self, events: pd.DataFrame):
        if "timestamp_ns" not in events.columns:
            raise ValueError("events DataFrame must include timestamp_ns")
        clean_events = events.dropna(subset=["timestamp_ns"]).copy()
        clean_events = clean_events.sort_values(["timestamp_ns", "sequence"], na_position="last")

        self.events: list[IndexedEvent] = []
        self.outgoing: dict[str, list[IndexedEvent]] = defaultdict(list)
        self.incoming: dict[str, list[IndexedEvent]] = defaultdict(list)
        self.incident: dict[str, list[IndexedEvent]] = defaultdict(list)

        clean_events = ensure_event_edge_ids(clean_events)
        for row_id, row in enumerate(clean_events.itertuples(index=False)):
            event = IndexedEvent.from_tuple(row_id, row)
            self.events.append(event)
            self.outgoing[event.flow_src_uuid].append(event)
            self.incoming[event.flow_dst_uuid].append(event)
            self.incident[event.actor_uuid].append(event)
            self.incident[event.object_uuid].append(event)

    def events_between(self, events: Iterable[IndexedEvent], start_ns: int, end_ns: int) -> Iterable[IndexedEvent]:
        for event in events:
            if start_ns <= event.timestamp_ns <= end_ns:
                yield event

    def incoming_edges(self, node_uuid: str, start_ns: int, end_ns: int) -> Iterable[IndexedEvent]:
        return self.events_between(self.incoming.get(node_uuid, []), start_ns, end_ns)

    def outgoing_edges(self, node_uuid: str, start_ns: int, end_ns: int) -> Iterable[IndexedEvent]:
        return self.events_between(self.outgoing.get(node_uuid, []), start_ns, end_ns)

    def incident_edges(self, node_uuid: str) -> list[IndexedEvent]:
        return self.incident.get(node_uuid, [])


def ensure_event_edge_ids(events: pd.DataFrame) -> pd.DataFrame:
    if "event_edge_id" in events.columns and events["event_edge_id"].notna().all():
        return events
    result = events.copy()
    if "event_edge_id" not in result.columns:
        result["event_edge_id"] = None
    missing = result["event_edge_id"].isna()
    if not missing.any():
        return result
    required = [
        "dataset",
        "event_uuid",
        "object_role",
        "actor_uuid",
        "object_uuid",
        "event_type",
        "timestamp_ns",
        "sequence",
    ]
    subset = result.loc[missing, required]
    result.loc[missing, "event_edge_id"] = [
        stable_event_edge_id_from_values(
            row.dataset,
            row.event_uuid,
            row.object_role,
            row.actor_uuid,
            row.object_uuid,
            row.event_type,
            row.timestamp_ns,
            row.sequence,
        )
        for row in subset.itertuples(index=False)
    ]
    return result


def build_or_load_temporal_index(
    events: pd.DataFrame,
    cache_path: Path | None = None,
    cache_key: dict | None = None,
    rebuild: bool = False,
) -> tuple[TemporalGraphIndex, str]:
    expected_key = cache_key or {}
    if cache_path and cache_path.exists() and not rebuild:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if (
            isinstance(payload, dict)
            and payload.get("version") == INDEX_CACHE_VERSION
            and payload.get("cache_key") == expected_key
            and isinstance(payload.get("index"), TemporalGraphIndex)
        ):
            return payload["index"], "loaded"

    index = TemporalGraphIndex(events)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(
                {
                    "version": INDEX_CACHE_VERSION,
                    "cache_key": expected_key,
                    "index": index,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    return index, "built"
