from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


EVENT_COLUMNS = [
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


@dataclass
class EventRecord:
    event_edge_id: str
    event_uuid: str
    dataset: str
    host: Optional[str]
    split: str
    actor_uuid: str
    actor_type: str
    object_uuid: str
    object_type: str
    object_path: Optional[str]
    event_type: str
    timestamp_ns: Optional[int]
    sequence: Optional[int]
    flow_src_uuid: str
    flow_src_type: str
    flow_dst_uuid: str
    flow_dst_type: str
    object_role: str
    source_file: Optional[str] = None

    def to_dict(self) -> dict:
        row = asdict(self)
        return {column: row.get(column) for column in EVENT_COLUMNS}
