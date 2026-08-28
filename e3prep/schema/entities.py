from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


ENTITY_COLUMNS = [
    "uuid",
    "dataset",
    "host",
    "node_type",
    "name",
    "path",
    "cmdline",
    "ip",
    "port",
    "raw_subtype",
    "raw_type",
    "source_file",
]


@dataclass
class EntityRecord:
    uuid: str
    dataset: str
    host: Optional[str] = None
    node_type: str = "UNKNOWN"
    name: Optional[str] = None
    path: Optional[str] = None
    cmdline: Optional[str] = None
    ip: Optional[str] = None
    port: Optional[int] = None
    raw_subtype: Optional[str] = None
    raw_type: Optional[str] = None
    source_file: Optional[str] = None

    def to_dict(self) -> dict:
        row = asdict(self)
        return {column: row.get(column) for column in ENTITY_COLUMNS}

    def merge(self, other: "EntityRecord") -> "EntityRecord":
        """Prefer non-empty metadata while keeping the first observed source."""
        current = self.to_dict()
        incoming = other.to_dict()
        for key, value in incoming.items():
            if key == "source_file":
                continue
            if current.get(key) in (None, "", "UNKNOWN") and value not in (None, ""):
                current[key] = value
        return EntityRecord(**current)

