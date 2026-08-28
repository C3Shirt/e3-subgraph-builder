from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


LABEL_COLUMNS = [
    "uuid",
    "dataset",
    "label",
    "label_source",
    "attack_id",
    "confidence",
    "start_time_ns",
    "end_time_ns",
]


@dataclass
class LabelRecord:
    uuid: str
    dataset: str
    label: int = 1
    label_source: str = "threatrace"
    attack_id: Optional[str] = None
    confidence: str = "entity_level"
    start_time_ns: Optional[int] = None
    end_time_ns: Optional[int] = None

    def to_dict(self) -> dict:
        row = asdict(self)
        return {column: row.get(column) for column in LABEL_COLUMNS}

