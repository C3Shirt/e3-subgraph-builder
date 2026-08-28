from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any


EVENT_EDGE_ID_FIELDS = [
    "dataset",
    "event_uuid",
    "object_role",
    "actor_uuid",
    "object_uuid",
    "event_type",
    "timestamp_ns",
    "sequence",
]

UUID_FIELDS = {"event_uuid", "actor_uuid", "object_uuid"}


def _normalize_part(field: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value)
    if text.lower() in {"nan", "nat", "none"}:
        return ""
    text = text.strip()
    if field in UUID_FIELDS:
        text = text.upper()
    return text


def stable_event_edge_id(row: Mapping[str, Any]) -> str:
    payload = "\x1f".join(_normalize_part(field, row.get(field)) for field in EVENT_EDGE_ID_FIELDS)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:24]


def stable_event_edge_id_from_values(
    dataset: Any,
    event_uuid: Any,
    object_role: Any,
    actor_uuid: Any,
    object_uuid: Any,
    event_type: Any,
    timestamp_ns: Any,
    sequence: Any,
) -> str:
    return stable_event_edge_id(
        {
            "dataset": dataset,
            "event_uuid": event_uuid,
            "object_role": object_role,
            "actor_uuid": actor_uuid,
            "object_uuid": object_uuid,
            "event_type": event_type,
            "timestamp_ns": timestamp_ns,
            "sequence": sequence,
        }
    )
