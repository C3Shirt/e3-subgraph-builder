from __future__ import annotations


REVERSE_INFORMATION_FLOW_KEYWORDS = (
    "READ",
    "RECV",
    "LOAD",
    "EXECUTE",
)


def normalize_event_type(event_type: str | None) -> str:
    if not event_type:
        return "EVENT_UNKNOWN"
    event_type = event_type.upper()
    return event_type if event_type.startswith("EVENT_") else f"EVENT_{event_type}"


def relation_name(event_type: str | None) -> str:
    return normalize_event_type(event_type).removeprefix("EVENT_").lower()


def is_reverse_information_flow(event_type: str | None) -> bool:
    normalized = normalize_event_type(event_type)
    return any(keyword in normalized for keyword in REVERSE_INFORMATION_FLOW_KEYWORDS)


def derive_information_flow(
    actor_uuid: str,
    actor_type: str,
    object_uuid: str,
    object_type: str,
    event_type: str | None,
) -> tuple[str, str, str, str]:
    if is_reverse_information_flow(event_type):
        return object_uuid, object_type, actor_uuid, actor_type
    return actor_uuid, actor_type, object_uuid, object_type

