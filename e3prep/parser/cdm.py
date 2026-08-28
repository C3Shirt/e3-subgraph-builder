from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from e3prep.graph.direction import derive_information_flow, normalize_event_type
from e3prep.graph.identity import stable_event_edge_id_from_values
from e3prep.io import iter_jsonl_records
from e3prep.parser.base import CdmParser
from e3prep.schema.entities import EntityRecord
from e3prep.schema.events import EventRecord


ZERO_UUID = "00000000-0000-0000-0000-000000000000"
ENTITY_RECORD_TYPES = {
    "Subject",
    "FileObject",
    "NetFlowObject",
    "MemoryObject",
    "SrcSinkObject",
    "UnnamedPipeObject",
    "Principal",
}


def unwrap_union(value):
    if isinstance(value, dict):
        for key in (
            "string",
            "int",
            "long",
            "boolean",
            "float",
            "double",
            "bytes",
        ):
            if key in value:
                return value[key]
        if len(value) == 1:
            inner_key, inner_value = next(iter(value.items()))
            if inner_key.endswith(".UUID"):
                return inner_value
    return value


def unwrap_datum(record: dict) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    datum = record.get("datum")
    if not isinstance(datum, dict) or len(datum) != 1:
        return None, None, None
    raw_type, payload = next(iter(datum.items()))
    if not isinstance(payload, dict):
        return None, None, raw_type
    return raw_type.rsplit(".", 1)[-1], payload, raw_type


def properties_map(payload: dict) -> dict:
    base_object = payload.get("baseObject") or {}
    properties = base_object.get("properties") or payload.get("properties") or {}
    properties = unwrap_union(properties) or {}
    if not isinstance(properties, dict):
        return {}
    mapping = properties.get("map") or {}
    return mapping if isinstance(mapping, dict) else {}


def get_host(payload: dict) -> Optional[str]:
    base_object = payload.get("baseObject") or {}
    return unwrap_union(payload.get("hostId")) or unwrap_union(base_object.get("hostId"))


def get_uuid(value) -> Optional[str]:
    value = unwrap_union(value)
    if isinstance(value, str) and value and value != ZERO_UUID:
        return value
    return None


def first_present(*values) -> Optional[str]:
    for value in values:
        value = unwrap_union(value)
        if value not in (None, ""):
            return str(value)
    return None


def canonical_node_type(record_type: str, payload: dict) -> tuple[str, Optional[str]]:
    raw_subtype = first_present(payload.get("type"))
    upper_subtype = (raw_subtype or "").upper()
    if record_type == "Subject":
        if upper_subtype == "SUBJECT_UNIT":
            return "UNIT", raw_subtype
        return "PROCESS", raw_subtype or "SUBJECT_PROCESS"
    if record_type == "FileObject":
        return "FILE", raw_subtype
    if record_type == "NetFlowObject":
        return "SOCKET", raw_subtype or "NetFlowObject"
    if record_type == "MemoryObject":
        return "MEMORY", raw_subtype or "MemoryObject"
    if record_type == "UnnamedPipeObject":
        return "PIPE", raw_subtype or "UnnamedPipeObject"
    if record_type == "Principal":
        return "PRINCIPAL", raw_subtype or "Principal"
    return record_type.upper(), raw_subtype


def parse_entity_record(
    dataset: str,
    source_file: Path,
    record_type: str,
    raw_type: str,
    payload: dict,
) -> Optional[EntityRecord]:
    uuid = get_uuid(payload.get("uuid"))
    if not uuid:
        return None
    node_type, raw_subtype = canonical_node_type(record_type, payload)
    if raw_subtype == "SUBJECT_UNIT":
        return None

    props = properties_map(payload)
    path = first_present(
        props.get("path"),
        payload.get("path"),
        payload.get("filename"),
        props.get("filename"),
    )
    cmdline = first_present(
        props.get("cmdLine"),
        props.get("cmdline"),
        props.get("commandLine"),
        payload.get("cmdLine"),
    )
    name = first_present(props.get("name"), payload.get("name"), path)
    local_address = first_present(payload.get("localAddress"), props.get("localAddress"))
    remote_address = first_present(payload.get("remoteAddress"), props.get("remoteAddress"))
    remote_port = unwrap_union(payload.get("remotePort")) or unwrap_union(props.get("remotePort"))
    local_port = unwrap_union(payload.get("localPort")) or unwrap_union(props.get("localPort"))
    port = remote_port if remote_port is not None else local_port

    try:
        port = int(port) if port is not None else None
    except (TypeError, ValueError):
        port = None

    if node_type == "SOCKET":
        if not name:
            name = remote_address or local_address
        ip = remote_address or local_address
    else:
        ip = None

    return EntityRecord(
        uuid=uuid,
        dataset=dataset,
        host=get_host(payload),
        node_type=node_type,
        name=name,
        path=path,
        cmdline=cmdline,
        ip=ip,
        port=port,
        raw_subtype=raw_subtype,
        raw_type=raw_type,
        source_file=os.fspath(source_file),
    )


def infer_entities_from_event(
    dataset: str,
    source_file: Path,
    payload: dict,
) -> list[EntityRecord]:
    inferred: list[EntityRecord] = []
    props = properties_map(payload)
    host = first_present(payload.get("hostId"))
    actor_uuid = get_uuid(payload.get("subject"))
    if actor_uuid:
        exec_name = first_present(props.get("exec"), props.get("name"), props.get("path"))
        inferred.append(
            EntityRecord(
                uuid=actor_uuid,
                dataset=dataset,
                host=host,
                node_type="PROCESS",
                name=exec_name,
                path=exec_name if exec_name and "/" in exec_name else None,
                cmdline=first_present(props.get("cmdLine"), props.get("cmdline"), props.get("commandLine")),
                raw_subtype="SUBJECT_PROCESS",
                raw_type="inferred_from_event",
                source_file=os.fspath(source_file),
            )
        )

    event_type = normalize_event_type(first_present(payload.get("type")))
    for role, object_value in (
        ("predicateObject", payload.get("predicateObject")),
        ("predicateObject2", payload.get("predicateObject2")),
    ):
        object_uuid = get_uuid(object_value)
        if not object_uuid:
            continue
        path = event_object_path(payload, role)
        object_type = "FILE" if path else "UNKNOWN"
        if object_type == "UNKNOWN" and any(keyword in event_type for keyword in ("CONNECT", "SEND", "RECV")):
            object_type = "SOCKET"
        inferred.append(
            EntityRecord(
                uuid=object_uuid,
                dataset=dataset,
                host=host,
                node_type=object_type,
                name=path,
                path=path if object_type == "FILE" else None,
                raw_subtype=f"inferred_{object_type.lower()}",
                raw_type="inferred_from_event",
                source_file=os.fspath(source_file),
            )
        )
    return inferred


def event_object_path(event_payload: dict, object_role: str) -> Optional[str]:
    if object_role == "predicateObject2":
        return first_present(event_payload.get("predicateObject2Path"))
    return first_present(event_payload.get("predicateObjectPath"))


class CdmJsonParser(CdmParser):
    def __init__(self, dataset: str):
        self.dataset = dataset

    def collect_entities(
        self,
        paths: Iterable[Path],
        progress_every: Optional[int] = None,
        infer_from_events: bool = True,
    ) -> dict[str, EntityRecord]:
        entities: dict[str, EntityRecord] = {}
        rows_seen = 0
        current_file: Path | None = None
        include_patterns = sorted(ENTITY_RECORD_TYPES)
        if infer_from_events:
            include_patterns.append("Event")
        for source_file, _line_no, record in iter_jsonl_records(list(paths), include_patterns=include_patterns):
            if progress_every and source_file != current_file:
                current_file = source_file
                print(f"[entity-pass] reading {source_file}", flush=True)
            rows_seen += 1
            if progress_every and rows_seen % progress_every == 0:
                print(f"[entity-pass] rows={rows_seen} entities={len(entities)}", flush=True)
            record_type, payload, raw_type = unwrap_datum(record)
            if infer_from_events and record_type == "Event" and payload is not None:
                for entity in infer_entities_from_event(self.dataset, source_file, payload):
                    if entity.uuid in entities:
                        entities[entity.uuid] = entities[entity.uuid].merge(entity)
                    else:
                        entities[entity.uuid] = entity
                continue
            if not record_type or record_type not in ENTITY_RECORD_TYPES or raw_type is None:
                continue
            entity = parse_entity_record(self.dataset, source_file, record_type, raw_type, payload)
            if entity is None:
                continue
            if entity.uuid in entities:
                entities[entity.uuid] = entities[entity.uuid].merge(entity)
            else:
                entities[entity.uuid] = entity
        return entities

    def parse_events(
        self,
        paths: Iterable[Path],
        entities: dict[str, EntityRecord],
        split: str,
        progress_every: Optional[int] = None,
    ) -> Iterable[EventRecord]:
        rows_seen = 0
        events_seen = 0
        current_file: Path | None = None
        for source_file, _line_no, record in iter_jsonl_records(list(paths), include_patterns=("Event",)):
            if progress_every and source_file != current_file:
                current_file = source_file
                print(f"[event-pass:{split}] reading {source_file}", flush=True)
            rows_seen += 1
            record_type, payload, _raw_type = unwrap_datum(record)
            if record_type != "Event" or payload is None:
                continue
            for event in self._parse_event(source_file, payload, entities, split):
                events_seen += 1
                if progress_every and events_seen % progress_every == 0:
                    print(f"[event-pass:{split}] rows={rows_seen} events={events_seen}", flush=True)
                yield event

    def _parse_event(
        self,
        source_file: Path,
        payload: dict,
        entities: dict[str, EntityRecord],
        split: str,
    ) -> Iterable[EventRecord]:
        actor_uuid = get_uuid(payload.get("subject"))
        event_uuid = get_uuid(payload.get("uuid"))
        if not actor_uuid or not event_uuid:
            return

        actor = entities.get(actor_uuid)
        actor_type = actor.node_type if actor else "UNKNOWN"
        event_type = normalize_event_type(first_present(payload.get("type")))
        timestamp = unwrap_union(payload.get("timestampNanos"))
        sequence = unwrap_union(payload.get("sequence"))
        host = first_present(payload.get("hostId")) or (actor.host if actor else None)

        objects = [
            ("predicateObject", payload.get("predicateObject")),
            ("predicateObject2", payload.get("predicateObject2")),
        ]
        for object_role, object_value in objects:
            object_uuid = get_uuid(object_value)
            if not object_uuid:
                continue
            obj = entities.get(object_uuid)
            object_type = obj.node_type if obj else "UNKNOWN"
            flow_src_uuid, flow_src_type, flow_dst_uuid, flow_dst_type = derive_information_flow(
                actor_uuid,
                actor_type,
                object_uuid,
                object_type,
                event_type,
            )
            event_edge_id = stable_event_edge_id_from_values(
                self.dataset,
                event_uuid,
                object_role,
                actor_uuid,
                object_uuid,
                event_type,
                timestamp,
                sequence,
            )
            yield EventRecord(
                event_edge_id=event_edge_id,
                event_uuid=event_uuid,
                dataset=self.dataset,
                host=host,
                split=split,
                actor_uuid=actor_uuid,
                actor_type=actor_type,
                object_uuid=object_uuid,
                object_type=object_type,
                object_path=event_object_path(payload, object_role),
                event_type=event_type,
                timestamp_ns=int(timestamp) if timestamp is not None else None,
                sequence=int(sequence) if sequence is not None else None,
                flow_src_uuid=flow_src_uuid,
                flow_src_type=flow_src_type,
                flow_dst_uuid=flow_dst_uuid,
                flow_dst_type=flow_dst_type,
                object_role=object_role,
                source_file=os.fspath(source_file),
            )
