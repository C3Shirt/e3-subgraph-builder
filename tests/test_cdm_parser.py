from __future__ import annotations

import json
from pathlib import Path

from e3prep.graph.direction import derive_information_flow, relation_name
from e3prep.parser.cadets import discover_json_chunks
from e3prep.parser.cdm import CdmJsonParser


CDM = "com.bbn.tc.schema.avro.cdm18"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def datum(record_type: str, payload: dict) -> dict:
    return {"datum": {f"{CDM}.{record_type}": payload}, "CDMVersion": "18", "source": "test"}


def test_parse_entities_and_flow_direction(tmp_path: Path) -> None:
    source = tmp_path / "cadets.json"
    rows = [
        datum(
            "Subject",
            {
                "uuid": "P1",
                "type": "SUBJECT_PROCESS",
                "baseObject": {"hostId": "H", "properties": {"map": {"name": "bash", "cmdLine": "bash -c x"}}},
            },
        ),
        datum(
            "FileObject",
            {
                "uuid": "F1",
                "baseObject": {"hostId": "H", "properties": {"map": {"path": "/tmp/x"}}},
                "type": "FILE_OBJECT_FILE",
            },
        ),
        datum(
            "Event",
            {
                "uuid": "E1",
                "sequence": {"long": 1},
                "type": "EVENT_READ",
                "hostId": "H",
                "subject": {f"{CDM}.UUID": "P1"},
                "predicateObject": {f"{CDM}.UUID": "F1"},
                "predicateObjectPath": {"string": "/tmp/x"},
                "timestampNanos": 10,
            },
        ),
        datum(
            "Event",
            {
                "uuid": "E2",
                "sequence": {"long": 2},
                "type": "EVENT_WRITE",
                "hostId": "H",
                "subject": {f"{CDM}.UUID": "P1"},
                "predicateObject": {f"{CDM}.UUID": "F1"},
                "timestampNanos": 20,
            },
        ),
    ]
    write_jsonl(source, rows)

    parser = CdmJsonParser("cadets")
    entities = parser.collect_entities([source])
    events = list(parser.parse_events([source], entities, "train"))

    assert entities["P1"].node_type == "PROCESS"
    assert entities["F1"].node_type == "FILE"
    assert entities["P1"].cmdline == "bash -c x"
    assert len(events) == 2
    assert events[0].flow_src_uuid == "F1"
    assert events[0].flow_dst_uuid == "P1"
    assert events[1].flow_src_uuid == "P1"
    assert events[1].flow_dst_uuid == "F1"
    assert relation_name(events[1].event_type) == "write"


def test_execute_defaults_to_file_to_process_flow() -> None:
    flow = derive_information_flow("P1", "PROCESS", "F1", "FILE", "EVENT_EXECUTE")
    assert flow == ("F1", "FILE", "P1", "PROCESS")


def test_event_edge_id_distinguishes_predicate_edges(tmp_path: Path) -> None:
    source = tmp_path / "cadets.json"
    rows = [
        datum("Subject", {"uuid": "P1", "type": "SUBJECT_PROCESS"}),
        datum("FileObject", {"uuid": "F1", "type": "FILE_OBJECT_FILE"}),
        datum("FileObject", {"uuid": "F2", "type": "FILE_OBJECT_FILE"}),
        datum(
            "Event",
            {
                "uuid": "E1",
                "sequence": {"long": 1},
                "type": "EVENT_LINK",
                "subject": {f"{CDM}.UUID": "P1"},
                "predicateObject": {f"{CDM}.UUID": "F1"},
                "predicateObject2": {f"{CDM}.UUID": "F2"},
                "timestampNanos": 10,
            },
        ),
    ]
    write_jsonl(source, rows)

    parser = CdmJsonParser("cadets")
    entities = parser.collect_entities([source])
    events = list(parser.parse_events([source], entities, "train"))

    assert len(events) == 2
    assert events[0].event_uuid == events[1].event_uuid == "E1"
    assert events[0].event_edge_id != events[1].event_edge_id


def test_discover_json_chunks_filters_dataset_prefix(tmp_path: Path) -> None:
    names = [
        "ta1-cadets-e3-official.json",
        "ta1-cadets-e3-official.json.1",
        "ta1-cadets-e3-official.json.tar.gz",
        "ta1-theia-e3-official-6r.json",
        "ta1-trace-e3-official-1.json",
    ]
    for name in names:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    discovered = [path.name for path in discover_json_chunks(tmp_path, "cadets")]

    assert discovered == ["ta1-cadets-e3-official.json", "ta1-cadets-e3-official.json.1"]
