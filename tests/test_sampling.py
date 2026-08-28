from __future__ import annotations

from pathlib import Path

import pandas as pd

from e3prep.graph.index import build_or_load_temporal_index, ensure_event_edge_ids
from e3prep.sampling.budget import apply_budget
from e3prep.sampling.process_sampler import ProcessSubgraphSampler, SamplerConfig


def test_process_sampler_uses_only_current_split_events() -> None:
    entities = pd.DataFrame(
        [
            {"uuid": "P1", "node_type": "PROCESS", "name": "bash"},
            {"uuid": "F1", "node_type": "FILE", "path": "/tmp/a"},
            {"uuid": "F2", "node_type": "FILE", "path": "/tmp/b"},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_uuid": "E_train",
                "dataset": "cadets",
                "host": "H",
                "split": "train",
                "actor_uuid": "P1",
                "actor_type": "PROCESS",
                "object_uuid": "F1",
                "object_type": "FILE",
                "object_path": "/tmp/a",
                "event_type": "EVENT_WRITE",
                "timestamp_ns": 10,
                "sequence": 1,
                "flow_src_uuid": "P1",
                "flow_src_type": "PROCESS",
                "flow_dst_uuid": "F1",
                "flow_dst_type": "FILE",
                "object_role": "predicateObject",
                "source_file": "train.json",
            },
            {
                "event_uuid": "E_test",
                "dataset": "cadets",
                "host": "H",
                "split": "test",
                "actor_uuid": "P1",
                "actor_type": "PROCESS",
                "object_uuid": "F2",
                "object_type": "FILE",
                "object_path": "/tmp/b",
                "event_type": "EVENT_WRITE",
                "timestamp_ns": 11,
                "sequence": 2,
                "flow_src_uuid": "P1",
                "flow_src_type": "PROCESS",
                "flow_dst_uuid": "F2",
                "flow_dst_type": "FILE",
                "object_role": "predicateObject",
                "source_file": "test.json",
            },
        ]
    )

    sampler = ProcessSubgraphSampler(
        "cadets",
        events[events["split"] == "train"],
        entities,
        config=SamplerConfig(backward_hops=1, forward_hops=1, backward_context_sec=1, forward_context_sec=1),
    )
    samples = list(sampler.iter_samples("train", max_duration_sec=600, inactivity_gap_sec=120))

    assert len(samples) == 1
    assert {edge["event_uuid"] for edge in samples[0].edges} == {"E_train"}
    assert "F2" not in samples[0].nodes


def test_process_sampler_caps_parallel_edges_per_pair() -> None:
    entities = pd.DataFrame(
        [
            {"uuid": "P1", "node_type": "PROCESS", "name": "bash"},
            {"uuid": "F1", "node_type": "FILE", "path": "/tmp/a"},
        ]
    )
    rows = []
    for idx in range(5):
        rows.append(
            {
                "event_uuid": f"E{idx}",
                "dataset": "cadets",
                "host": "H",
                "split": "train",
                "actor_uuid": "P1",
                "actor_type": "PROCESS",
                "object_uuid": "F1",
                "object_type": "FILE",
                "object_path": "/tmp/a",
                "event_type": "EVENT_WRITE",
                "timestamp_ns": 10 + idx,
                "sequence": idx,
                "flow_src_uuid": "P1",
                "flow_src_type": "PROCESS",
                "flow_dst_uuid": "F1",
                "flow_dst_type": "FILE",
                "object_role": "predicateObject",
                "source_file": "train.json",
            }
        )
    events = pd.DataFrame(rows)

    sampler = ProcessSubgraphSampler(
        "cadets",
        events,
        entities,
        config=SamplerConfig(
            backward_hops=0,
            forward_hops=1,
            backward_context_sec=1,
            forward_context_sec=1,
            max_edges_per_pair=2,
        ),
    )
    sample = next(iter(sampler.iter_samples("train", max_duration_sec=600, inactivity_gap_sec=120)))

    assert len(sample.edges) == 2


def test_process_sampler_keeps_distinct_edges_with_same_event_uuid() -> None:
    entities = pd.DataFrame(
        [
            {"uuid": "P1", "node_type": "PROCESS", "name": "bash"},
            {"uuid": "F1", "node_type": "FILE", "path": "/tmp/a"},
            {"uuid": "F2", "node_type": "FILE", "path": "/tmp/b"},
        ]
    )
    rows = []
    for object_uuid in ("F1", "F2"):
        rows.append(
            {
                "event_uuid": "E_shared",
                "dataset": "cadets",
                "host": "H",
                "split": "train",
                "actor_uuid": "P1",
                "actor_type": "PROCESS",
                "object_uuid": object_uuid,
                "object_type": "FILE",
                "object_path": f"/tmp/{object_uuid.lower()}",
                "event_type": "EVENT_WRITE",
                "timestamp_ns": 10,
                "sequence": 1,
                "flow_src_uuid": "P1",
                "flow_src_type": "PROCESS",
                "flow_dst_uuid": object_uuid,
                "flow_dst_type": "FILE",
                "object_role": "predicateObject" if object_uuid == "F1" else "predicateObject2",
                "source_file": "train.json",
            }
        )
    sampler = ProcessSubgraphSampler(
        "cadets",
        pd.DataFrame(rows),
        entities,
        config=SamplerConfig(backward_hops=0, forward_hops=1, backward_context_sec=1, forward_context_sec=1),
    )

    sample = next(iter(sampler.iter_samples("train", max_duration_sec=600, inactivity_gap_sec=120)))

    assert len(sample.edges) == 2
    assert len({edge["event_edge_id"] for edge in sample.edges}) == 2
    assert {edge["flow_dst_uuid"] for edge in sample.edges} == {"F1", "F2"}


def test_temporal_index_cache_roundtrip(tmp_path: Path) -> None:
    events = ensure_event_edge_ids(
        pd.DataFrame(
            [
                {
                    "event_uuid": "E1",
                    "dataset": "cadets",
                    "host": "H",
                    "split": "train",
                    "actor_uuid": "P1",
                    "actor_type": "PROCESS",
                    "object_uuid": "F1",
                    "object_type": "FILE",
                    "object_path": "/tmp/a",
                    "event_type": "EVENT_WRITE",
                    "timestamp_ns": 10,
                    "sequence": 1,
                    "flow_src_uuid": "P1",
                    "flow_src_type": "PROCESS",
                    "flow_dst_uuid": "F1",
                    "flow_dst_type": "FILE",
                    "object_role": "predicateObject",
                    "source_file": "train.json",
                }
            ]
        )
    )
    cache_path = tmp_path / "index.pkl"

    built_index, built_source = build_or_load_temporal_index(events, cache_path, {"split": "train"})
    loaded_index, loaded_source = build_or_load_temporal_index(events, cache_path, {"split": "train"})

    assert built_source == "built"
    assert loaded_source == "loaded"
    assert loaded_index.events[0].event_edge_id == built_index.events[0].event_edge_id


def test_budget_prioritizes_edges_touching_positive_labels() -> None:
    edges = [
        {
            "event_uuid": "E0",
            "event_type": "EVENT_WRITE",
            "timestamp_ns": 10,
            "flow_src_uuid": "P1",
            "flow_src_type": "PROCESS",
            "flow_dst_uuid": "F0",
            "flow_dst_type": "FILE",
        },
        {
            "event_uuid": "E1",
            "event_type": "EVENT_WRITE",
            "timestamp_ns": 100,
            "flow_src_uuid": "P1",
            "flow_src_type": "PROCESS",
            "flow_dst_uuid": "F1",
            "flow_dst_type": "FILE",
        },
    ]

    kept, nodes = apply_budget(
        edges,
        center_uuid="P1",
        max_nodes=2,
        max_edges=1,
        midpoint_ns=0,
        positive_uuids={"F1"},
    )

    assert [edge["event_uuid"] for edge in kept] == ["E1"]
    assert "F1" in nodes
