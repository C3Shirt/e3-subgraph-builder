from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from e3prep.graph.index import TemporalGraphIndex
from e3prep.schema.samples import ProcessEpisode


NANOSECONDS = 1_000_000_000


def build_process_episodes(
    index: TemporalGraphIndex,
    process_uuids: Iterable[str],
    split: str,
    max_duration_sec: int = 600,
    inactivity_gap_sec: int = 120,
) -> list[ProcessEpisode]:
    max_duration_ns = max_duration_sec * NANOSECONDS
    inactivity_gap_ns = inactivity_gap_sec * NANOSECONDS
    episodes: list[ProcessEpisode] = []

    for process_uuid in sorted(set(process_uuids)):
        events = sorted(index.incident_edges(process_uuid), key=lambda event: (event.timestamp_ns, event.sequence or 0))
        if not events:
            continue
        current_start = events[0].timestamp_ns
        current_end = events[0].timestamp_ns
        current_ids = [events[0].event_edge_id]
        episode_index = 0

        for event in events[1:]:
            gap_too_large = event.timestamp_ns - current_end > inactivity_gap_ns
            duration_too_large = event.timestamp_ns - current_start > max_duration_ns
            if gap_too_large or duration_too_large:
                episodes.append(
                    ProcessEpisode(
                        process_uuid=process_uuid,
                        split=split,
                        t_start_ns=current_start,
                        t_end_ns=current_end,
                        event_ids=current_ids,
                        episode_index=episode_index,
                    )
                )
                episode_index += 1
                current_start = event.timestamp_ns
                current_ids = []
            current_end = event.timestamp_ns
            current_ids.append(event.event_edge_id)

        episodes.append(
            ProcessEpisode(
                process_uuid=process_uuid,
                split=split,
                t_start_ns=current_start,
                t_end_ns=current_end,
                event_ids=current_ids,
                episode_index=episode_index,
            )
        )
    return episodes


def process_uuids_from_entities(entities, canonical_type: str = "PROCESS") -> set[str]:
    return set(entities.loc[entities["node_type"] == canonical_type, "uuid"].astype(str).tolist())


def process_uuids_from_events(events) -> set[str]:
    candidates: dict[str, set[str]] = defaultdict(set)
    for _, row in events.iterrows():
        for uuid_col, type_col in (
            ("actor_uuid", "actor_type"),
            ("object_uuid", "object_type"),
            ("flow_src_uuid", "flow_src_type"),
            ("flow_dst_uuid", "flow_dst_type"),
        ):
            if str(row.get(type_col, "")).upper() == "PROCESS":
                candidates[str(row[uuid_col])].add(str(row.get("split", "unknown")))
    return set(candidates)
