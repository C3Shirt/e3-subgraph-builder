from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from e3prep.graph.index import TemporalGraphIndex
from e3prep.sampling.budget import apply_budget
from e3prep.sampling.episodes import NANOSECONDS, build_process_episodes, process_uuids_from_entities
from e3prep.schema.samples import ProcessEpisode, SubgraphSample


@dataclass
class SamplerConfig:
    backward_hops: int = 1
    forward_hops: int = 2
    backward_context_sec: int = 60
    forward_context_sec: int = 120
    max_nodes: int = 256
    max_edges: int = 1024
    min_nodes: int = 2
    label_strategy: str = "center"
    max_edges_per_pair: int | None = None
    max_edges_per_expansion_node: int | None = None


class LabelStore:
    def __init__(self, labels: Optional[pd.DataFrame] = None):
        self.labels = labels if labels is not None else pd.DataFrame()
        if self.labels.empty:
            self.positive_uuids = set()
        else:
            positives = self.labels[self.labels["label"] == 1]
            self.positive_uuids = set(positives["uuid"].dropna().astype(str).str.upper().tolist())
        if self.labels.empty or "label_source" not in self.labels.columns:
            self.primary_source = None
        else:
            sources = self.labels["label_source"].dropna().astype(str)
            self.primary_source = None if sources.empty else sources.mode().iloc[0]

    def label_episode(self, center_uuid: str, start_ns: int, end_ns: int) -> tuple[int, Optional[str], Optional[str]]:
        if self.labels.empty:
            return 0, None, None
        rows = self.labels[self.labels["uuid"].astype(str) == str(center_uuid)]
        if rows.empty:
            return 0, None, None
        for _, row in rows.iterrows():
            label_start = row.get("start_time_ns")
            label_end = row.get("end_time_ns")
            if pd.isna(label_start) or pd.isna(label_end):
                return int(row.get("label", 1)), row.get("label_source"), row.get("confidence")
            if int(label_start) <= end_ns and int(label_end) >= start_ns:
                return int(row.get("label", 1)), row.get("label_source"), row.get("confidence")
        return 0, None, None

    def positive_node_count(self, nodes: Iterable[str]) -> int:
        if not self.positive_uuids:
            return 0
        return sum(1 for node in nodes if str(node).upper() in self.positive_uuids)

    def event_touches_positive(self, event) -> bool:
        if not self.positive_uuids:
            return False
        return (
            event.actor_uuid.upper() in self.positive_uuids
            or event.object_uuid.upper() in self.positive_uuids
            or event.flow_src_uuid.upper() in self.positive_uuids
            or event.flow_dst_uuid.upper() in self.positive_uuids
        )


class ProcessSubgraphSampler:
    def __init__(
        self,
        dataset: str,
        events: pd.DataFrame,
        entities: pd.DataFrame,
        labels: Optional[pd.DataFrame] = None,
        config: Optional[SamplerConfig] = None,
        center_uuids: Optional[Iterable[str]] = None,
        index: Optional[TemporalGraphIndex] = None,
    ):
        self.dataset = dataset
        self.events = events
        self.entities = entities
        self.config = config or SamplerConfig()
        self.index = index or TemporalGraphIndex(events)
        self.labels = LabelStore(labels)
        self.center_uuids = set(str(uuid) for uuid in center_uuids) if center_uuids is not None else None

    def build_episodes(self, split: str, max_duration_sec: int, inactivity_gap_sec: int) -> list[ProcessEpisode]:
        process_uuids = self.center_uuids if self.center_uuids is not None else process_uuids_from_entities(self.entities)
        return build_process_episodes(
            self.index,
            process_uuids=process_uuids,
            split=split,
            max_duration_sec=max_duration_sec,
            inactivity_gap_sec=inactivity_gap_sec,
        )

    def sample_episode(self, episode: ProcessEpisode) -> Optional[SubgraphSample]:
        cfg = self.config
        context_start_ns = episode.t_start_ns - cfg.backward_context_sec * NANOSECONDS
        context_end_ns = episode.t_end_ns + cfg.forward_context_sec * NANOSECONDS
        midpoint_ns = episode.t_start_ns + ((episode.t_end_ns - episode.t_start_ns) // 2)
        selected: dict[str, dict] = {}

        self._expand(
            start_nodes={episode.process_uuid},
            hops=cfg.backward_hops,
            context_start_ns=context_start_ns,
            context_end_ns=context_end_ns,
            selected=selected,
            direction="backward",
            midpoint_ns=midpoint_ns,
        )
        self._expand(
            start_nodes={episode.process_uuid},
            hops=cfg.forward_hops,
            context_start_ns=context_start_ns,
            context_end_ns=context_end_ns,
            selected=selected,
            direction="forward",
            midpoint_ns=midpoint_ns,
        )

        edges, nodes = apply_budget(
            list(selected.values()),
            center_uuid=episode.process_uuid,
            max_nodes=cfg.max_nodes,
            max_edges=cfg.max_edges,
            midpoint_ns=midpoint_ns,
            max_edges_per_pair=cfg.max_edges_per_pair,
            positive_uuids=self.labels.positive_uuids,
        )
        if len(nodes) < cfg.min_nodes or not edges:
            return None

        positive_node_count = self.labels.positive_node_count(nodes)
        label, label_source, label_confidence = self.labels.label_episode(
            episode.process_uuid,
            episode.t_start_ns,
            episode.t_end_ns,
        )
        label_strategy = cfg.label_strategy
        if label_strategy == "subgraph_any_positive" and label == 0 and positive_node_count > 0:
            label = 1
            label_source = self.labels.primary_source
            label_confidence = "subgraph_contains_labeled_node"
        sample_id = (
            f"{self.dataset}_{episode.split}_{episode.process_uuid}_"
            f"ep{episode.episode_index:04d}_{episode.t_start_ns}_{episode.t_end_ns}"
        )
        return SubgraphSample(
            sample_id=sample_id,
            dataset=self.dataset,
            split=episode.split,
            center_uuid=episode.process_uuid,
            label=label,
            label_source=label_source,
            label_confidence=label_confidence,
            label_strategy=label_strategy,
            t_start_ns=episode.t_start_ns,
            t_end_ns=episode.t_end_ns,
            context_start_ns=context_start_ns,
            context_end_ns=context_end_ns,
            nodes=nodes,
            edges=edges,
            positive_node_count=positive_node_count,
        )

    def iter_samples(
        self,
        split: str,
        max_duration_sec: int,
        inactivity_gap_sec: int,
        max_samples: Optional[int] = None,
    ) -> Iterable[SubgraphSample]:
        produced = 0
        for episode in self.build_episodes(split, max_duration_sec, inactivity_gap_sec):
            sample = self.sample_episode(episode)
            if sample is None:
                continue
            yield sample
            produced += 1
            if max_samples is not None and produced >= max_samples:
                break

    def _expand(
        self,
        start_nodes: set[str],
        hops: int,
        context_start_ns: int,
        context_end_ns: int,
        selected: dict[str, dict],
        direction: str,
        midpoint_ns: int,
    ) -> None:
        frontier = set(start_nodes)
        visited = set(start_nodes)
        for hop in range(1, hops + 1):
            next_frontier: set[str] = set()
            for node_uuid in sorted(frontier):
                if direction == "backward":
                    edges = self.index.incoming_edges(node_uuid, context_start_ns, context_end_ns)
                    neighbor_column = "flow_src_uuid"
                else:
                    edges = self.index.outgoing_edges(node_uuid, context_start_ns, context_end_ns)
                    neighbor_column = "flow_dst_uuid"
                edge_list = list(edges)
                if self.config.max_edges_per_expansion_node:
                    edge_list = sorted(
                        edge_list,
                        key=lambda event: (
                            0 if self.labels.event_touches_positive(event) else 1,
                            abs(event.timestamp_ns - midpoint_ns),
                            event.sequence or 0,
                            event.event_edge_id,
                            event.event_uuid,
                        ),
                    )[: self.config.max_edges_per_expansion_node]
                for event in edge_list:
                    row = event.to_dict()
                    row["_hop"] = hop
                    row["_direction"] = direction
                    selected.setdefault(event.event_edge_id, row)
                    neighbor = row[neighbor_column]
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            visited.update(next_frontier)
            frontier = next_frontier
