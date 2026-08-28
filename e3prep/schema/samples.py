from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


SAMPLE_METADATA_COLUMNS = [
    "sample_id",
    "dataset",
    "split",
    "center_uuid",
    "label",
    "label_source",
    "label_confidence",
    "label_strategy",
    "t_start_ns",
    "t_end_ns",
    "context_start_ns",
    "context_end_ns",
    "n_nodes",
    "n_edges",
    "positive_node_count",
]


@dataclass
class ProcessEpisode:
    process_uuid: str
    split: str
    t_start_ns: int
    t_end_ns: int
    event_ids: list[str] = field(default_factory=list)
    episode_index: int = 0


@dataclass
class SubgraphSample:
    sample_id: str
    dataset: str
    split: str
    center_uuid: str
    label: int
    label_source: Optional[str]
    label_confidence: Optional[str]
    label_strategy: str
    t_start_ns: int
    t_end_ns: int
    context_start_ns: int
    context_end_ns: int
    nodes: dict[str, str]
    edges: list[dict]
    positive_node_count: int = 0

    def metadata(self) -> dict:
        row = asdict(self)
        row["n_nodes"] = len(self.nodes)
        row["n_edges"] = len(self.edges)
        return {column: row.get(column) for column in SAMPLE_METADATA_COLUMNS}
