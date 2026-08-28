from __future__ import annotations

from collections import defaultdict

import torch
from torch_geometric.data import HeteroData

from e3prep.graph.direction import relation_name
from e3prep.schema.samples import SubgraphSample


def _safe_node_type(node_type: str | None) -> str:
    return (node_type or "UNKNOWN").lower()


def sample_to_heterodata(sample: SubgraphSample, entity_by_uuid: dict[str, dict]) -> HeteroData:
    data = HeteroData()
    nodes_by_type: dict[str, list[str]] = defaultdict(list)
    for uuid, node_type in sample.nodes.items():
        nodes_by_type[_safe_node_type(node_type)].append(uuid)

    local_index: dict[str, tuple[str, int]] = {}
    in_degree: dict[str, int] = defaultdict(int)
    out_degree: dict[str, int] = defaultdict(int)
    for edge in sample.edges:
        out_degree[edge["flow_src_uuid"]] += 1
        in_degree[edge["flow_dst_uuid"]] += 1

    for node_type, uuids in nodes_by_type.items():
        uuids = sorted(uuids)
        x_rows = []
        for idx, uuid in enumerate(uuids):
            local_index[uuid] = (node_type, idx)
            entity = entity_by_uuid.get(uuid, {})
            x_rows.append(
                [
                    1.0 if uuid == sample.center_uuid else 0.0,
                    float(in_degree.get(uuid, 0)),
                    float(out_degree.get(uuid, 0)),
                    1.0 if entity.get("name") else 0.0,
                    1.0 if entity.get("path") else 0.0,
                    1.0 if entity.get("ip") else 0.0,
                ]
            )
        data[node_type].x = torch.tensor(x_rows, dtype=torch.float)

    edge_groups: dict[tuple[str, str, str], list[list[int]]] = defaultdict(list)
    edge_times: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for edge in sample.edges:
        src_type, src_index = local_index[edge["flow_src_uuid"]]
        dst_type, dst_index = local_index[edge["flow_dst_uuid"]]
        key = (src_type, relation_name(edge["event_type"]), dst_type)
        edge_groups[key].append([src_index, dst_index])
        edge_times[key].append(int(edge["timestamp_ns"]))

    for key, pairs in edge_groups.items():
        data[key].edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
        data[key].edge_time = torch.tensor(edge_times[key], dtype=torch.long)

    center_type, center_index = local_index[sample.center_uuid]
    data.y = torch.tensor([sample.label], dtype=torch.long)
    data.center_type = center_type
    data.center_index = int(center_index)
    data.sample_id = sample.sample_id
    data.t_start_ns = int(sample.t_start_ns)
    data.t_end_ns = int(sample.t_end_ns)
    data.positive_node_count = int(sample.positive_node_count)
    return data
