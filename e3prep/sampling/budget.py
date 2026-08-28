from __future__ import annotations


def edge_touches_any(edge: dict, uuids: set[str] | None) -> bool:
    if not uuids:
        return False
    return any(
        str(edge.get(column, "")).upper() in uuids
        for column in ("actor_uuid", "object_uuid", "flow_src_uuid", "flow_dst_uuid")
    )


def edge_rank_key(edge: dict, center_uuid: str, midpoint_ns: int, positive_uuids: set[str] | None = None) -> tuple:
    timestamp_ns = int(edge.get("timestamp_ns") or midpoint_ns)
    touches_center = edge.get("flow_src_uuid") == center_uuid or edge.get("flow_dst_uuid") == center_uuid
    touches_positive = edge_touches_any(edge, positive_uuids)
    return (
        int(edge.get("_hop", 999)),
        0 if touches_center else 1,
        0 if touches_positive else 1,
        abs(timestamp_ns - midpoint_ns),
        str(edge.get("_direction", "")),
        str(edge.get("event_edge_id", "")),
        str(edge.get("event_uuid", "")),
    )


def apply_budget(
    edges: list[dict],
    center_uuid: str,
    max_nodes: int,
    max_edges: int,
    midpoint_ns: int,
    max_edges_per_pair: int | None = None,
    positive_uuids: set[str] | None = None,
) -> tuple[list[dict], dict[str, str]]:
    ranked_edges = sorted(edges, key=lambda edge: edge_rank_key(edge, center_uuid, midpoint_ns, positive_uuids))
    kept_edges: list[dict] = []
    kept_nodes: dict[str, str] = {center_uuid: "PROCESS"}
    pair_counts: dict[tuple[str, str, str], int] = {}

    for edge in ranked_edges:
        src = edge["flow_src_uuid"]
        dst = edge["flow_dst_uuid"]
        src_type = edge.get("flow_src_type") or "UNKNOWN"
        dst_type = edge.get("flow_dst_type") or "UNKNOWN"
        pair_key = (src, dst, str(edge.get("event_type", "EVENT_UNKNOWN")))
        if max_edges_per_pair and pair_counts.get(pair_key, 0) >= max_edges_per_pair:
            continue
        would_add = int(src not in kept_nodes) + int(dst not in kept_nodes)
        if len(kept_edges) >= max_edges:
            break
        if len(kept_nodes) + would_add > max_nodes:
            continue
        kept_edges.append(edge)
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
        kept_nodes.setdefault(src, src_type)
        kept_nodes.setdefault(dst, dst_type)

    return kept_edges, kept_nodes
