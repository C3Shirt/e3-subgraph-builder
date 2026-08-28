from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from e3prep.export.shards import write_pyg_shards
from e3prep.graph.index import build_or_load_temporal_index, ensure_event_edge_ids
from e3prep.io import read_parquet, write_json
from e3prep.sampling.process_sampler import MVP_NODE_TYPES, ProcessSubgraphSampler, SamplerConfig


EVENT_COLUMNS_FOR_BUILD = [
    "event_edge_id",
    "event_uuid",
    "dataset",
    "host",
    "split",
    "actor_uuid",
    "actor_type",
    "object_uuid",
    "object_type",
    "object_path",
    "event_type",
    "timestamp_ns",
    "sequence",
    "flow_src_uuid",
    "flow_src_type",
    "flow_dst_uuid",
    "flow_dst_type",
    "object_role",
    "source_file",
]

ENTITY_COLUMNS_FOR_BUILD = ["uuid", "node_type", "name", "path", "cmdline", "ip", "port"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build process-centered temporal subgraphs from an event store.")
    parser.add_argument("--dataset", default="cadets", choices=["cadets", "theia", "trace"])
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split", action="append", help="Split to build. May be repeated. Defaults to all splits.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--episode-max-duration-sec", type=int, default=600)
    parser.add_argument("--episode-inactivity-gap-sec", type=int, default=120)
    parser.add_argument("--backward-hops", type=int, default=1)
    parser.add_argument("--forward-hops", type=int, default=2)
    parser.add_argument("--backward-context-sec", type=int, default=60)
    parser.add_argument("--forward-context-sec", type=int, default=120)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--max-edges", type=int, default=1024)
    parser.add_argument("--max-edges-per-pair", type=int, default=None)
    parser.add_argument("--max-edges-per-expansion-node", type=int, default=None)
    parser.add_argument("--min-nodes", type=int, default=2)
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--write-sidecar", action="store_true", help="Write sample nodes.parquet and edges.parquet.")
    parser.add_argument("--center-uuid", action="append", default=[], help="Restrict sampling to a center UUID.")
    parser.add_argument("--center-file", type=Path, default=None, help="Restrict sampling to UUIDs listed in a text file.")
    parser.add_argument(
        "--labeled-centers-only",
        action="store_true",
        help="DIAGNOSTIC-ONLY. Restrict sampling to positive labels that are also PROCESS entities.",
    )
    parser.add_argument(
        "--centers-touching-labels",
        action="store_true",
        help="DIAGNOSTIC-ONLY. Restrict sampling to process UUIDs whose events touch any positive labeled UUID.",
    )
    parser.add_argument(
        "--allow-diagnostic-label-selection",
        action="store_true",
        help="Required to use --labeled-centers-only or --centers-touching-labels. Do not use for formal datasets.",
    )
    parser.add_argument(
        "--node-type",
        action="append",
        default=[],
        help="Node type allowed in sampled subgraphs. Defaults to PROCESS, FILE, SOCKET. May be repeated.",
    )
    parser.add_argument("--center-limit", type=int, default=None, help="Limit candidate center UUIDs after sorting.")
    parser.add_argument(
        "--label-strategy",
        default="center",
        choices=["center", "subgraph_any_positive"],
        help="How to derive sample labels from labels.parquet.",
    )
    parser.add_argument(
        "--index-cache-dir",
        type=Path,
        default=None,
        help="Optional directory for pickled split TemporalGraphIndex caches.",
    )
    parser.add_argument("--rebuild-index-cache", action="store_true", help="Ignore and overwrite existing index caches.")
    return parser.parse_args()


def read_center_file(path: Path) -> list[str]:
    uuids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip().split()
            if value:
                uuids.append(value[0])
    return uuids


def resolve_candidate_centers(
    args: argparse.Namespace,
    entities: pd.DataFrame,
    labels: pd.DataFrame | None,
) -> set[str]:
    explicit = list(args.center_uuid or [])
    if args.center_file:
        explicit.extend(read_center_file(args.center_file))

    candidates: set[str] = set(str(uuid) for uuid in explicit)
    if args.labeled_centers_only:
        if labels is None or labels.empty:
            raise ValueError("--labeled-centers-only requires labels.parquet")
        positive_uuids = set(labels.loc[labels["label"] == 1, "uuid"].dropna().astype(str).str.upper())
        process_rows = entities[entities["node_type"].astype(str).str.upper() == "PROCESS"]
        process_uuids = set(process_rows["uuid"].dropna().astype(str))
        process_by_upper = {uuid.upper(): uuid for uuid in process_uuids}
        candidates.update(process_by_upper[uuid] for uuid in positive_uuids if uuid in process_by_upper)

    return candidates


def apply_center_limit(candidates: set[str], center_limit: int | None) -> set[str]:
    if center_limit is None:
        return candidates
    return set(sorted(candidates)[:center_limit])


def centers_touching_positive_labels(events: pd.DataFrame, labels: pd.DataFrame | None) -> set[str]:
    if labels is None or labels.empty:
        raise ValueError("--centers-touching-labels requires labels.parquet")
    positive_uuids = set(labels.loc[labels["label"] == 1, "uuid"].dropna().astype(str).str.upper())
    if not positive_uuids:
        return set()

    actor_keys = events["actor_uuid"].fillna("").astype(str).str.upper()
    object_keys = events["object_uuid"].fillna("").astype(str).str.upper()
    actor_is_process = events["actor_type"].astype(str).str.upper() == "PROCESS"
    object_is_process = events["object_type"].astype(str).str.upper() == "PROCESS"
    actor_touches_label = actor_keys.isin(positive_uuids) | object_keys.isin(positive_uuids)
    object_touches_label = actor_touches_label

    centers = set(events.loc[actor_is_process & actor_touches_label, "actor_uuid"].dropna().astype(str).tolist())
    centers.update(events.loc[object_is_process & object_touches_label, "object_uuid"].dropna().astype(str).tolist())
    return centers


def split_names_from_events(events_path: Path) -> list[str]:
    split_df = read_parquet(events_path, columns=["split"])
    return sorted(split_df["split"].dropna().astype(str).unique().tolist())


def read_split_events(events_path: Path, split: str) -> pd.DataFrame:
    events = read_parquet(
        events_path,
        columns=EVENT_COLUMNS_FOR_BUILD,
        filters=[("split", "==", split)],
        allow_missing_columns=True,
    )
    return ensure_event_edge_ids(events)


def index_cache_key(events_path: Path, split: str, rows: int) -> dict:
    stat = events_path.stat()
    return {
        "version": 1,
        "events_path": str(events_path.resolve()),
        "events_size": stat.st_size,
        "events_mtime_ns": stat.st_mtime_ns,
        "split": split,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if (args.labeled_centers_only or args.centers_touching_labels) and not args.allow_diagnostic_label_selection:
        raise ValueError(
            "--labeled-centers-only and --centers-touching-labels are diagnostic-only. "
            "Pass --allow-diagnostic-label-selection for exploratory runs; omit them for formal dataset builds."
        )
    events_path = args.store_dir / "events.parquet"
    entities = read_parquet(args.store_dir / "entities.parquet", columns=ENTITY_COLUMNS_FOR_BUILD)
    labels_path = args.store_dir / "labels.parquet"
    labels = read_parquet(labels_path) if labels_path.exists() else None

    split_names = args.split or split_names_from_events(events_path)
    candidate_centers = resolve_candidate_centers(args, entities, labels)
    cfg = SamplerConfig(
        backward_hops=args.backward_hops,
        forward_hops=args.forward_hops,
        backward_context_sec=args.backward_context_sec,
        forward_context_sec=args.forward_context_sec,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        min_nodes=args.min_nodes,
        label_strategy=args.label_strategy,
        max_edges_per_pair=args.max_edges_per_pair,
        max_edges_per_expansion_node=args.max_edges_per_expansion_node,
        allowed_node_types=tuple(node_type.upper() for node_type in (args.node_type or MVP_NODE_TYPES)),
    )

    summary = {}
    for split in split_names:
        print(f"[build-subgraphs] reading split={split}", flush=True)
        split_events = read_split_events(events_path, split)
        print(f"[build-subgraphs] split={split} events={len(split_events)}", flush=True)
        split_candidate_centers = set(candidate_centers)
        if args.centers_touching_labels:
            split_candidate_centers.update(centers_touching_positive_labels(split_events, labels))
        split_candidate_centers = apply_center_limit(split_candidate_centers, args.center_limit)
        center_uuids = split_candidate_centers if split_candidate_centers else None
        print(
            f"[build-subgraphs] split={split} candidate_centers="
            f"{'all' if center_uuids is None else len(center_uuids)}",
            flush=True,
        )
        print(f"[build-subgraphs] split={split} building temporal index", flush=True)
        cache_path = None
        if args.index_cache_dir:
            cache_path = args.index_cache_dir / f"{args.dataset}_{split}_temporal_index.pkl"
        index, index_source = build_or_load_temporal_index(
            split_events,
            cache_path=cache_path,
            cache_key=index_cache_key(events_path, split, len(split_events)),
            rebuild=args.rebuild_index_cache,
        )
        print(f"[build-subgraphs] split={split} temporal index {index_source}", flush=True)
        sampler = ProcessSubgraphSampler(
            args.dataset,
            split_events,
            entities,
            labels=labels,
            config=cfg,
            center_uuids=center_uuids,
            index=index,
        )
        print(f"[build-subgraphs] split={split} writing shards", flush=True)
        samples = sampler.iter_samples(
            split=split,
            max_duration_sec=args.episode_max_duration_sec,
            inactivity_gap_sec=args.episode_inactivity_gap_sec,
            max_samples=args.max_samples,
        )
        summary[split] = write_pyg_shards(
            samples,
            entities,
            args.out_dir / split,
            args.samples_per_shard,
            labels=labels,
            write_sidecar=args.write_sidecar,
        )
        summary[split]["candidate_centers"] = None if center_uuids is None else len(center_uuids)
        summary[split]["label_strategy"] = args.label_strategy
        summary[split]["node_type_policy"] = {
            "sampled_node_types": list(cfg.allowed_node_types),
            "canonical_tables_keep_all_types": True,
        }
        summary[split]["diagnostic_label_selection"] = {
            "enabled": bool(args.labeled_centers_only or args.centers_touching_labels),
            "labeled_centers_only": bool(args.labeled_centers_only),
            "centers_touching_labels": bool(args.centers_touching_labels),
            "formal_dataset_eligible": not bool(args.labeled_centers_only or args.centers_touching_labels),
        }
        summary[split]["index_cache"] = None if cache_path is None else str(cache_path)
        summary[split]["index_source"] = index_source
    write_json(summary, args.out_dir / "subgraph_build_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
