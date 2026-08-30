from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pyarrow.parquet as pq

from e3prep.graph.direction import relation_name
from e3prep.io import parquet_columns, read_parquet, write_json
from e3prep.sampling.episodes import NANOSECONDS
from e3prep.validation.statistics import describe_series


EVENT_COLUMNS = [
    "split",
    "event_edge_id",
    "event_uuid",
    "event_type",
    "timestamp_ns",
    "actor_uuid",
    "actor_type",
    "object_uuid",
    "object_type",
    "flow_src_uuid",
    "flow_src_type",
    "flow_dst_uuid",
    "flow_dst_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether an E3 store/subgraph dataset supports supervised or anomaly-detection protocols."
    )
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--subgraph-root", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--episode-max-duration-sec", type=int, default=600)
    parser.add_argument("--episode-inactivity-gap-sec", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=250_000)
    return parser.parse_args()


def normalize_uuid_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.upper()


def add_process_centers_by_split(
    df: pd.DataFrame,
    split_values: pd.Series,
    uuid_col: str,
    mask: pd.Series,
    output: dict[str, set[str]],
) -> None:
    for split, values in df.loc[mask].groupby(split_values[mask]):
        output[str(split)].update(values[uuid_col].dropna().astype(str).str.upper().tolist())


def entity_type_lookup(entities: pd.DataFrame) -> dict[str, str]:
    if entities.empty:
        return {}
    typed = entities[["uuid", "node_type"]].dropna(subset=["uuid"])
    return {
        str(row.uuid).upper(): str(row.node_type).upper()
        for row in typed.itertuples(index=False)
    }


def positive_label_uuids(labels: pd.DataFrame) -> set[str]:
    if labels.empty:
        return set()
    return set(labels.loc[labels["label"] == 1, "uuid"].dropna().astype(str).str.upper())


def count_episodes_from_timestamps(
    timestamps: list[int],
    max_duration_sec: int,
    inactivity_gap_sec: int,
) -> int:
    if not timestamps:
        return 0
    timestamps = sorted(set(int(value) for value in timestamps))
    max_duration_ns = max_duration_sec * NANOSECONDS
    inactivity_gap_ns = inactivity_gap_sec * NANOSECONDS
    episodes = 1
    current_start = timestamps[0]
    current_end = timestamps[0]
    for timestamp in timestamps[1:]:
        if timestamp - current_end > inactivity_gap_ns or timestamp - current_start > max_duration_ns:
            episodes += 1
            current_start = timestamp
        current_end = timestamp
    return episodes


def semantic_coverage(df: pd.DataFrame, type_column: str = "node_type") -> dict:
    if df.empty or type_column not in df.columns:
        return {}
    result = {}
    for node_type, group in df.groupby(df[type_column].fillna("UNKNOWN").astype(str)):
        row = {"total": int(len(group))}
        semantic_any = pd.Series(False, index=group.index)
        for column in ("name", "path", "cmdline", "ip", "port"):
            if column in group.columns:
                present = group[column].notna()
                row[f"{column}_non_null"] = int(present.sum())
                semantic_any = semantic_any | present
        row["any_semantic_non_null"] = int(semantic_any.sum())
        result[str(node_type)] = row
    return dict(sorted(result.items()))


def scan_event_label_coverage(
    events_path: Path,
    positive_uuids: set[str],
    positive_process_uuids: set[str],
    max_duration_sec: int,
    inactivity_gap_sec: int,
    batch_size: int,
) -> dict:
    available = set(parquet_columns(events_path))
    columns = [column for column in EVENT_COLUMNS if column in available]
    split_process_centers: dict[str, set[str]] = defaultdict(set)
    labeled_endpoint_uuids_by_split: dict[str, set[str]] = defaultdict(set)
    labeled_process_endpoint_uuids_by_split: dict[str, set[str]] = defaultdict(set)
    positive_process_timestamps: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    endpoint_label_events_by_split: dict[str, int] = defaultdict(int)
    process_label_events_by_split: dict[str, int] = defaultdict(int)

    parquet_file = pq.ParquetFile(events_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        if df.empty:
            continue
        if "split" not in df.columns:
            df["split"] = "unknown"
        for column in EVENT_COLUMNS:
            if column not in df.columns:
                df[column] = None

        split_values = df["split"].fillna("unknown").astype(str)
        actor_uuid = normalize_uuid_series(df["actor_uuid"])
        object_uuid = normalize_uuid_series(df["object_uuid"])
        flow_src_uuid = normalize_uuid_series(df["flow_src_uuid"])
        flow_dst_uuid = normalize_uuid_series(df["flow_dst_uuid"])
        actor_is_process = df["actor_type"].fillna("").astype(str).str.upper() == "PROCESS"
        object_is_process = df["object_type"].fillna("").astype(str).str.upper() == "PROCESS"
        flow_src_is_process = df["flow_src_type"].fillna("").astype(str).str.upper() == "PROCESS"
        flow_dst_is_process = df["flow_dst_type"].fillna("").astype(str).str.upper() == "PROCESS"

        add_process_centers_by_split(df, split_values, "actor_uuid", actor_is_process, split_process_centers)
        add_process_centers_by_split(df, split_values, "object_uuid", object_is_process, split_process_centers)
        add_process_centers_by_split(df, split_values, "flow_src_uuid", flow_src_is_process, split_process_centers)
        add_process_centers_by_split(df, split_values, "flow_dst_uuid", flow_dst_is_process, split_process_centers)

        endpoint_masks = [
            actor_uuid.isin(positive_uuids),
            object_uuid.isin(positive_uuids),
            flow_src_uuid.isin(positive_uuids),
            flow_dst_uuid.isin(positive_uuids),
        ]
        process_masks = [
            actor_uuid.isin(positive_process_uuids) & actor_is_process,
            object_uuid.isin(positive_process_uuids) & object_is_process,
            flow_src_uuid.isin(positive_process_uuids) & flow_src_is_process,
            flow_dst_uuid.isin(positive_process_uuids) & flow_dst_is_process,
        ]
        any_endpoint_label = endpoint_masks[0] | endpoint_masks[1] | endpoint_masks[2] | endpoint_masks[3]
        any_process_label = process_masks[0] | process_masks[1] | process_masks[2] | process_masks[3]

        for split, group_index in df.loc[any_endpoint_label].groupby(split_values[any_endpoint_label]).groups.items():
            split = str(split)
            endpoint_label_events_by_split[split] += len(group_index)
            for series, mask in zip(
                (actor_uuid, object_uuid, flow_src_uuid, flow_dst_uuid),
                endpoint_masks,
            ):
                labeled_endpoint_uuids_by_split[split].update(series.loc[group_index][mask.loc[group_index]].tolist())

        for split, group_index in df.loc[any_process_label].groupby(split_values[any_process_label]).groups.items():
            split = str(split)
            process_label_events_by_split[split] += len(group_index)
            for series, mask in zip(
                (actor_uuid, object_uuid, flow_src_uuid, flow_dst_uuid),
                process_masks,
            ):
                matched_index = mask.loc[group_index][mask.loc[group_index]].index
                uuids = series.loc[matched_index]
                labeled_process_endpoint_uuids_by_split[split].update(uuids.tolist())
                for uuid in sorted(set(uuids.tolist())):
                    uuid_index = uuids[uuids == uuid].index
                    timestamps = df.loc[uuid_index, "timestamp_ns"].dropna().astype("int64")
                    positive_process_timestamps[split][uuid].extend(timestamps.tolist())

    positive_center_episodes_by_split = {}
    for split, by_uuid in positive_process_timestamps.items():
        positive_center_episodes_by_split[split] = sum(
            count_episodes_from_timestamps(timestamps, max_duration_sec, inactivity_gap_sec)
            for timestamps in by_uuid.values()
        )

    return {
        "process_center_candidates_by_split": {
            split: len(uuids) for split, uuids in sorted(split_process_centers.items())
        },
        "labeled_endpoint_events_by_split": dict(sorted(endpoint_label_events_by_split.items())),
        "labeled_endpoint_uuids_by_split": {
            split: len(uuids) for split, uuids in sorted(labeled_endpoint_uuids_by_split.items())
        },
        "labeled_process_events_by_split": dict(sorted(process_label_events_by_split.items())),
        "labeled_process_center_uuids_by_split": {
            split: len(uuids) for split, uuids in sorted(labeled_process_endpoint_uuids_by_split.items())
        },
        "positive_center_episodes_by_split": dict(sorted(positive_center_episodes_by_split.items())),
    }


def subgraph_split_dirs(root: Path | None) -> dict[str, Path]:
    if root is None:
        return {}
    return {
        split: root / split
        for split in ("train", "val", "test")
        if (root / split / "metadata.parquet").exists()
    }


def merge_semantic_coverage(
    target: dict[str, dict[str, dict[str, int]]],
    split: str,
    node_type: str,
    group: pd.DataFrame,
) -> None:
    row = target.setdefault(split, {}).setdefault(
        node_type,
        {
            "total": 0,
            "name_non_null": 0,
            "path_non_null": 0,
            "cmdline_non_null": 0,
            "ip_non_null": 0,
            "port_non_null": 0,
            "any_semantic_non_null": 0,
        },
    )
    row["total"] += int(len(group))
    semantic_any = pd.Series(False, index=group.index)
    for column in ("name", "path", "cmdline", "ip", "port"):
        if column in group.columns:
            present = group[column].notna()
            row[f"{column}_non_null"] += int(present.sum())
            semantic_any = semantic_any | present
    row["any_semantic_non_null"] += int(semantic_any.sum())


def stream_node_sidecar_report(split_dirs: dict[str, Path], batch_size: int) -> dict:
    node_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    semantic_counts: dict[str, dict[str, dict[str, int]]] = {}
    for default_split, path in sorted(split_dirs.items()):
        nodes_path = path / "nodes.parquet"
        if not nodes_path.exists():
            continue
        available = set(parquet_columns(nodes_path))
        columns = [column for column in ("split", "node_type", "name", "path", "cmdline", "ip", "port") if column in available]
        if not columns:
            continue
        parquet_file = pq.ParquetFile(nodes_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            nodes = batch.to_pandas()
            if nodes.empty:
                continue
            if "split" not in nodes.columns:
                nodes["split"] = default_split
            else:
                nodes["split"] = nodes["split"].fillna(default_split)
            if "node_type" not in nodes.columns:
                nodes["node_type"] = "UNKNOWN"
            split_values = nodes["split"].fillna(default_split).astype(str)
            node_values = nodes["node_type"].fillna("UNKNOWN").astype(str)
            for split, group in nodes.groupby(split_values):
                split = str(split)
                node_type_counts[split].update(group["node_type"].fillna("UNKNOWN").astype(str).tolist())
            for (split, node_type), group in nodes.groupby([split_values, node_values]):
                merge_semantic_coverage(semantic_counts, str(split), str(node_type), group)
    return {
        "node_type_counts_by_split": {
            split: dict(sorted((node_type, int(count)) for node_type, count in counts.items()))
            for split, counts in sorted(node_type_counts.items())
        },
        "semantic_coverage_by_split": {
            split: dict(sorted(type_counts.items()))
            for split, type_counts in sorted(semantic_counts.items())
        },
    }


def stream_edge_sidecar_report(split_dirs: dict[str, Path], batch_size: int) -> dict:
    event_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    relation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for default_split, path in sorted(split_dirs.items()):
        edges_path = path / "edges.parquet"
        if not edges_path.exists():
            continue
        available = set(parquet_columns(edges_path))
        if "event_type" not in available:
            continue
        columns = ["event_type"]
        if "split" in available:
            columns.insert(0, "split")
        parquet_file = pq.ParquetFile(edges_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            edges = batch.to_pandas()
            if edges.empty:
                continue
            if "split" not in edges.columns:
                edges["split"] = default_split
            else:
                edges["split"] = edges["split"].fillna(default_split)
            edges["event_type"] = edges["event_type"].fillna("UNKNOWN").astype(str)
            edges["relation"] = edges["event_type"].apply(relation_name)
            split_values = edges["split"].fillna(default_split).astype(str)
            for split, group in edges.groupby(split_values):
                split = str(split)
                event_type_counts[split].update(group["event_type"].tolist())
                relation_counts[split].update(group["relation"].tolist())
    return {
        "relation_counts_by_split": {
            split: dict(sorted((relation, int(count)) for relation, count in counts.items()))
            for split, counts in sorted(relation_counts.items())
        },
        "event_type_counts_by_split": {
            split: dict(sorted((event_type, int(count)) for event_type, count in counts.items()))
            for split, counts in sorted(event_type_counts.items())
        },
    }


def read_subgraph_sidecars(root: Path | None, batch_size: int) -> dict:
    split_dirs = subgraph_split_dirs(root)
    if not split_dirs:
        return {}

    metadata_frames = []
    for split, path in sorted(split_dirs.items()):
        metadata = read_parquet(
            path / "metadata.parquet",
            columns=["sample_id", "split", "label", "center_uuid", "n_nodes", "n_edges", "positive_node_count"],
            allow_missing_columns=True,
        )
        metadata["split"] = metadata["split"].fillna(split)
        metadata_frames.append(metadata)

    metadata = pd.concat(metadata_frames, ignore_index=True)
    report = {
        "subgraph_dirs": {split: str(path) for split, path in sorted(split_dirs.items())},
        "sample_counts_by_split": metadata["split"].value_counts().sort_index().astype(int).to_dict(),
        "label_counts_by_split": {
            split: group["label"].value_counts().sort_index().astype(int).to_dict()
            for split, group in metadata.groupby("split")
        },
        "positive_node_count_by_split": {
            split: {
                "zero": int((group["positive_node_count"].fillna(0) == 0).sum()),
                "nonzero": int((group["positive_node_count"].fillna(0) > 0).sum()),
            }
            for split, group in metadata.groupby("split")
        },
        "nodes_per_subgraph": describe_series(metadata["n_nodes"]),
        "edges_per_subgraph": describe_series(metadata["n_edges"]),
    }

    node_report = stream_node_sidecar_report(split_dirs, batch_size)
    if node_report["node_type_counts_by_split"]:
        report.update(node_report)

    edge_report = stream_edge_sidecar_report(split_dirs, batch_size)
    if edge_report["relation_counts_by_split"]:
        report.update(edge_report)

    return report


def protocol_readiness(subgraph_report: dict, event_coverage: dict) -> dict:
    warnings: list[str] = []
    split_names = ("train", "val", "test")
    label_counts = subgraph_report.get("label_counts_by_split", {})
    if label_counts:
        positives = {split: int(label_counts.get(split, {}).get(1, 0)) for split in split_names}
        negatives = {split: int(label_counts.get(split, {}).get(0, 0)) for split in split_names}
        protocol_a_ready = all(positives[split] > 0 and negatives[split] > 0 for split in split_names)
        protocol_b_ready = negatives["train"] > 0 and positives["test"] > 0 and negatives["test"] > 0
    else:
        positives = {split: int(event_coverage.get("positive_center_episodes_by_split", {}).get(split, 0)) for split in split_names}
        negatives = {}
        protocol_a_ready = False
        protocol_b_ready = positives["test"] > 0
        warnings.append("No subgraph metadata found; readiness is estimated from labeled process-center episodes only.")

    for split in split_names:
        if positives.get(split, 0) == 0:
            warnings.append(f"{split} has zero positive process-center/subgraph labels under the current protocol.")
    if not protocol_a_ready:
        warnings.append("Protocol A supervised benign/malicious classification is not ready with the current split/labels.")
    if protocol_b_ready and positives.get("train", 0) == 0:
        warnings.append("Protocol B anomaly detection is plausible: train can be clean-normal and test contains positives.")

    return {
        "protocol_a_supervised_ready": protocol_a_ready,
        "protocol_b_anomaly_ready": protocol_b_ready,
        "positive_counts_used_for_readiness": positives,
        "negative_counts_used_for_readiness": negatives,
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    entities = read_parquet(
        args.store_dir / "entities.parquet",
        columns=["uuid", "node_type", "name", "path", "cmdline", "ip", "port"],
        allow_missing_columns=True,
    )
    labels_path = args.store_dir / "labels.parquet"
    labels = read_parquet(labels_path) if labels_path.exists() else pd.DataFrame()
    lookup = entity_type_lookup(entities)
    positive_uuids = positive_label_uuids(labels)
    positive_process_uuids = {uuid for uuid in positive_uuids if lookup.get(uuid) == "PROCESS"}

    labels_by_entity_type: dict[str, int] = defaultdict(int)
    for uuid in positive_uuids:
        labels_by_entity_type[lookup.get(uuid, "MISSING")] += 1

    event_coverage = scan_event_label_coverage(
        args.store_dir / "events.parquet",
        positive_uuids=positive_uuids,
        positive_process_uuids=positive_process_uuids,
        max_duration_sec=args.episode_max_duration_sec,
        inactivity_gap_sec=args.episode_inactivity_gap_sec,
        batch_size=args.batch_size,
    )
    subgraph_report = read_subgraph_sidecars(args.subgraph_root, args.batch_size)
    report = {
        "store_dir": str(args.store_dir),
        "subgraph_root": None if args.subgraph_root is None else str(args.subgraph_root),
        "episode_policy": {
            "max_duration_sec": args.episode_max_duration_sec,
            "inactivity_gap_sec": args.episode_inactivity_gap_sec,
        },
        "label_summary": {
            "labels_total": int(len(labels)),
            "positive_labels": len(positive_uuids),
            "positive_labels_by_entity_type": dict(sorted(labels_by_entity_type.items())),
            "positive_process_labels": len(positive_process_uuids),
        },
        "entity_semantic_coverage": semantic_coverage(entities),
        "event_label_coverage": event_coverage,
        "subgraph_report": subgraph_report,
    }
    report["protocol_readiness"] = protocol_readiness(subgraph_report, event_coverage)
    write_json(report, args.out)
    print(report)


if __name__ == "__main__":
    main()
