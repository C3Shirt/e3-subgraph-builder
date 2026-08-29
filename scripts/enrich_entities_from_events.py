from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import pyarrow.parquet as pq

from e3prep.io import parquet_columns, write_json, write_parquet_records


EVENT_COLUMNS = ["event_type", "actor_uuid", "actor_type", "object_uuid", "object_type", "object_path"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich entity metadata from canonical events. This is label-agnostic and does not alter event splits."
        )
    )
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--out-entities", type=Path, default=None)
    parser.add_argument("--replace", action="store_true", help="Overwrite entities.parquet after making a backup.")
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument(
        "--enrich-process-from-execute-object",
        action="store_true",
        help=(
            "DIAGNOSTIC-ONLY. Fill PROCESS name/path from EVENT_EXECUTE object_path. "
            "CADETS often records dynamic-loader paths here, so this is disabled by default."
        ),
    )
    return parser.parse_args()


def first_non_empty(current, candidate):
    if current is not None and not pd.isna(current) and str(current) != "":
        return current
    if candidate is None or pd.isna(candidate) or str(candidate) == "":
        return current
    return str(candidate)


def update_first(mapping: dict[str, str], uuid, value) -> None:
    if uuid is None or pd.isna(uuid) or value is None or pd.isna(value):
        return
    uuid = str(uuid).upper()
    value = str(value)
    if not uuid or not value:
        return
    mapping.setdefault(uuid, value)


def collect_event_metadata(
    events_path: Path,
    batch_size: int,
    enrich_process_from_execute_object: bool = False,
) -> tuple[dict[str, str], dict[str, str], dict]:
    available = set(parquet_columns(events_path))
    columns = [column for column in EVENT_COLUMNS if column in available]
    file_paths: dict[str, str] = {}
    process_exec_paths: dict[str, str] = {}
    file_path_conflicts: dict[str, set[str]] = defaultdict(set)
    process_exec_conflicts: dict[str, set[str]] = defaultdict(set)
    rows_with_object_path = 0

    parquet_file = pq.ParquetFile(events_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        df = batch.to_pandas()
        for column in EVENT_COLUMNS:
            if column not in df.columns:
                df[column] = None
        with_path = df[df["object_path"].notna()].copy()
        with_path = with_path[with_path["object_path"].astype(str) != ""]
        rows_with_object_path += int(len(with_path))
        if with_path.empty:
            continue

        file_rows = with_path[with_path["object_type"].fillna("").astype(str).str.upper() == "FILE"]
        for row in file_rows[["object_uuid", "object_path"]].itertuples(index=False):
            uuid = str(row.object_uuid).upper() if row.object_uuid is not None and not pd.isna(row.object_uuid) else ""
            if uuid and uuid in file_paths and file_paths[uuid] != str(row.object_path):
                file_path_conflicts[uuid].update((file_paths[uuid], str(row.object_path)))
            update_first(file_paths, row.object_uuid, row.object_path)

        if enrich_process_from_execute_object:
            exec_rows = with_path[
                (with_path["actor_type"].fillna("").astype(str).str.upper() == "PROCESS")
                & (with_path["event_type"].fillna("").astype(str).str.upper() == "EVENT_EXECUTE")
            ]
            for row in exec_rows[["actor_uuid", "object_path"]].itertuples(index=False):
                uuid = str(row.actor_uuid).upper() if row.actor_uuid is not None and not pd.isna(row.actor_uuid) else ""
                if uuid and uuid in process_exec_paths and process_exec_paths[uuid] != str(row.object_path):
                    process_exec_conflicts[uuid].update((process_exec_paths[uuid], str(row.object_path)))
                update_first(process_exec_paths, row.actor_uuid, row.object_path)

    summary = {
        "rows_with_object_path": rows_with_object_path,
        "file_path_candidates": len(file_paths),
        "process_exec_path_candidates": len(process_exec_paths),
        "process_exec_object_path_enrichment_enabled": bool(enrich_process_from_execute_object),
        "file_path_conflicts": len(file_path_conflicts),
        "process_exec_path_conflicts": len(process_exec_conflicts),
        "file_path_conflict_examples": {
            uuid: sorted(values)[:5] for uuid, values in list(sorted(file_path_conflicts.items()))[:10]
        },
        "process_exec_path_conflict_examples": {
            uuid: sorted(values)[:5] for uuid, values in list(sorted(process_exec_conflicts.items()))[:10]
        },
    }
    return file_paths, process_exec_paths, summary


def enrich_entities(entities: pd.DataFrame, file_paths: dict[str, str], process_exec_paths: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    enriched = entities.copy()
    enriched["_uuid_upper"] = enriched["uuid"].fillna("").astype(str).str.upper()
    updated_by_field: dict[str, int] = defaultdict(int)
    updated_by_type: dict[str, int] = defaultdict(int)

    for index, row in enriched.iterrows():
        uuid = row["_uuid_upper"]
        node_type = str(row.get("node_type") or "").upper()
        candidate = None
        if node_type == "FILE":
            candidate = file_paths.get(uuid)
        elif node_type == "PROCESS":
            candidate = process_exec_paths.get(uuid)
        if not candidate:
            continue

        old_path = row.get("path")
        old_name = row.get("name")
        new_path = first_non_empty(old_path, candidate)
        new_name = first_non_empty(old_name, candidate)
        if new_path != old_path:
            enriched.at[index, "path"] = new_path
            updated_by_field["path"] += 1
            updated_by_type[node_type] += 1
        if new_name != old_name:
            enriched.at[index, "name"] = new_name
            updated_by_field["name"] += 1

    enriched = enriched.drop(columns=["_uuid_upper"])
    summary = {
        "updated_by_field": dict(sorted(updated_by_field.items())),
        "updated_by_type": dict(sorted(updated_by_type.items())),
        "semantic_coverage_after": semantic_coverage(enriched),
    }
    return enriched, summary


def semantic_coverage(entities: pd.DataFrame) -> dict:
    result = {}
    for node_type, group in entities.groupby(entities["node_type"].fillna("UNKNOWN").astype(str)):
        row = {"total": int(len(group))}
        for column in ("name", "path", "cmdline", "ip", "port"):
            if column in group.columns:
                row[f"{column}_non_null"] = int(group[column].notna().sum())
        result[str(node_type)] = row
    return dict(sorted(result.items()))


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    entities_path = args.store_dir / "entities.parquet"
    if args.replace:
        backup_path = args.store_dir / "entities.before_event_enrichment.parquet"
        return entities_path, entities_path, backup_path
    out_entities = args.out_entities or (args.store_dir / "entities.enriched.parquet")
    return entities_path, out_entities, None


def main() -> None:
    args = parse_args()
    source_entities, out_entities, backup_path = resolve_output_paths(args)
    events_path = args.store_dir / "events.parquet"
    entities = pd.read_parquet(source_entities)
    file_paths, process_exec_paths, event_summary = collect_event_metadata(
        events_path,
        args.batch_size,
        enrich_process_from_execute_object=args.enrich_process_from_execute_object,
    )
    enriched, enrich_summary = enrich_entities(entities, file_paths, process_exec_paths)

    if backup_path is not None and not backup_path.exists():
        shutil.copy2(source_entities, backup_path)
    write_parquet_records(enriched.to_dict("records"), out_entities, "entities")

    summary = {
        "store_dir": str(args.store_dir),
        "source_entities": str(source_entities),
        "out_entities": str(out_entities),
        "backup_entities": None if backup_path is None else str(backup_path),
        "replace": bool(args.replace),
        "semantic_coverage_before": semantic_coverage(entities),
        **event_summary,
        **enrich_summary,
    }
    summary_path = out_entities.with_suffix(".summary.json")
    write_json(summary, summary_path)
    print(summary)


if __name__ == "__main__":
    main()
