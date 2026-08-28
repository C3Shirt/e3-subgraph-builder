from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from e3prep.io import read_parquet, write_json


EVENT_COLUMNS = ["split", "actor_uuid", "actor_type", "object_uuid", "object_type"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select process center UUIDs for controlled subgraph sampling.")
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["touching-labels", "nonlabel-touching", "labeled-process"],
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def positive_labels(store_dir: Path) -> set[str]:
    labels_path = store_dir / "labels.parquet"
    if not labels_path.exists():
        return set()
    labels = read_parquet(labels_path, columns=["uuid", "label"])
    rows = labels[labels["label"] == 1]
    return set(rows["uuid"].dropna().astype(str).str.upper().tolist())


def process_endpoint_sets(events: pd.DataFrame, positive_uuids: set[str]) -> tuple[set[str], set[str]]:
    actor_uuid = events["actor_uuid"].fillna("").astype(str)
    object_uuid = events["object_uuid"].fillna("").astype(str)
    actor_upper = actor_uuid.str.upper()
    object_upper = object_uuid.str.upper()
    actor_is_process = events["actor_type"].fillna("").astype(str).str.upper() == "PROCESS"
    object_is_process = events["object_type"].fillna("").astype(str).str.upper() == "PROCESS"
    touches_label = actor_upper.isin(positive_uuids) | object_upper.isin(positive_uuids)

    all_processes = set(actor_uuid[actor_is_process].tolist())
    all_processes.update(object_uuid[object_is_process].tolist())

    touching = set(actor_uuid[actor_is_process & touches_label].tolist())
    touching.update(object_uuid[object_is_process & touches_label].tolist())
    touching.discard("")
    all_processes.discard("")
    return all_processes, touching


def labeled_processes(store_dir: Path, positive_uuids: set[str]) -> set[str]:
    entities = read_parquet(store_dir / "entities.parquet", columns=["uuid", "node_type"])
    process_rows = entities[entities["node_type"].fillna("").astype(str).str.upper() == "PROCESS"]
    process_by_upper = {uuid.upper(): uuid for uuid in process_rows["uuid"].dropna().astype(str)}
    return {process_by_upper[uuid] for uuid in positive_uuids if uuid in process_by_upper}


def choose(candidates: set[str], count: int, seed: int) -> list[str]:
    ordered = sorted(candidates)
    random.Random(seed).shuffle(ordered)
    return ordered[:count]


def main() -> None:
    args = parse_args()
    positive_uuids = positive_labels(args.store_dir)
    selected: list[str]
    summary = {
        "store_dir": str(args.store_dir),
        "split": args.split,
        "mode": args.mode,
        "positive_labels": len(positive_uuids),
        "requested_count": args.count,
        "seed": args.seed,
    }

    if args.mode == "labeled-process":
        candidates = labeled_processes(args.store_dir, positive_uuids)
        summary["candidate_processes"] = len(candidates)
        selected = choose(candidates, args.count, args.seed)
    else:
        events = read_parquet(
            args.store_dir / "events.parquet",
            columns=EVENT_COLUMNS,
            filters=[("split", "==", args.split)],
        )
        all_processes, touching = process_endpoint_sets(events, positive_uuids)
        if args.mode == "touching-labels":
            candidates = touching
        else:
            candidates = all_processes - touching
        summary["all_processes_in_split"] = len(all_processes)
        summary["processes_touching_labels"] = len(touching)
        summary["candidate_processes"] = len(candidates)
        selected = choose(candidates, args.count, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")
    summary["selected"] = len(selected)
    summary["out"] = str(args.out)
    summary_path = args.summary_out or args.out.with_suffix(".json")
    write_json(summary, summary_path)
    print(summary)


if __name__ == "__main__":
    main()
