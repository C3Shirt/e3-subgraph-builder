from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e3prep.io import write_json, write_parquet_records
from e3prep.parser.cadets import SPLIT_MODES, discover_json_chunks, normalize_split_mode, resolve_split_paths, split_mode_warnings
from e3prep.parser.cdm import CdmJsonParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse DARPA E3 CDM JSON into canonical Parquet tables.")
    parser.add_argument("--dataset", default="cadets", choices=["cadets", "theia", "trace"])
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split-mode", default="chronological_disjoint", choices=[*SPLIT_MODES, "magic"])
    parser.add_argument(
        "--chronological-val-fraction",
        type=float,
        default=0.1,
        help="Validation interval fraction carved from the pre-test time period for chronological_disjoint.",
    )
    parser.add_argument(
        "--chronological-test-fraction",
        type=float,
        default=0.2,
        help="Final time interval fraction held out as test for chronological_disjoint.",
    )
    parser.add_argument("--strict-split-files", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--no-infer-entities-from-events",
        action="store_true",
        help="Only use explicit entity definitions in pass 1. This is faster for full E3 runs.",
    )
    return parser.parse_args()


def chronological_policy(timestamp_range: dict, val_fraction: float, test_fraction: float) -> dict:
    if not 0 <= val_fraction < 1:
        raise ValueError("--chronological-val-fraction must be in [0, 1)")
    if not 0 < test_fraction < 1:
        raise ValueError("--chronological-test-fraction must be in (0, 1)")
    min_ts = timestamp_range.get("min")
    max_ts = timestamp_range.get("max")
    if min_ts is None or max_ts is None:
        raise ValueError("Cannot create chronological_disjoint split without event timestamps")
    min_ts = int(min_ts)
    max_ts = int(max_ts)
    if max_ts <= min_ts:
        raise ValueError("Cannot create chronological_disjoint split from a zero-duration event range")

    span = max_ts - min_ts
    test_start_ns = min_ts + int(span * (1.0 - test_fraction))
    val_start_ns = None
    if val_fraction > 0:
        pre_test_span = test_start_ns - min_ts
        val_start_ns = min_ts + int(pre_test_span * (1.0 - val_fraction))
    return {
        "mode": "chronological_disjoint",
        "timestamp_range_ns": {"min": min_ts, "max": max_ts},
        "val_fraction_of_pre_test_period": val_fraction,
        "test_fraction_of_full_period": test_fraction,
        "val_start_ns": val_start_ns,
        "test_start_ns": test_start_ns,
        "missing_timestamp_split": "train",
        "intervals": {
            "train": {"start_ns": min_ts, "end_ns_exclusive": val_start_ns or test_start_ns},
            "val": None
            if val_start_ns is None
            else {"start_ns_inclusive": val_start_ns, "end_ns_exclusive": test_start_ns},
            "test": {"start_ns_inclusive": test_start_ns, "end_ns": max_ts},
        },
    }


def assign_chronological_split(timestamp_ns: int | None, policy: dict) -> str:
    if timestamp_ns is None:
        return str(policy["missing_timestamp_split"])
    timestamp_ns = int(timestamp_ns)
    val_start_ns = policy.get("val_start_ns")
    test_start_ns = int(policy["test_start_ns"])
    if val_start_ns is not None and timestamp_ns >= int(val_start_ns):
        if timestamp_ns < test_start_ns:
            return "val"
        return "test"
    if timestamp_ns >= test_start_ns:
        return "test"
    return "train"


def main() -> None:
    args = parse_args()
    args.split_mode = normalize_split_mode(args.split_mode)
    if not args.raw_dir.exists():
        raise FileNotFoundError(args.raw_dir)

    parser = CdmJsonParser(args.dataset)
    entity_paths = discover_json_chunks(args.raw_dir, args.dataset)
    split_paths = resolve_split_paths(
        args.dataset,
        args.raw_dir,
        split_mode=args.split_mode,
        strict=args.strict_split_files,
    )
    warnings = split_mode_warnings(args.dataset, args.split_mode)
    for warning in warnings:
        print(f"[split-warning] {warning}", flush=True)

    split_policy = {"mode": args.split_mode}
    if args.split_mode == "chronological_disjoint":
        timestamp_range = parser.event_timestamp_range(
            split_paths["all"],
            progress_every=args.progress_every,
        )
        split_policy = chronological_policy(
            timestamp_range,
            val_fraction=args.chronological_val_fraction,
            test_fraction=args.chronological_test_fraction,
        )

    print(
        {
            "dataset": args.dataset,
            "entity_files": len(entity_paths),
            "splits": {split: len(paths) for split, paths in split_paths.items()},
            "split_mode": args.split_mode,
            "infer_entities_from_events": not args.no_infer_entities_from_events,
        },
        flush=True,
    )

    entities = parser.collect_entities(
        entity_paths,
        progress_every=args.progress_every,
        infer_from_events=not args.no_infer_entities_from_events,
    )
    entity_count = write_parquet_records(
        (entity.to_dict() for entity in entities.values()),
        args.out_dir / "entities.parquet",
        "entities",
        chunk_size=args.chunk_size,
    )

    def event_rows():
        if args.split_mode == "chronological_disjoint":
            for event in parser.parse_events(
                split_paths["all"],
                entities,
                "chronological_disjoint",
                progress_every=args.progress_every,
            ):
                event.split = assign_chronological_split(event.timestamp_ns, split_policy)
                yield event.to_dict()
            return
        for split, paths in split_paths.items():
            for event in parser.parse_events(paths, entities, split, progress_every=args.progress_every):
                yield event.to_dict()

    event_count = write_parquet_records(
        event_rows(),
        args.out_dir / "events.parquet",
        "events",
        chunk_size=args.chunk_size,
    )

    summary = {
        "dataset": args.dataset,
        "raw_dir": str(args.raw_dir),
        "out_dir": str(args.out_dir),
        "split_mode": args.split_mode,
        "split_policy": split_policy,
        "split_warnings": warnings,
        "formal_split_eligible": args.split_mode == "chronological_disjoint",
        "entity_source_files": [str(path) for path in entity_paths],
        "split_files": {split: [str(path) for path in paths] for split, paths in split_paths.items()},
        "entities": entity_count,
        "events": event_count,
        "infer_entities_from_events": not args.no_infer_entities_from_events,
    }
    write_json(summary, args.out_dir / "parse_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
