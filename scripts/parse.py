from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e3prep.io import write_json, write_parquet_records
from e3prep.parser.cadets import discover_json_chunks, resolve_split_paths
from e3prep.parser.cdm import CdmJsonParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse DARPA E3 CDM JSON into canonical Parquet tables.")
    parser.add_argument("--dataset", default="cadets", choices=["cadets", "theia", "trace"])
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split-mode", default="magic", choices=["magic", "all"])
    parser.add_argument("--strict-split-files", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--no-infer-entities-from-events",
        action="store_true",
        help="Only use explicit entity definitions in pass 1. This is faster for full E3 runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    print(
        {
            "dataset": args.dataset,
            "entity_files": len(entity_paths),
            "splits": {split: len(paths) for split, paths in split_paths.items()},
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
