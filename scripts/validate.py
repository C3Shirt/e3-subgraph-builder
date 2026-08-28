from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e3prep.io import write_json
from e3prep.validation.coverage import label_coverage_report
from e3prep.validation.leakage import leakage_report, overlap_total
from e3prep.validation.relations import relation_audit_report
from e3prep.validation.statistics import build_dataset_report


FORMAL_SUBGRAPH_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an E3 event store and optional subgraph directories.")
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument(
        "--subgraph-root",
        type=Path,
        default=None,
        help="Root containing train/val/test subgraph directories. Use this for formal dataset validation.",
    )
    parser.add_argument(
        "--subgraph-dir",
        action="append",
        default=[],
        help="Subgraph directory to validate. May be repeated; accepts either PATH or split=PATH.",
    )
    parser.add_argument(
        "--required-subgraph-split",
        action="append",
        default=None,
        help="Required split below --subgraph-root. Defaults to train, val, and test.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--relation-top-n", type=int, default=50)
    parser.add_argument(
        "--fail-on-event-leakage",
        action="store_true",
        help="Exit nonzero if shared event_edge_id appears across train/val/test event or subgraph splits.",
    )
    return parser.parse_args()


def parse_subgraph_dir_spec(value: str) -> tuple[str, Path]:
    if "=" in value:
        split, path_text = value.split("=", 1)
        split = split.strip()
        if not split:
            raise ValueError(f"Invalid --subgraph-dir split name: {value}")
        return split, Path(path_text)
    path = Path(value)
    return path.name, path


def require_subgraph_metadata(path: Path) -> None:
    metadata_path = path / "metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)


def resolve_subgraph_dirs(args: argparse.Namespace) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if args.subgraph_root is not None:
        required_splits = args.required_subgraph_split or list(FORMAL_SUBGRAPH_SPLITS)
        missing = []
        for split in required_splits:
            path = args.subgraph_root / split
            if not path.exists():
                missing.append(path)
            else:
                require_subgraph_metadata(path)
                result[str(split)] = path
        if missing:
            missing_text = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"Missing required subgraph directories:\n{missing_text}")

    for spec in args.subgraph_dir:
        split, path = parse_subgraph_dir_spec(str(spec))
        if not path.exists():
            raise FileNotFoundError(path)
        require_subgraph_metadata(path)
        result[split] = path
    return result


def event_leakage_failures(leakage: Mapping[str, object]) -> list[str]:
    failures = []
    canonical = overlap_total(leakage.get("shared_event_edge_ids"))
    if canonical is None:
        failures.append("shared_event_edge_ids unavailable")
    elif canonical > 0:
        failures.append(f"shared_event_edge_ids={canonical}")

    subgraph = overlap_total(leakage.get("shared_subgraph_event_edge_ids"))
    if "subgraph_dirs" in leakage and subgraph is None:
        failures.append("shared_subgraph_event_edge_ids unavailable")
    elif subgraph is not None and subgraph > 0:
        failures.append(f"shared_subgraph_event_edge_ids={subgraph}")
    return failures


def main() -> None:
    args = parse_args()
    subgraph_dirs = resolve_subgraph_dirs(args)
    subgraph_arg = subgraph_dirs or None
    report = build_dataset_report(args.store_dir, subgraph_arg)
    report["label_coverage"] = label_coverage_report(args.store_dir, subgraph_arg)
    report["relation_audit"] = relation_audit_report(args.store_dir, args.relation_top_n)
    report["leakage"] = leakage_report(args.store_dir, subgraph_arg)
    report["validation_policy"] = {
        "subgraph_dirs": {split: str(path) for split, path in sorted(subgraph_dirs.items())},
        "formal_expected_splits": list(FORMAL_SUBGRAPH_SPLITS),
        "fail_on_event_leakage": bool(args.fail_on_event_leakage),
    }
    out_path = args.out or (args.store_dir / "dataset_report.json")
    write_json(report, out_path)
    print(report)
    if args.fail_on_event_leakage:
        failures = event_leakage_failures(report["leakage"])
        if failures:
            raise SystemExit("Event leakage hard gate failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
