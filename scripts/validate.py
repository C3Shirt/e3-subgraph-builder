from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e3prep.io import write_json
from e3prep.validation.coverage import label_coverage_report
from e3prep.validation.leakage import leakage_report
from e3prep.validation.relations import relation_audit_report
from e3prep.validation.statistics import build_dataset_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an E3 event store and optional subgraph directory.")
    parser.add_argument("--store-dir", required=True, type=Path)
    parser.add_argument("--subgraph-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--relation-top-n", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dataset_report(args.store_dir, args.subgraph_dir)
    report["label_coverage"] = label_coverage_report(args.store_dir, args.subgraph_dir)
    report["relation_audit"] = relation_audit_report(args.store_dir, args.relation_top_n)
    report["leakage"] = leakage_report(args.store_dir, args.subgraph_dir)
    out_path = args.out or (args.store_dir / "dataset_report.json")
    write_json(report, out_path)
    print(report)


if __name__ == "__main__":
    main()
