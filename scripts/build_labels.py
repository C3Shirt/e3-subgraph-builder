from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from e3prep.io import write_parquet_records
from e3prep.labels.threatrace import read_threatrace_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a label store from ThreaTrace-style UUID files.")
    parser.add_argument("--dataset", default="cadets", choices=["cadets", "theia", "trace"])
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--label-source", default="threatrace")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = read_threatrace_labels(args.groundtruth, args.dataset, args.label_source)
    count = write_parquet_records(
        (label.to_dict() for label in labels),
        args.out_dir / "labels.parquet",
        "labels",
    )
    print({"labels": count, "out": str(args.out_dir / "labels.parquet")})


if __name__ == "__main__":
    main()

