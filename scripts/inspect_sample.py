from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one HeteroData sample from a shard.")
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = torch.load(args.shard, map_location="cpu", weights_only=False)
    sample = samples[args.index]
    print(
        {
            "sample_id": getattr(sample, "sample_id", None),
            "y": sample.y.tolist() if hasattr(sample, "y") else None,
            "node_types": sample.node_types,
            "edge_types": sample.edge_types,
            "center_type": getattr(sample, "center_type", None),
            "center_index": getattr(sample, "center_index", None),
        }
    )


if __name__ == "__main__":
    main()

