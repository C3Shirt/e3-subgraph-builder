from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one HeteroData sample from a shard.")
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--subgraph-dir", type=Path)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--top-n", type=int, default=10)
    return parser.parse_args()


def compact_value(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def inspect_sidecar(subgraph_dir: Path, sample_id: str | None, top_n: int) -> dict:
    metadata = pd.read_parquet(subgraph_dir / "metadata.parquet")
    nodes = pd.read_parquet(subgraph_dir / "nodes.parquet")
    edges = pd.read_parquet(subgraph_dir / "edges.parquet")
    if sample_id is None:
        sample_id = str(metadata.iloc[0]["sample_id"])
    meta = metadata[metadata["sample_id"].astype(str) == sample_id]
    if meta.empty:
        raise ValueError(f"Unknown sample_id in {subgraph_dir}: {sample_id}")
    sample_nodes = nodes[nodes["sample_id"].astype(str) == sample_id].copy()
    sample_edges = edges[edges["sample_id"].astype(str) == sample_id].copy()

    endpoint_counts = pd.concat(
        [
            sample_edges["flow_src_uuid"].dropna().astype(str),
            sample_edges["flow_dst_uuid"].dropna().astype(str),
        ],
        ignore_index=True,
    ).value_counts()
    node_lookup = {
        str(row.uuid): row
        for row in sample_nodes.itertuples(index=False)
    }
    top_nodes = []
    for uuid, degree in endpoint_counts.head(top_n).items():
        row = node_lookup.get(str(uuid))
        top_nodes.append(
            {
                "uuid": str(uuid),
                "degree": int(degree),
                "node_type": None if row is None else compact_value(row.node_type),
                "is_center": None if row is None else int(row.is_center),
                "is_labeled_positive": None if row is None else int(row.is_labeled_positive),
                "name": None if row is None else compact_value(row.name),
                "path": None if row is None else compact_value(row.path),
                "cmdline": None if row is None else compact_value(row.cmdline),
                "ip": None if row is None else compact_value(row.ip),
                "port": None if row is None else compact_value(row.port),
            }
        )

    positive_nodes = sample_nodes[sample_nodes["is_labeled_positive"] == 1]
    return {
        "metadata": meta.iloc[0].to_dict(),
        "node_type_counts": sample_nodes["node_type"].value_counts().sort_index().astype(int).to_dict(),
        "event_type_counts": sample_edges["event_type"].value_counts().sort_index().astype(int).to_dict(),
        "direction_counts": sample_edges["direction"].value_counts().sort_index().astype(int).to_dict(),
        "top_degree_nodes": top_nodes,
        "positive_nodes": positive_nodes.head(top_n).to_dict("records"),
    }


def main() -> None:
    args = parse_args()
    if args.subgraph_dir is not None:
        print(json.dumps(inspect_sidecar(args.subgraph_dir, args.sample_id, args.top_n), indent=2, default=str))
        return
    if args.shard is None:
        raise ValueError("Pass either --shard or --subgraph-dir")
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
