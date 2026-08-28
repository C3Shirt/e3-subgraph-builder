from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from e3prep.io import read_parquet, write_json, write_parquet_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed balanced dataset from existing subgraph shards.")
    parser.add_argument("--source-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--samples-per-shard", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--positive-min-positive-nodes", type=int, default=1)
    parser.add_argument("--negative-max-positive-nodes", type=int, default=0)
    parser.add_argument("--write-sidecar", action="store_true")
    return parser.parse_args()


def read_source_metadata(source_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for source_dir in source_dirs:
        metadata_path = source_dir / "metadata.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        frame = read_parquet(metadata_path)
        frame["source_dir"] = str(source_dir)
        frames.append(frame)
    metadata = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if metadata.empty:
        raise ValueError("No source metadata rows found")
    metadata = metadata.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)
    return metadata


def select_balanced(
    metadata: pd.DataFrame,
    samples_per_class: int | None,
    seed: int,
    positive_min_positive_nodes: int,
    negative_max_positive_nodes: int,
) -> pd.DataFrame:
    positives = metadata[
        (metadata["label"].astype(int) == 1)
        & (metadata["positive_node_count"].fillna(0).astype(int) >= positive_min_positive_nodes)
    ]
    negatives = metadata[
        (metadata["label"].astype(int) == 0)
        & (metadata["positive_node_count"].fillna(0).astype(int) <= negative_max_positive_nodes)
    ]
    if positives.empty or negatives.empty:
        raise ValueError(f"Need both classes after filtering, got positives={len(positives)} negatives={len(negatives)}")

    target = min(len(positives), len(negatives))
    if samples_per_class is not None:
        target = min(target, samples_per_class)

    selected = pd.concat(
        [
            positives.sample(n=target, random_state=seed),
            negatives.sample(n=target, random_state=seed + 1),
        ],
        ignore_index=True,
    )
    selected = selected.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return selected


def iter_source_graphs(source_dir: Path):
    graph_dir = source_dir / "graphs"
    if not graph_dir.exists():
        raise FileNotFoundError(graph_dir)
    for shard in sorted(graph_dir.glob("shard_*.pt")):
        for graph in torch.load(shard, weights_only=False):
            yield graph


def write_graph_shards(selected: pd.DataFrame, out_dir: Path, samples_per_shard: int) -> int:
    out_graph_dir = out_dir / "graphs"
    out_graph_dir.mkdir(parents=True, exist_ok=True)
    selected_by_source: dict[str, set[str]] = {}
    metadata_by_sample = selected.set_index("sample_id").to_dict("index")
    for _, row in selected.iterrows():
        selected_by_source.setdefault(str(row["source_dir"]), set()).add(str(row["sample_id"]))

    shard_index = 0
    shard = []
    written_ids = set()

    def flush() -> None:
        nonlocal shard_index, shard
        if not shard:
            return
        torch.save(shard, out_graph_dir / f"shard_{shard_index:04d}.pt")
        shard = []
        shard_index += 1

    for source_dir, sample_ids in sorted(selected_by_source.items()):
        for graph in iter_source_graphs(Path(source_dir)):
            sample_id = str(graph.sample_id)
            if sample_id not in sample_ids:
                continue
            meta = metadata_by_sample[sample_id]
            graph.y = torch.tensor([int(meta["label"])], dtype=torch.long)
            graph.positive_node_count = int(meta.get("positive_node_count") or 0)
            shard.append(graph)
            written_ids.add(sample_id)
            if len(shard) >= samples_per_shard:
                flush()
    flush()

    missing = sorted(set(selected["sample_id"].astype(str)) - written_ids)
    if missing:
        raise ValueError(f"Selected graph IDs missing from shards: {missing[:10]}")
    return shard_index


def write_selected_sidecars(selected: pd.DataFrame, out_dir: Path) -> dict:
    sample_ids = set(selected["sample_id"].astype(str))
    node_frames = []
    edge_frames = []
    for source_dir in sorted(set(selected["source_dir"].astype(str))):
        nodes_path = Path(source_dir) / "nodes.parquet"
        edges_path = Path(source_dir) / "edges.parquet"
        if nodes_path.exists():
            nodes = read_parquet(nodes_path)
            node_frames.append(nodes[nodes["sample_id"].astype(str).isin(sample_ids)])
        if edges_path.exists():
            edges = read_parquet(edges_path)
            edge_frames.append(edges[edges["sample_id"].astype(str).isin(sample_ids)])

    result = {}
    if node_frames:
        nodes = pd.concat(node_frames, ignore_index=True)
        write_parquet_records(nodes.to_dict("records"), out_dir / "nodes.parquet", "sample_nodes")
        result["nodes"] = str(out_dir / "nodes.parquet")
    if edge_frames:
        edges = pd.concat(edge_frames, ignore_index=True)
        write_parquet_records(edges.to_dict("records"), out_dir / "edges.parquet", "sample_edges")
        result["edges"] = str(out_dir / "edges.parquet")
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    metadata = read_source_metadata(args.source_dir)
    selected = select_balanced(
        metadata,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
        positive_min_positive_nodes=args.positive_min_positive_nodes,
        negative_max_positive_nodes=args.negative_max_positive_nodes,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = write_graph_shards(selected, args.out_dir, args.samples_per_shard)

    metadata_rows = selected.drop(columns=["source_dir"]).to_dict("records")
    write_parquet_records(metadata_rows, args.out_dir / "metadata.parquet", "sample_metadata")
    sidecars = write_selected_sidecars(selected, args.out_dir) if args.write_sidecar else {}

    summary = {
        "source_dirs": [str(path) for path in args.source_dir],
        "out_dir": str(args.out_dir),
        "samples": int(len(selected)),
        "shards": shards,
        "samples_per_class": selected["label"].value_counts().sort_index().astype(int).to_dict(),
        "positive_min_positive_nodes": args.positive_min_positive_nodes,
        "negative_max_positive_nodes": args.negative_max_positive_nodes,
        "seed": args.seed,
        "metadata": str(args.out_dir / "metadata.parquet"),
        "graph_dir": str(args.out_dir / "graphs"),
        **sidecars,
    }
    write_json(summary, args.out_dir / "balanced_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
