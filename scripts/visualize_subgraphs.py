from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from e3prep.io import read_parquet, write_json


NODE_COLORS = {
    "PROCESS": "#4C78A8",
    "FILE": "#F58518",
    "SOCKET": "#54A24B",
    "MEMORY": "#B279A2",
    "PIPE": "#72B7B2",
    "PRINCIPAL": "#E45756",
    "UNKNOWN": "#8C8C8C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw audit figures from sample sidecar Parquet files.")
    parser.add_argument("--subgraph-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--positive", type=int, default=20)
    parser.add_argument("--negative", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-label-len", type=int, default=34)
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:120]


def compact_text(value, limit: int) -> str:
    if pd.isna(value) or value is None:
        return ""
    text = str(value).replace("\\", "/")
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3) :]


def node_label(row: pd.Series, limit: int) -> str:
    primary = compact_text(row.get("path"), limit) or compact_text(row.get("name"), limit)
    if not primary:
        ip = compact_text(row.get("ip"), limit)
        port = row.get("port")
        if ip and not pd.isna(port):
            primary = f"{ip}:{int(port)}"
        else:
            primary = ip
    if not primary:
        primary = str(row["uuid"])[:8]
    prefix = "*" if int(row.get("is_center", 0)) else ""
    return f"{prefix}{row['node_type']}\\n{primary}"


def select_samples(metadata: pd.DataFrame, positive: int, negative: int, seed: int) -> pd.DataFrame:
    parts = []
    if positive > 0:
        pos = metadata[metadata["label"] == 1]
        if not pos.empty:
            parts.append(pos.sample(n=min(positive, len(pos)), random_state=seed))
    if negative > 0:
        neg = metadata[metadata["label"] == 0]
        if not neg.empty:
            parts.append(neg.sample(n=min(negative, len(neg)), random_state=seed))
    if not parts:
        return metadata.head(0)
    return pd.concat(parts, ignore_index=True)


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for _, row in nodes.iterrows():
        graph.add_node(
            row["uuid"],
            node_type=row.get("node_type", "UNKNOWN"),
            is_center=int(row.get("is_center", 0)),
            is_labeled_positive=int(row.get("is_labeled_positive", 0)),
            row=row,
        )
    for _, row in edges.iterrows():
        src = row.get("flow_src_uuid")
        dst = row.get("flow_dst_uuid")
        if pd.isna(src) or pd.isna(dst):
            continue
        graph.add_edge(
            src,
            dst,
            key=row.get("event_edge_id") or f"{row.get('event_uuid')}:{row.get('object_role')}",
            event_type=row.get("event_type", "EVENT_UNKNOWN"),
            direction=row.get("direction"),
        )
    return graph


def draw_sample(
    sample_id: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    out_path: Path,
    max_label_len: int,
) -> None:
    graph = build_graph(nodes, edges)
    if graph.number_of_nodes() == 0:
        return

    width = max(8, min(18, 3 + graph.number_of_nodes() * 0.45))
    height = max(6, min(14, 3 + graph.number_of_nodes() * 0.35))
    fig, ax = plt.subplots(figsize=(width, height))
    pos = nx.spring_layout(graph, seed=11, k=1.2)

    colors = []
    sizes = []
    line_widths = []
    for node in graph.nodes:
        attrs = graph.nodes[node]
        node_type = str(attrs.get("node_type") or "UNKNOWN").upper()
        colors.append(NODE_COLORS.get(node_type, NODE_COLORS["UNKNOWN"]))
        sizes.append(1300 if attrs.get("is_center") else 850)
        line_widths.append(3.0 if attrs.get("is_labeled_positive") else 1.4)

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=colors,
        node_size=sizes,
        linewidths=line_widths,
        edgecolors="#222222",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        edge_color="#555555",
        width=1.2,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    labels = {node: node_label(graph.nodes[node]["row"], max_label_len) for node in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7, font_color="#111111", ax=ax)

    edge_labels = {}
    for src, dst, _key, attrs in graph.edges(keys=True, data=True):
        label = str(attrs.get("event_type") or "EVENT_UNKNOWN").removeprefix("EVENT_")
        edge_labels[(src, dst)] = label
    if graph.number_of_edges() <= 30:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=6, ax=ax)

    ax.set_title(sample_id, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metadata_path = args.subgraph_dir / "metadata.parquet"
    nodes_path = args.subgraph_dir / "nodes.parquet"
    edges_path = args.subgraph_dir / "edges.parquet"
    if not metadata_path.exists() or not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            "Visualization requires metadata.parquet, nodes.parquet, and edges.parquet. "
            "Rebuild subgraphs with --write-sidecar."
        )

    metadata = read_parquet(metadata_path)
    nodes = read_parquet(nodes_path)
    edges = read_parquet(edges_path)
    selected = select_samples(metadata, args.positive, args.negative, args.seed)

    out_dir = args.out_dir or (args.subgraph_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for _, sample in selected.iterrows():
        sample_id = str(sample["sample_id"])
        sample_nodes = nodes[nodes["sample_id"] == sample_id]
        sample_edges = edges[edges["sample_id"] == sample_id]
        label_name = "malicious" if int(sample.get("label", 0)) == 1 else "benign"
        out_path = out_dir / f"{label_name}_{safe_name(sample_id)}.png"
        draw_sample(sample_id, sample_nodes, sample_edges, out_path, args.max_label_len)
        manifest.append(
            {
                "sample_id": sample_id,
                "label": int(sample.get("label", 0)),
                "nodes": int(len(sample_nodes)),
                "edges": int(len(sample_edges)),
                "path": str(out_path),
            }
        )
    write_json({"figures": manifest}, out_dir / "figures_manifest.json")
    print({"figures": len(manifest), "out_dir": str(out_dir)})


if __name__ == "__main__":
    main()
