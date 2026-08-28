from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch_geometric.loader import DataLoader

from e3prep.io import write_json
from e3prep.models import GraphListDataset, load_graphs_from_dir, make_edge_type_predictor
from e3prep.models.dataset import class_counts, metadata_from_graphs
from e3prep.models.edge_prediction import edge_prediction_loss, graph_scores_from_edge_losses
from e3prep.models.utils import binary_metrics_at_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train unsupervised E3 baselines. The training loss never uses attack labels; "
            "labels are used only for optional train filtering and final evaluation."
        )
    )
    parser.add_argument("--model", required=True, choices=["hgt", "gin"])
    parser.add_argument("--method", default="both", choices=["edge-nll", "one-class-svm", "both"])
    parser.add_argument("--train-dir", action="append", type=Path, default=[], required=True)
    parser.add_argument("--val-dir", action="append", type=Path, default=[])
    parser.add_argument("--test-dir", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-labels", default=None, help="Optional comma-separated labels kept for one-class training, e.g. 0.")
    parser.add_argument(
        "--require-train-positive-node-count",
        default="any",
        choices=["any", "zero", "nonzero"],
        help="Optional training filter. 'zero' is useful for clean-normal one-class runs.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help="Expected anomaly fraction. Used for edge-NLL train-quantile threshold and default SVM nu.",
    )
    parser.add_argument("--svm-nu", type=float, default=None)
    parser.add_argument("--svm-kernel", default="rbf")
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def parse_label_filter(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def load_graphs_from_dirs(
    paths: Iterable[Path],
    max_samples: int | None = None,
    labels: set[int] | None = None,
    require_positive_node_count: str = "any",
) -> list:
    graphs = []
    for path in paths:
        remaining = None if max_samples is None else max_samples - len(graphs)
        if remaining is not None and remaining <= 0:
            break
        graphs.extend(
            load_graphs_from_dir(
                path,
                max_samples=remaining,
                labels=labels,
                require_positive_node_count=require_positive_node_count,
            )
        )
    return graphs


def sample_ids_from_batch(batch) -> list[str]:
    sample_id = getattr(batch, "sample_id", None)
    if isinstance(sample_id, (list, tuple)):
        return [str(value) for value in sample_id]
    if sample_id is None:
        return [f"graph_{idx}" for idx in range(int(batch.num_graphs))]
    if int(batch.num_graphs) == 1:
        return [str(sample_id)]
    return [str(value) for value in sample_id]


def train_epoch(model, loader: DataLoader, optimizer, device: torch.device) -> dict:
    model.train()
    total_loss = 0.0
    total_edges = 0
    skipped_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, target, _graph_ids = model(batch)
        if target.numel() == 0:
            skipped_batches += 1
            continue
        loss = edge_prediction_loss(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(target.numel())
        total_edges += int(target.numel())
    return {
        "edge_ce": total_loss / total_edges if total_edges else None,
        "edges": total_edges,
        "skipped_batches": skipped_batches,
    }


@torch.no_grad()
def score_edge_nll(model, loader: DataLoader, device: torch.device, split: str) -> tuple[list[dict], dict]:
    model.eval()
    rows: list[dict] = []
    total_loss = 0.0
    total_edges = 0
    for batch in loader:
        batch = batch.to(device)
        logits, target, graph_ids = model(batch)
        sample_ids = sample_ids_from_batch(batch)
        labels = batch.y.view(-1).detach().cpu().long()
        if target.numel():
            losses = F.cross_entropy(logits, target, reduction="none")
            scores, edge_counts = graph_scores_from_edge_losses(losses, graph_ids, int(batch.num_graphs))
            total_loss += float(losses.sum().item())
            total_edges += int(target.numel())
        else:
            scores = torch.zeros(int(batch.num_graphs), device=device)
            edge_counts = torch.zeros(int(batch.num_graphs), dtype=torch.long, device=device)
        for idx, sample_id in enumerate(sample_ids):
            rows.append(
                {
                    "split": split,
                    "sample_id": sample_id,
                    "label": int(labels[idx].item()),
                    "edge_nll": float(scores[idx].detach().cpu().item()),
                    "edge_count": int(edge_counts[idx].detach().cpu().item()),
                }
            )
    summary = {"edge_ce": total_loss / total_edges if total_edges else None, "edges": total_edges}
    return rows, summary


@torch.no_grad()
def collect_center_embeddings(model, loader: DataLoader, device: torch.device, split: str) -> tuple[list[dict], torch.Tensor]:
    model.eval()
    rows: list[dict] = []
    embeddings = []
    for batch in loader:
        batch = batch.to(device)
        centers = model.encode_centers(batch).detach().cpu()
        labels = batch.y.view(-1).detach().cpu().long()
        sample_ids = sample_ids_from_batch(batch)
        embeddings.append(centers)
        for idx, sample_id in enumerate(sample_ids):
            rows.append({"split": split, "sample_id": sample_id, "label": int(labels[idx].item())})
    if not embeddings:
        return rows, torch.empty((0, 0))
    return rows, torch.cat(embeddings, dim=0)


def quantile_threshold(rows: list[dict], score_key: str, contamination: float) -> float | None:
    scores = torch.tensor([float(row[score_key]) for row in rows], dtype=torch.float)
    if scores.numel() == 0:
        return None
    return float(torch.quantile(scores, 1.0 - contamination).item())


def metrics_from_rows(rows: list[dict], score_key: str, threshold: float | None) -> dict | None:
    if threshold is None or not rows:
        return None
    y = torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)
    scores = torch.tensor([float(row[score_key]) for row in rows], dtype=torch.float)
    return binary_metrics_at_threshold(y, scores, threshold)


def write_score_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "sample_id", "label", "edge_nll", "edge_count", "svm_score"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def add_svm_scores(model, loaders: dict[str, DataLoader], rows_by_split: dict[str, list[dict]], args, device: torch.device) -> dict:
    train_meta, train_embeddings = collect_center_embeddings(model, loaders["train"], device, "train")
    if train_embeddings.numel() == 0:
        raise ValueError("No train embeddings available for OneClassSVM")
    svm = make_pipeline(
        StandardScaler(),
        OneClassSVM(
            kernel=args.svm_kernel,
            gamma=args.svm_gamma,
            nu=args.svm_nu if args.svm_nu is not None else args.contamination,
        ),
    )
    svm.fit(train_embeddings.numpy())

    metrics = {
        "nu": args.svm_nu if args.svm_nu is not None else args.contamination,
        "kernel": args.svm_kernel,
        "gamma": args.svm_gamma,
        "threshold": 0.0,
        "threshold_source": "OneClassSVM decision boundary; anomaly_score=-decision_function",
    }
    for split, loader in loaders.items():
        meta, embeddings = (train_meta, train_embeddings) if split == "train" else collect_center_embeddings(model, loader, device, split)
        if embeddings.numel() == 0:
            split_scores = []
        else:
            split_scores = (-svm.decision_function(embeddings.numpy())).tolist()
        keyed_rows = {row["sample_id"]: row for row in rows_by_split[split]}
        for meta_row, score in zip(meta, split_scores):
            keyed_rows[meta_row["sample_id"]]["svm_score"] = float(score)
        metrics[split] = metrics_from_rows(rows_by_split[split], "svm_score", 0.0)
    return metrics


def main() -> None:
    args = parse_args()
    if not 0.0 < args.contamination <= 0.5:
        raise ValueError("--contamination must be in (0, 0.5]")
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_label_filter = parse_label_filter(args.train_labels)
    train_graphs = load_graphs_from_dirs(
        args.train_dir,
        max_samples=args.max_train_samples,
        labels=train_label_filter,
        require_positive_node_count=args.require_train_positive_node_count,
    )
    val_graphs = load_graphs_from_dirs(args.val_dir, max_samples=args.max_val_samples) if args.val_dir else []
    test_graphs = load_graphs_from_dirs(args.test_dir, max_samples=args.max_test_samples) if args.test_dir else []
    if not train_graphs:
        raise ValueError("No training graphs loaded")

    all_graphs = train_graphs + val_graphs + test_graphs
    metadata = metadata_from_graphs(all_graphs)
    model = make_edge_type_predictor(
        args.model,
        metadata,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loaders = {
        "train": DataLoader(GraphListDataset(train_graphs), batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(GraphListDataset(val_graphs), batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(GraphListDataset(test_graphs), batch_size=args.batch_size, shuffle=False),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_train_loss = None
    best_path = args.out_dir / f"best_{args.model}_edge_predictor.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, loaders["train"], optimizer, device)
        row = {"epoch": epoch, "train": train_metrics}
        history.append(row)
        loss_value = train_metrics["edge_ce"]
        if loss_value is not None and (best_train_loss is None or loss_value < best_train_loss):
            best_train_loss = float(loss_value)
            torch.save(model.state_dict(), best_path)
        print(json.dumps(row, sort_keys=True), flush=True)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    elif args.epochs == 0:
        torch.save(model.state_dict(), best_path)

    rows_by_split: dict[str, list[dict]] = {}
    eval_summary = {}
    for split, loader in loaders.items():
        rows, split_summary = score_edge_nll(model, loader, device, split)
        rows_by_split[split] = rows
        eval_summary[split] = split_summary

    edge_threshold = quantile_threshold(rows_by_split["train"], "edge_nll", args.contamination)
    edge_nll_metrics = {
        "threshold": edge_threshold,
        "threshold_source": "train edge-NLL quantile",
        "contamination": args.contamination,
        "edge_prediction": eval_summary,
        "train": metrics_from_rows(rows_by_split["train"], "edge_nll", edge_threshold),
        "val": metrics_from_rows(rows_by_split["val"], "edge_nll", edge_threshold),
        "test": metrics_from_rows(rows_by_split["test"], "edge_nll", edge_threshold),
    }

    for rows in rows_by_split.values():
        for row in rows:
            row.setdefault("svm_score", None)

    svm_metrics = None
    if args.method in ("one-class-svm", "both"):
        svm_metrics = add_svm_scores(model, loaders, rows_by_split, args, device)

    all_rows = rows_by_split["train"] + rows_by_split["val"] + rows_by_split["test"]
    write_score_csv(all_rows, args.out_dir / "scores.csv")

    final = {
        "protocol": "unsupervised_anomaly_detection",
        "uses_attack_labels_in_loss": False,
        "note": (
            "Attack labels are not passed to the edge-prediction loss. "
            "--train-labels, if set, only filters the training population for a one-class clean-normal protocol."
        ),
        "model": args.model,
        "method": args.method,
        "device": str(device),
        "train_dir": [str(path) for path in args.train_dir],
        "val_dir": [str(path) for path in args.val_dir],
        "test_dir": [str(path) for path in args.test_dir],
        "train_label_filter": sorted(train_label_filter) if train_label_filter is not None else None,
        "require_train_positive_node_count": args.require_train_positive_node_count,
        "counts": {
            "train": class_counts(train_graphs),
            "val": class_counts(val_graphs),
            "test": class_counts(test_graphs),
        },
        "metadata": {
            "node_types": metadata[0],
            "edge_types": [list(edge_type) for edge_type in metadata[1]],
        },
        "history": history,
        "edge_nll": edge_nll_metrics if args.method in ("edge-nll", "both") else None,
        "one_class_svm": svm_metrics,
        "scores": str(args.out_dir / "scores.csv"),
        "best_model": str(best_path),
    }
    write_json(final, args.out_dir / "metrics.json")
    print(json.dumps({"metrics": str(args.out_dir / "metrics.json"), "scores": str(args.out_dir / "scores.csv")}, sort_keys=True))


if __name__ == "__main__":
    main()
