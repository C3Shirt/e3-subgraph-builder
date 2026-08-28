from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from e3prep.models import GINClassifier, GraphListDataset, HGTClassifier, load_graphs_from_dir, split_graphs_stratified
from e3prep.models.dataset import class_counts, limit_per_class, metadata_from_graphs
from e3prep.models.utils import binary_classification_metrics, binary_metrics_at_threshold, best_f1_threshold
from e3prep.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight HGT/GIN sanity baseline on E3 subgraph shards.")
    parser.add_argument("--model", required=True, choices=["hgt", "gin"])
    parser.add_argument("--data-dir", type=Path, default=None, help="Single subgraph dir to split internally.")
    parser.add_argument("--train-dir", action="append", type=Path, default=[])
    parser.add_argument("--val-dir", action="append", type=Path, default=[])
    parser.add_argument("--test-dir", action="append", type=Path, default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
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


def load_splits(args: argparse.Namespace) -> tuple[list, list, list]:
    if args.data_dir:
        graphs = load_graphs_from_dir(args.data_dir, max_samples=args.max_samples)
        graphs = limit_per_class(graphs, args.max_per_class, args.seed)
        return split_graphs_stratified(graphs, args.val_fraction, args.test_fraction, args.seed)

    if not args.train_dir:
        raise ValueError("Provide --data-dir or --train-dir")
    train = load_graphs_from_dirs(args.train_dir, max_samples=args.max_samples)
    train = limit_per_class(train, args.max_per_class, args.seed)
    val = load_graphs_from_dirs(args.val_dir) if args.val_dir else []
    test = load_graphs_from_dirs(args.test_dir) if args.test_dir else []
    if not val and args.val_fraction > 0:
        train, val, held_out = split_graphs_stratified(train, args.val_fraction, 0.0, args.seed)
        if not test:
            test = held_out
    return train, val, test


def load_graphs_from_dirs(paths: list[Path], max_samples: int | None = None) -> list:
    graphs = []
    for path in paths:
        remaining = None if max_samples is None else max_samples - len(graphs)
        if remaining is not None and remaining <= 0:
            break
        graphs.extend(load_graphs_from_dir(path, max_samples=remaining))
    return graphs


def class_weight_tensor(labels: list[int], device: torch.device) -> torch.Tensor | None:
    counts = {0: labels.count(0), 1: labels.count(1)}
    if counts[0] == 0 or counts[1] == 0:
        return None
    total = counts[0] + counts[1]
    return torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float, device=device)


def make_model(args: argparse.Namespace, graphs: list, device: torch.device) -> nn.Module:
    if args.model == "gin":
        return GINClassifier(
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

    if args.hidden_channels % args.heads != 0:
        raise ValueError("--hidden-channels must be divisible by --heads for HGT")
    metadata = metadata_from_graphs(graphs)
    return HGTClassifier(
        metadata=metadata,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer, device: torch.device) -> dict:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        target = batch.y.view(-1).long()
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(target.numel())
        total_items += int(target.numel())
    return {"loss": total_loss / total_items if total_items else 0.0}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[dict, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    logits_parts = []
    target_parts = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        target = batch.y.view(-1).long()
        loss = criterion(logits, target)
        total_loss += float(loss.item()) * int(target.numel())
        logits_parts.append(logits.detach().cpu())
        target_parts.append(target.detach().cpu())
    if not target_parts:
        empty = {"loss": None, "accuracy": None, "precision": None, "recall": None, "f1": None, "auc": None}
        return empty, torch.empty(0), torch.empty(0)
    logits_all = torch.cat(logits_parts, dim=0)
    target_all = torch.cat(target_parts, dim=0)
    metrics = binary_classification_metrics(target_all, logits_all)
    metrics["loss"] = total_loss / int(target_all.numel())
    return metrics, target_all, logits_all


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    metrics, _target, _logits = predict(model, loader, criterion, device)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_graphs, val_graphs, test_graphs = load_splits(args)
    if not train_graphs:
        raise ValueError("No training graphs loaded")

    all_graphs = train_graphs + val_graphs + test_graphs
    model = make_model(args, all_graphs, device)
    train_dataset = GraphListDataset(train_graphs)
    val_dataset = GraphListDataset(val_graphs)
    test_dataset = GraphListDataset(test_graphs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    weights = class_weight_tensor(train_dataset.labels, device) if args.class_weight == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_f1 = -1.0
    best_path = args.out_dir / f"best_{args.model}.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        }
        history.append(row)
        val_f1 = val_metrics.get("f1")
        if val_f1 is not None and val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            torch.save(model.state_dict(), best_path)
        print(json.dumps(row, sort_keys=True), flush=True)

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    final_val_argmax, val_y, val_logits = predict(model, val_loader, criterion, device)
    final_test_argmax, test_y, test_logits = predict(model, test_loader, criterion, device)
    threshold, threshold_val_metrics = best_f1_threshold(val_y, val_logits.softmax(dim=-1)[:, 1]) if val_y.numel() else (None, None)
    threshold_test_metrics = (
        binary_metrics_at_threshold(test_y, test_logits.softmax(dim=-1)[:, 1], threshold)
        if threshold is not None and test_y.numel()
        else None
    )

    final = {
        "model": args.model,
        "device": str(device),
        "train_dir": [str(path) for path in args.train_dir],
        "val_dir": [str(path) for path in args.val_dir],
        "test_dir": [str(path) for path in args.test_dir],
        "data_dir": str(args.data_dir) if args.data_dir else None,
        "counts": {
            "train": class_counts(train_graphs),
            "val": class_counts(val_graphs),
            "test": class_counts(test_graphs),
        },
        "best_val_f1": best_val_f1 if best_val_f1 >= 0 else None,
        "final_train": evaluate(model, train_loader, criterion, device),
        "final_val": final_val_argmax,
        "final_test": final_test_argmax,
        "val_selected_threshold": threshold,
        "final_val_at_selected_threshold": threshold_val_metrics,
        "final_test_at_selected_threshold": threshold_test_metrics,
        "history": history,
    }
    write_json(final, args.out_dir / "metrics.json")
    print(json.dumps({"metrics": str(args.out_dir / "metrics.json"), "best_model": str(best_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
