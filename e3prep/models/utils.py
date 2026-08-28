from __future__ import annotations

import torch


def center_process_embeddings(batch, process_x: torch.Tensor) -> torch.Tensor:
    if "process" not in batch.node_types:
        raise ValueError("Batch does not contain process nodes")

    center_mask = batch["process"].x[:, 0] > 0.5
    centers = process_x[center_mask]
    expected = int(batch.y.view(-1).numel())
    if centers.size(0) == expected:
        return centers

    if hasattr(batch["process"], "ptr") and hasattr(batch, "center_index"):
        ptr = batch["process"].ptr.to(process_x.device)
        center_index = batch.center_index.to(process_x.device)
        absolute_index = ptr[:-1] + center_index
        return process_x[absolute_index]

    raise ValueError(f"Expected {expected} center process nodes, found {centers.size(0)}")


def binary_classification_metrics(y_true: torch.Tensor, logits: torch.Tensor) -> dict:
    y_true = y_true.detach().cpu().view(-1).long()
    logits = logits.detach().cpu()
    pred = logits.argmax(dim=-1).view(-1).long()
    score = logits.softmax(dim=-1)[:, 1]
    metrics = binary_metrics_from_predictions(y_true, pred, score)
    metrics["threshold"] = None
    return metrics


def binary_metrics_at_threshold(y_true: torch.Tensor, score: torch.Tensor, threshold: float) -> dict:
    y_true = y_true.detach().cpu().view(-1).long()
    score = score.detach().cpu().view(-1)
    pred = (score >= threshold).long()
    metrics = binary_metrics_from_predictions(y_true, pred, score)
    metrics["threshold"] = float(threshold)
    return metrics


def binary_metrics_from_predictions(y_true: torch.Tensor, pred: torch.Tensor, score: torch.Tensor) -> dict:
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    total = int(y_true.numel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": binary_auc(y_true, score),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def best_f1_threshold(y_true: torch.Tensor, score: torch.Tensor) -> tuple[float | None, dict | None]:
    y_true = y_true.detach().cpu().view(-1).long()
    score = score.detach().cpu().view(-1)
    if y_true.numel() == 0 or (y_true == 0).sum() == 0 or (y_true == 1).sum() == 0:
        return None, None
    thresholds = sorted(set(float(value) for value in score.tolist()))
    candidates = [0.5]
    candidates.extend(thresholds)
    best_threshold = None
    best_metrics = None
    for threshold in candidates:
        metrics = binary_metrics_at_threshold(y_true, score, threshold)
        if best_metrics is None or (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            metrics["accuracy"],
        ) > (
            best_metrics["f1"],
            best_metrics["recall"],
            best_metrics["precision"],
            best_metrics["accuracy"],
        ):
            best_threshold = threshold
            best_metrics = metrics
    return best_threshold, best_metrics


def binary_auc(y_true: torch.Tensor, score: torch.Tensor) -> float | None:
    positives = score[y_true == 1]
    negatives = score[y_true == 0]
    if positives.numel() == 0 or negatives.numel() == 0:
        return None
    comparisons = (positives[:, None] > negatives[None, :]).float()
    ties = (positives[:, None] == negatives[None, :]).float() * 0.5
    return float((comparisons + ties).mean().item())
