"""Losses for the ISIC baseline and the two imbalance strategies.

All three losses take raw logits and integer class indices in ``[0, C)``.
Class weights are computed only from the training split.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from isic_baseline import CLASS_NAMES

LOSS_NAMES = ("cross_entropy", "weighted_cross_entropy", "focal")


def training_class_counts(train_records: list[dict], class_names: list[str] | None = None) -> list[int]:
    names = list(class_names or CLASS_NAMES)
    counts = [0 for _ in names]
    for record in train_records:
        label = int(record["label"])
        if label < 0 or label >= len(names):
            raise RuntimeError(f"Training label {label} is outside 0..{len(names) - 1}.")
        counts[label] += 1
    missing = [name for name, count in zip(names, counts) if count == 0]
    if missing:
        raise RuntimeError(
            "Training set has no samples for "
            + ", ".join(missing)
            + ". Inverse-frequency weights are undefined; refusing to continue."
        )
    return counts


def inverse_frequency_weights(counts: list[int], beta: float = 0.5) -> list[float]:
    """Return ``w_c = n_c ** (-beta)`` from training-set counts.

    ``beta = 1`` is plain inverse frequency. ``beta = 0.5`` uses the inverse square
    root, so rare classes are upweighted less sharply. These values are not rescaled
    here; each batch divides them by their own mean.
    """
    if any(count <= 0 for count in counts):
        raise RuntimeError(f"Class counts must be positive, got {counts}.")
    if beta < 0:
        raise ValueError(f"weighted_cross_entropy beta must be >= 0, got {beta}.")
    raw = [count ** (-float(beta)) for count in counts]
    print(f"Inverse frequency weights: raw={raw}")
    return raw


def prepare_loss_spec(cfg: dict, train_records: list[dict]) -> dict:
    """Build the loss record stored in the run config. Weights use training labels only."""
    raw_spec = cfg.get("loss") or {"name": "cross_entropy"}
    if isinstance(raw_spec, str):
        raw_spec = {"name": raw_spec}
    name = raw_spec.get("name", "cross_entropy")
    if name not in LOSS_NAMES:
        raise ValueError(f"loss.name must be one of {LOSS_NAMES}, got {name!r}.")
    if name == "cross_entropy":
        return {
            "name": "cross_entropy",
            "formula": "mean CrossEntropyLoss on logits; no class weights",
            "class_order": list(CLASS_NAMES),
            "class_weights": None,
            "alpha": None,
        }
    if name == "weighted_cross_entropy":
        counts = training_class_counts(train_records, CLASS_NAMES)
        beta = float(raw_spec.get("beta", 0.5))
        weights = inverse_frequency_weights(counts, beta)
        return {
            "name": "weighted_cross_entropy",
            "formula": "w_c = n_c^(-beta) from the training split; each batch uses mean((w_y / mean(w_y)) * cross_entropy)",
            "beta": beta,
            "class_order": list(CLASS_NAMES),
            "class_counts": counts,
            "class_weights": weights,
            "weight_source": "training split true labels",
            "alpha": None,
        }
    gamma = float(raw_spec.get("gamma", 2.0))
    alpha = raw_spec.get("alpha", None)
    if alpha is not None:
        alpha = [float(value) for value in alpha]
        if len(alpha) != len(CLASS_NAMES):
            raise ValueError(f"focal alpha must have length {len(CLASS_NAMES)}, got {len(alpha)}.")
    return {
        "name": "focal",
        "formula": "mean over samples of -(1 - p_t)^gamma * log(p_t); p_t = exp(log_softmax(logits)[y]); alpha is not the inverse-frequency weight",
        "gamma": gamma,
        "alpha": alpha,
        "class_order": list(CLASS_NAMES),
        "class_weights": None,
    }


class FocalLoss(nn.Module):
    """Single-label multi-class focal loss on logits.

    ``gamma == 0`` and ``alpha is None`` matches ``CrossEntropyLoss`` with mean reduction.
    """

    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.long()
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.gamma == 0.0:
            loss = -log_pt
        else:
            pt = log_pt.exp()
            loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


class WeightedCrossEntropy(nn.Module):
    """Inverse-frequency cross-entropy with the weight mean taken inside the batch."""

    def __init__(self, class_weights: list[float]) -> None:
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_sample = F.cross_entropy(logits, target, reduction="none")
        weights = self.class_weights[target.long()]
        expectation = weights.mean()
        normalized = weights / expectation
        return (per_sample * normalized).mean()


def build_criterion(spec: dict, device: torch.device) -> nn.Module:
    name = spec.get("name", "cross_entropy")
    if name == "cross_entropy":
        return nn.CrossEntropyLoss().to(device)
    if name == "weighted_cross_entropy":
        weights = spec.get("class_weights")
        if not weights or len(weights) != len(CLASS_NAMES):
            raise RuntimeError("weighted_cross_entropy requires the saved 7-class training-set weights.")
        return WeightedCrossEntropy(weights).to(device)
    if name == "focal":
        return FocalLoss(gamma=float(spec.get("gamma", 2.0)), alpha=spec.get("alpha")).to(device)
    raise ValueError(f"Unknown loss {name!r}.")


def format_loss_spec(spec: dict) -> str:
    lines = [f"loss.name={spec['name']}", f"formula: {spec['formula']}"]
    if spec["name"] == "weighted_cross_entropy":
        lines.append(f"beta={spec['beta']}")
        lines.append("class_order=" + ",".join(spec["class_order"]))
        lines.append("class_counts=" + ",".join(str(value) for value in spec["class_counts"]))
        lines.append("class_weights=" + ",".join(f"{value:.6f}" for value in spec["class_weights"]))
    if spec["name"] == "focal":
        lines.append(f"gamma={spec['gamma']}")
        lines.append(f"alpha={spec['alpha']}")
    return "\n".join(lines)
