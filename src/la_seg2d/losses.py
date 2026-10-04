"""CE + soft Dice. Same weights, smooth term, and batch-mean reduction as the 3D loss.

The 3D implementation reduces Dice over (D, H, W). This one reduces over (H, W).
Per-sample Dice values are averaged over the batch. CE is the default mean
CrossEntropyLoss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_CROSS_ENTROPY = nn.CrossEntropyLoss()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float) -> torch.Tensor:
    """Per-sample soft Dice on the foreground class, then the batch mean.

    `logits` is [B, 2, H, W]. `target` is integer [B, H, W].
    Dice uses softmax probabilities, not argmax labels.
    """
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError(f"expected logits [B,2,H,W], got {tuple(logits.shape)}")
    probs = torch.softmax(logits, dim=1)[:, 1]
    gt = (target == 1).to(dtype=probs.dtype)
    dims = (1, 2)
    intersection = (probs * gt).sum(dim=dims)
    denom = probs.sum(dim=dims) + gt.sum(dim=dims)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def ce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ce_weight: float,
    dice_weight: float,
    dice_smooth: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = _CROSS_ENTROPY(logits, target.long())
    dice = soft_dice_loss(logits, target, smooth=dice_smooth)
    total = ce_weight * ce + dice_weight * dice
    return total, ce, dice
