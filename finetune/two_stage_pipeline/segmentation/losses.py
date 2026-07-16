"""Segmentation losses tuned for tiny, sparse MS lesions.

Soft Dice handles the extreme foreground/background imbalance, and Focal-Tversky
with ``alpha > beta`` penalises false negatives more than false positives, pushing
recall up — exactly what the end-to-end target needs. The default combo sums both.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 8.0
) -> torch.Tensor:
    """Per-pixel BCE with a positive-class weight.

    Gives every lesion pixel a strong, stable gradient regardless of batch
    composition, which stops the net from collapsing to an all-background mask.
    """
    weight = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (2, 3)
    numerator = 2.0 * (prob * target).sum(dims) + eps
    denominator = (prob + target).sum(dims) + eps
    return (1.0 - numerator / denominator).mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 0.75,
    eps: float = 1.0,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (2, 3)
    tp = (prob * target).sum(dims)
    fp = (prob * (1 - target)).sum(dims)
    fn = ((1 - prob) * target).sum(dims)
    tversky = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    return ((1.0 - tversky) ** gamma).mean()


def combo_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Dice + Tversky, recall-oriented (alpha>beta weights misses).

    A plain (gamma=1) Tversky is used instead of a Focal variant, and the mean-
    reduced BCE is dropped: on the many background pixels its false-positive
    penalty drives logits strongly negative, saturating the sigmoid and pushing
    the net into the collapsed all-empty basin. Dice + Tversky already make the
    empty mask a high-loss solution per image.
    """
    return dice_loss(logits, target) + focal_tversky_loss(
        logits, target, alpha=alpha, beta=beta, gamma=gamma
    )


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = (2, 3)
    numerator = 2.0 * (pred * target).sum(dims)
    denominator = (pred + target).sum(dims)
    # Slices with no lesion and no prediction score a perfect 1.
    return ((numerator + eps) / (denominator + eps)).mean()
