"""Various contrastive loss definitions"""

from collections.abc import Callable
from typing import Literal

import torch
from torch.nn.functional import cross_entropy, normalize

from shimmer.modules.domain import LossOutput

ContrastiveLossType = Callable[[torch.Tensor, torch.Tensor], LossOutput]
"""
Contrastive loss function type.

A function taking the prediction and targets and returning a LossOutput.
"""

ContrastiveLossWithConfidenceType = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], LossOutput
]
"""
Contrastive loss function type for GlobalWorkspaceWithConfidence.

A function taking the prediction mean, prediction std, target mean and target std and
    returns a LossOutput.
"""


def info_nce(
    x: torch.Tensor,
    y: torch.Tensor,
    logit_scale: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    InfoNCE loss

    Args:
        x (`torch.Tensor`): prediction
        y (`torch.Tensor`): target
        logit_scale (`torch.Tensor`): logit scale
        reduction (`Literal["mean", "sum", "none"]`): reduction to apply

    Returns: the InfoNCE loss
    """
    xn = normalize(x)
    yn = normalize(y)
    logits = torch.clamp(logit_scale.exp(), max=100) * xn @ yn.t()
    labels = torch.arange(xn.size(0)).to(logits.device)
    return cross_entropy(logits, labels, reduction=reduction)


def contrastive_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    logit_scale: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    CLIP-like contrastive loss

    Args:
        x (`torch.Tensor`): prediction
        y (`torch.Tensor`): target
        logit_scale (`torch.Tensor`): logit scale
        reduction (`Literal["mean", "sum", "none"]`): reduction to apply

    Returns: the contrastive loss
    """
    xn = normalize(x)
    yn = normalize(y)
    logits = torch.clamp(logit_scale.exp(), max=100) * xn @ yn.t()
    labels = torch.arange(xn.size(0)).to(logits.device)
    ce = cross_entropy(logits, labels, reduction=reduction)
    ce_t = cross_entropy(logits.t(), labels, reduction=reduction)
    return 0.5 * (ce + ce_t)


def contrastive_loss_with_confidence(
    x: torch.Tensor,
    x_confidence: torch.Tensor,
    y: torch.Tensor,
    y_confidence: torch.Tensor,
    logit_scale: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """
    CLIP-like contrastive loss with confidence.
    This is used in Global Workspaces with confidence.

    Args:
        x (`torch.Tensor`): prediction
        x_confidence (`torch.Tensor`): logvar of the prediction
        y (`torch.Tensor`): target
        y_confidence (`torch.Tensor`): logvar of the target
        logit_scale (`torch.Tensor`): logit scale
        reduction (`Literal["mean", "sum", "none"]`): reduction to apply

    Returns: the contrastive loss with confidence.
    """
    confidence_norm = 1 + torch.exp(0.5 * x_confidence) + torch.exp(0.5 * y_confidence)
    xn = normalize(x) / confidence_norm
    yn = normalize(y) / confidence_norm
    logits = torch.clamp(logit_scale.exp(), max=100) * xn @ yn.t()
    labels = torch.arange(xn.size(0)).to(logits.device)
    ce = cross_entropy(logits, labels, reduction=reduction)
    ce_t = cross_entropy(logits.t(), labels, reduction=reduction)
    return 0.5 * (ce + ce_t)


class ContrastiveLoss(torch.nn.Module):
    """CLIP-like ContrastiveLoss torch module."""

    def __init__(
        self,
        logit_scale: torch.Tensor,
        reduction: Literal["mean", "sum", "none"] = "mean",
        learn_logit_scale: bool = False,
    ) -> None:
        """
        Initializes a contrastive loss.

        Args:
            logit_scale (`torch.Tensor`): logit_scale tensor.
            reduction (`Literal["mean", "sum", "none"]`): reduction to apply to the
                loss. Defaults to `"mean"`.
            learn_logit_scale (`torch.Tensor`): whether to learn the `logit_scale`
                parameter. Defaults to `False`.
        """
        super().__init__()

        if learn_logit_scale:
            self.logit_scale = torch.nn.Parameter(logit_scale)
        else:
            self.register_buffer("logit_scale", logit_scale)
        self.learn_logit_scale = learn_logit_scale
        self.reduction: Literal["mean", "sum", "none"] = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> LossOutput:
        """
        Computes the loss.

        Args:
            x (`torch.Tensor`): prediction
            y (`torch.Tensor`): target

        Returns:
            LossOutput of the loss. Contains a `logit_scale` metric.
        """
        return LossOutput(
            contrastive_loss(x, y, self.logit_scale, self.reduction),
            {"logit_scale": self.logit_scale.exp()},
        )


class ContrastiveLossWithConfidence(torch.nn.Module):
    """
    CLIP-like contrastive loss with confidence module.

    This is used in Global Workspaces with confidence.
    """

    def __init__(
        self,
        logit_scale: torch.Tensor,
        reduction: Literal["mean", "sum", "none"] = "mean",
        learn_logit_scale: bool = False,
    ) -> None:
        """
        ContrastiveLoss used for GlobalWorkspaceWithConfidence

        Args:
            logit_scale (`torch.Tensor`): logit_scale tensor.
            reduction (`Literal["mean", "sum", "none"]`): reduction to apply to
                the loss. Defaults to `"mean"`.
            learn_logit_scale (`bool`): whether to learn the logit_scale parameter.
            Defaults to `False`.
        """
        super().__init__()

        if learn_logit_scale:
            self.logit_scale = torch.nn.Parameter(logit_scale)
        else:
            self.register_buffer("logit_scale", logit_scale)
        self.learn_logit_scale = learn_logit_scale
        self.reduction: Literal["mean", "sum", "none"] = reduction

    def forward(
        self,
        x: torch.Tensor,
        x_confidence: torch.Tensor,
        y: torch.Tensor,
        y_confidence: torch.Tensor,
    ) -> LossOutput:
        """
        Computes the loss
        Args:
            x: prediction
            x_confidence: prediction logvar
            y: target
            y_confidence: target logvar

        Returns:
            LossOutput of the loss. Contains a `"logit_scale"` metric and a
            `"no_confidence"` metric with the classic contrastive loss computed without
            the logvar information.
        """
        loss = contrastive_loss_with_confidence(
            x,
            x_confidence,
            y,
            y_confidence,
            self.logit_scale,
            self.reduction,
        )

        metrics = {
            "no_confidence": contrastive_loss(x, y, self.logit_scale, self.reduction),
            "logit_scale": self.logit_scale.exp(),
        }

        return LossOutput(loss, metrics)
