from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal, TypedDict

import torch
from torch.nn.functional import cross_entropy, normalize

from shimmer.modules.domain import LossOutput


def info_nce(
    x: torch.Tensor,
    y: torch.Tensor,
    logit_scale: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
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
    xn = normalize(x)
    yn = normalize(y)
    logits = torch.clamp(logit_scale.exp(), max=100) * xn @ yn.t()
    labels = torch.arange(xn.size(0)).to(logits.device)
    ce = cross_entropy(logits, labels, reduction=reduction)
    ce_t = cross_entropy(logits.t(), labels, reduction=reduction)
    return 0.5 * (ce + ce_t)


def contrastive_loss_with_uncertainty(
    x: torch.Tensor,
    x_log_uncertainty: torch.Tensor,
    y: torch.Tensor,
    y_log_uncertainty: torch.Tensor,
    logit_scale: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    uncertainty_norm = (
        1
        + torch.exp(0.5 * x_log_uncertainty)
        + torch.exp(0.5 * y_log_uncertainty)
    )
    xn = normalize(x) / uncertainty_norm
    yn = normalize(y) / uncertainty_norm
    logits = torch.clamp(logit_scale.exp(), max=100) * xn @ yn.t()
    labels = torch.arange(xn.size(0)).to(logits.device)
    ce = cross_entropy(logits, labels, reduction=reduction)
    ce_t = cross_entropy(logits.t(), labels, reduction=reduction)
    return 0.5 * (ce + ce_t)


class ContrastiveLossBase(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> LossOutput:
        ...


class ContrastiveLoss(ContrastiveLossBase):
    logit_scale: torch.Tensor

    def __init__(
        self,
        logit_scale: torch.Tensor,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()

        self.register_buffer("logit_scale", logit_scale)
        self.reduction: Literal["mean", "sum", "none"] = reduction

    def __call__(self, *args, **kwargs) -> LossOutput:
        return super().__call__(*args, **kwargs)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> LossOutput:
        return LossOutput(
            contrastive_loss(x, y, self.logit_scale, self.reduction),
            {"logit_scale": self.logit_scale.exp()},
        )


class ContrastiveLossWithUncertainty(ContrastiveLossBase):
    logit_scale: torch.Tensor

    def __init__(
        self,
        logit_scale: torch.Tensor,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()

        self.register_buffer("logit_scale", logit_scale)
        self.reduction: Literal["mean", "sum", "none"] = reduction

    def forward(
        self,
        x: torch.Tensor,
        x_log_uncertainty: torch.Tensor,
        y: torch.Tensor,
        y_log_uncertainty: torch.Tensor,
    ) -> LossOutput:
        return LossOutput(
            contrastive_loss_with_uncertainty(
                x,
                x_log_uncertainty,
                y,
                y_log_uncertainty,
                self.logit_scale,
                self.reduction,
            ),
            {
                "no_uncertainty": contrastive_loss(
                    x, y, self.logit_scale, self.reduction
                ),
                "logit_scale": self.logit_scale.exp(),
            },
        )
