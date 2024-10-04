from dataclasses import dataclass, field
from typing import Any

import lightning.pytorch as pl
import torch


@dataclass
class LossOutput:
    """
    This is a python dataclass use as a returned value for losses.
    It keeps track of what is used for training (`loss`) and what is used
    only for logging (`metrics`).
    """

    loss: torch.Tensor
    """Loss used during training."""

    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    """Some additional metrics to log (not used during training)."""

    def __post_init__(self):
        if "loss" in self.metrics:
            raise ValueError("'loss' cannot be a key of metrics.")

    @property
    def all(self) -> dict[str, torch.Tensor]:
        """
        Returns a dict with all metrics and loss with "loss" key.
        """
        return {**self.metrics, "loss": self.loss}

    def add(self, other: "LossOutput", /, prefix: str | None = None) -> None:
        """
        Adds the other.loss to self.loss.
        Metrics of other that are not part of self.metrics, will also be added.

        You can also prepend a prefix to the metric names with the `prefix` argument.

        Args:
            other (`LossOutput`): `LossOutput` to add
            prefix (`str | None`): prefix to use. Defaults to no prefix.
        """
        prefix = prefix or ""
        for name, metric in other.metrics.items():
            metric_name = f"{prefix}{name}"
            if metric_name in self.metrics:
                self.metrics[metric_name] += metric
            else:
                self.metrics[metric_name] = metric
        self.loss += other.loss


class DomainModule(pl.LightningModule):
    """
    Base class for a DomainModule that defines domain specific modules of the GW.
    """

    def __init__(
        self,
        latent_dim: int,
    ) -> None:
        """
        Initializes a DomainModule.

        Args:
            latent_dim (`int`): latent dimension of the unimodal module
        """
        super().__init__()

        self.latent_dim = latent_dim
        """The latent dimension of the module."""

        self.is_frozen: bool | None = None
        """ Whether the module is frozen. If None, it is frozen by default. """

    def freeze(self) -> None:
        """
        Freezes the module. This is the default mode.
        """
        self.is_frozen = True
        return super().freeze()

    def unfreeze(self) -> None:
        """
        Unfreezes the module. This is usefull to train the domain module end-to-end.
        This also unlocks `compute_domain_loss` during training.
        """
        self.is_frozen = False
        return super().unfreeze()

    def encode(self, x: Any) -> torch.Tensor:
        """
        Encode the domain data into a unimodal representation.

        Args:
            x (`Any`): data of the domain.
        Returns:
            `torch.Tensor`: a unimodal representation.
        """
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> Any:
        """
        Decode data from unimodal representation back to the domain data.

        Args:
            z (`torch.Tensor`): unimodal representation of the domain.
        Returns:
            `Any`: the original domain data.
        """
        raise NotImplementedError

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput | None:
        """
        Generic loss computation  the modality.

        Args:
            pred (`torch.Tensor`): prediction of the model
            target (`torch.Tensor`): target tensor
            raw_target (`Any`): raw data from the input
        Results:
            `LossOutput | None`: LossOuput with training loss and additional metrics.
                If `None` is returned, this loss will be ignored and will not
                participate in the total loss.
        """
        raise NotImplementedError

    def compute_dcy_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput | None:
        """
        Computes the loss for a demi-cycle. Override if the demi-cycle loss is
        different that the generic loss.

        Args:
            pred (`torch.Tensor`): prediction of the model
            target (`torch.Tensor`): target tensor
            raw_target (`Any`): raw data from the input
        Results:
            `LossOutput | None`: LossOuput with training loss and additional metrics.
                If `None` is returned, this loss will be ignored and will not
                participate in the total loss; it can be used to deactivate
                demi-cycle loss for this domain.
        """
        return self.compute_loss(pred, target, raw_target)

    def compute_cy_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput | None:
        """
        Computes the loss for a cycle. Override if the cycle loss is
        different that the generic loss.

        Args:
            pred (`torch.Tensor`): prediction of the model
            target (`torch.Tensor`): target tensor
            raw_target (`Any`): raw data from the input
        Results:
            `LossOutput | None`: LossOuput with training loss and additional metrics.
                If `None` is returned, this loss will be ignored and will not
                participate in the total loss; it can be used to deactivate
                cycle loss for this domain.
        """
        return self.compute_loss(pred, target, raw_target)

    def compute_tr_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput | None:
        """
        Computes the loss for a translation. Override if the translation loss is
        different that the generic loss.

        Args:
            pred (`torch.Tensor`): prediction of the model
            target (`torch.Tensor`): target tensor
            raw_target (`Any`): raw data from the input
        Results:
            `LossOutput | None`: LossOuput with training loss and additional metrics.
                If `None` is returned, this loss will be ignored and will not
                participate in the total loss; it can be used to deactivate
                translation loss for this domain.
        """
        return self.compute_loss(pred, target, raw_target)

    def compute_fused_loss(
        self, pred: torch.Tensor, target: torch.Tensor, raw_target: Any
    ) -> LossOutput | None:
        """
        Computes the loss for fused (fusion). Override if the fused loss is
        different that the generic loss.

        Args:
            pred (`torch.Tensor`): prediction of the model
            target (`torch.Tensor`): target tensor
            raw_target (`Any`): raw data from the input
        Results:
            `LossOutput | None`: LossOuput with training loss and additional metrics.
                If `None` is returned, this loss will be ignored and will not
                participate in the total loss; it can be used to deactivate
                fused loss for this domain.
        """
        return self.compute_loss(pred, target, raw_target)

    def compute_domain_loss(self, domain: Any) -> LossOutput | None:
        """
        Compute the unimodal domain loss.

        Args:
            domain (`Any`): domain input
        Results:
            `LossOutput | None`: LossOuput with training loss and additional metrics.
                If `None` is returned, this loss will be ignored and will not
                participate in the total loss.
        """
        return None
