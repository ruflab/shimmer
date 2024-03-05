from dataclasses import dataclass, field
from typing import Any

import lightning.pytorch as pl
import torch


@dataclass
class LossOutput:
    """This is a python dataclass use as a returned value for losses.
    It keeps track of what is used for training (`loss`) and what is used
    only for logging (`metrics`).
    """

    loss: torch.Tensor
    """Loss used during training."""

    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    """Some additional metrics to log (not used during training)."""

    def __post_init__(self):
        if "loss" in self.metrics.keys():
            raise ValueError("'loss' cannot be a key of metrics.")

    @property
    def all(self) -> dict[str, torch.Tensor]:
        """
        Returns a dict with all metrics and loss with "loss" key.
        """
        return {**self.metrics, "loss": self.loss}


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

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        """Generic loss computation  the modality.

        Args:
            pred (`torch.Tensor`): prediction of the model
            target (`torch.Tensor`): target tensor
        Results:
            `LossOutput`: LossOuput with training loss and additional metrics.
        """
        raise NotImplementedError

    def compute_dcy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        """
        Computes the loss for a demi-cycle. Override if the demi-cycle loss is
        different that the generic loss.

        Args:
            pred (`torch.Tensor`): prediction of the model
            target (`torch.Tensor`): target tensor
        Results:
            `LossOutput`: LossOuput with training loss and additional metrics.
        """
        return self.compute_loss(pred, target)

    def compute_cy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        """
        Computes the loss for a cycle. Override if the cycle loss is
        different that the generic loss.

        Args:
            pred (`torch.Tensor`): prediction of the model
            target (`torch.Tensor`): target tensor
        Results:
            `LossOutput`: LossOuput with training loss and additional metrics.
        """
        return self.compute_loss(pred, target)

    def compute_tr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        """
        Computes the loss for a translation. Override if the translation loss is
        different that the generic loss.

        Args:
            pred (`torch.Tensor`): prediction of the model
            target (`torch.Tensor`): target tensor
        Results:
            `LossOutput`: LossOuput with training loss and additional metrics.
        """
        return self.compute_loss(pred, target)

    def compute_broadcast_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> LossOutput:
        """
        Computes the loss for a broadcast (fusion). Override if the broadcast loss is
        different that the generic loss.

        Args:
            pred (`torch.Tensor`): prediction of the model
            target (`torch.Tensor`): target tensor
        Results:
            `LossOutput`: LossOuput with training loss and additional metrics.
        """
        return self.compute_loss(pred, target)

    def on_before_gw_encode_dcy(self, z: torch.Tensor) -> torch.Tensor:
        """Some additional computation to do before encoding the unimodal latent
        representation to the GW when doing a demi-cycle loss.

        If not defined, will return the input (identity function).

        Args:
            z (`torch.Tensor`): latent representation

        Returns:
            `torch.Tensor`: The updated latent representation
        """
        return z

    def on_before_gw_encode_cont(self, z: torch.Tensor) -> torch.Tensor:
        """Some additional computation to do before encoding the unimodal latent
        representation to the GW when doing a contrastive loss.

        If not defined, will return the input (identity function).

        Args:
            z (`torch.Tensor`): latent representation

        Returns:
            `torch.Tensor`: The updated latent representation
        """
        return z

    def on_before_gw_encode_tr(self, z: torch.Tensor) -> torch.Tensor:
        """Some additional computation to do before encoding the unimodal latent
        representation to the GW when doing a translation loss.

        If not defined, will return the input (identity function).

        Args:
            z (`torch.Tensor`): latent representation

        Returns:
            `torch.Tensor`: the updated latent representation
        """
        return z

    def on_before_gw_encode_cy(self, z: torch.Tensor) -> torch.Tensor:
        """Some additional computation to do before encoding the unimodal latent
        representation to the GW when doing a cycle loss.

        If not defined, will return the input (identity function).

        Args:
            z (`torch.Tensor`): latent representation

        Returns:
            `torch.Tensor`: the updated latent representation
        """
        return z

    def on_before_gw_encode_broadcast(self, z: torch.Tensor) -> torch.Tensor:
        """Some additional computation to do before encoding the unimodal latent
        representation to the GW when doing a broadcast loss (used in Fusion GW).

        If not defined, will return the input (identity function).

        Args:
            z (`torch.Tensor`): latent representation

        Returns:
            `torch.Tensor`: the updated latent representation
        """
        return z
