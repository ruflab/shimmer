from dataclasses import dataclass, field
from typing import Any

import lightning.pytorch as pl
import torch


@dataclass
class LossOutput:
    # Loss used during training
    loss: torch.Tensor
    # Some additional metrics to log (not used during training)
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if "loss" in self.metrics.keys():
            raise ValueError("'loss' cannot be a key of metrics.")

    @property
    def all(self) -> dict[str, torch.Tensor]:
        """
        Returns a dict with all metrics and loss with "loss" key
        """
        return {**self.metrics, "loss": self.loss}


class DomainModule(pl.LightningModule):
    """
    Base class for a DomainModule.
    We do not use ABC here because some modules could be without encore or decoder.
    """

    def __init__(
        self,
        latent_dim: int,
    ) -> None:
        """
        Args:
            latent_dim: latent dimension of the unimodal module
            encoder_hidden_dim: number of hidden
        """
        super().__init__()

        self.latent_dim = latent_dim

    def encode(self, x: Any) -> torch.Tensor:
        """
        Encode data to the unimodal representation.
        Args:
            x: data of the domain.
        Returns:
            a unimodal representation.
        """
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> Any:
        """
        Decode data back to the domain data.
        Args:
            z: unimodal representation of the domain.
        Returns:
            the original domain data.
        """
        raise NotImplementedError

    def on_before_gw_encode_dcy(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def on_before_gw_encode_cont(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def on_before_gw_encode_tr(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def on_before_gw_encode_cy(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def on_before_gw_encode_broadcast(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        """
        Computes the loss of the modality. If you implement compute_dcy_loss,
        compute_cy_loss and compute_tr_loss independently, no need to define this
        function.
        Args:
            pred: tensor with a predicted latent unimodal representation
            target: target tensor
        Results:
            Dict of losses. Must contain the "loss" key with the total loss
            used for training. Any other key will be logged, but not trained on.
        """
        raise NotImplementedError

    def compute_dcy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        """
        Computes the loss for a demi-cycle. Override if the demi-cycle loss is
        different that the generic loss.
        Args:
            pred: tensor with a predicted latent unimodal representation
            target: target tensor
        Results:
            Dict of losses. Must contain the "loss" key with the total loss
            used for training. Any other key will be logged, but not trained on.
        """
        return self.compute_loss(pred, target)

    def compute_cy_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        """
        Computes the loss for a cycle. Override if the cycle loss is
        different that the generic loss.
        Args:
            pred: tensor with a predicted latent unimodal representation
            target: target tensor
        Results:
            Dict of losses. Must contain the "loss" key with the total loss
            used for training. Any other key will be logged, but not trained on.
        """
        return self.compute_loss(pred, target)

    def compute_tr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        """
        Computes the loss for a translation. Override if the translation loss is
        different that the generic loss.
        Args:
            pred: tensor with a predicted latent unimodal representation
            target: target tensor
        Results:
            Dict of losses. Must contain the "loss" key with the total loss
            used for training. Any other key will be logged, but not trained on.
        """
        return self.compute_loss(pred, target)

    def compute_broadcast_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> LossOutput:
        """
        Computes the loss for a broadcast (fusion). Override if the translation loss is
        different that the generic loss.
        Args:
            pred: tensor with a predicted latent unimodal representation
            target: target tensor
        Results:
            Dict of losses. Must contain the "loss" key with the total loss
            used for training. Any other key will be logged, but not trained on.
        """
        return self.compute_loss(pred, target)
