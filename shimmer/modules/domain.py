from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch
from torch.nn.functional import mse_loss


class DomainModule(pl.LightningModule):
    def encode(self, x: Any) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> Any:
        raise NotImplementedError

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {"loss": mse_loss(pred, target, reduction="sum")}


@dataclass
class DomainDescription:
    module: DomainModule
    latent_dim: int
    encoder_hidden_dim: int
    encoder_n_layers: int
    decoder_hidden_dim: int
    decoder_n_layers: int
