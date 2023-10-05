from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch
from torch.nn.functional import mse_loss


class DomainModule(pl.LightningModule):
    def encode(self, x: Any) -> torch.Tensor:
        raise NotImplementedError

    def on_before_gw_encode_dcy(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def on_before_gw_encode_cont(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def on_before_gw_encode_tr(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def on_before_gw_encode_cy(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> Any:
        raise NotImplementedError

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {"loss": mse_loss(pred, target)}

    def compute_dcy_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.compute_loss(pred, target)

    def compute_cy_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.compute_loss(pred, target)

    def compute_tr_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.compute_loss(pred, target)


@dataclass
class DomainDescription:
    module: DomainModule
    latent_dim: int
    encoder_hidden_dim: int
    encoder_n_layers: int
    decoder_hidden_dim: int
    decoder_n_layers: int
