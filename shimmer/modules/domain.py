from dataclasses import dataclass
from typing import Any

import lightning.pytorch as pl
import torch


class DomainModule(pl.LightningModule):
    def encode(self, x: Any) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> Any:
        raise NotImplementedError


@dataclass
class DomainDescription:
    module: DomainModule
    latent_dim: int
