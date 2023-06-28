from typing import Any

import torch


class DomainModule(torch.nn.Module):
    def encode(self, x: Any) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> Any:
        raise NotImplementedError
