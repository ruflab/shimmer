import torch

from abc import ABC, abstractmethod


from shimmer.types import LatentsDomainGroupT


class SelectionBase(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, domains: LatentsDomainGroupT, gw_state: torch.Tensor
    ) -> dict[str, torch.Tensor]: ...
