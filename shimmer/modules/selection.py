from abc import ABC, abstractmethod

import torch

from shimmer.types import LatentsDomainGroupT
from shimmer.utils import group_batch_size, group_device


class SelectionBase(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, domains: LatentsDomainGroupT, gw_state: torch.Tensor
    ) -> dict[str, torch.Tensor]: ...


class SingleDomainSelection(SelectionBase):
    """
    This selection mechanism handles groups that can have multiple domains, but always
    return a selection of 1 domain from the group with a uniform distribution.

    For example, if the group has 2 domains, there is a 50% chance of selecting each
    domain.
    """

    def forward(
        self, domains: LatentsDomainGroupT, gw_state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the module.

        Args:
            domains (`LatentsDomainGroupT`): input unimodal latent representations
            gw_state (`torch.Tensor`): the previous GW state

        Returns:
            `dict[str, torch.Tensor]`: whether the domain is selected for each input
            in the batch.
        """
        selection: dict[str, torch.Tensor] = {}
        bs = group_batch_size(domains)
        choice = torch.randint(len(domains), size=(bs,), device=group_device(domains))
        for k, domain in enumerate(domains.keys()):
            selection[domain] = (choice == k).to(torch.float32)
        return selection
