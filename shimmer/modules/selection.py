from abc import ABC, abstractmethod

import torch
import torch.nn as nn

import shimmer
from shimmer.types import LatentsDomainGroupT


class SelectionBase(torch.nn.Module, ABC):
    """
    This is the base class for the selection mechanism.
    The selection mechanisms handles the "competition" between modules and *selects*
    fusion coefficients for the domains.
    """

    def update_gw_state(self, gw_state: torch.Tensor) -> None:
        """
        Update the internal copy of the previous GW state.
        By default, this is not implemented and will raise an error if used.

        :note..
            This is not defined as an abstractmethod as some selection method may
            not need it.

        Args:
            gw_state (`torch.Tensor`): the previous GW state
        """
        pass

    @abstractmethod
    def forward(self, domains: LatentsDomainGroupT) -> dict[str, torch.Tensor]:
        """
        Forward pass of the selection method.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.

        Returns:
            `dict[str, torch.Tensor]`: for each domain in the group, the fusion
            coefficient for each item in the batch.

        Example:
            >>> SomeSelectionImplementation().forward(
            ...     {"v": torch.randn(3, 4), "t": torch.randn(3, 8)}
            ... )
            {"v": torch.Tensor([0.0, 0.4, 1.0]), "t": torch.Tensor([1.0, 0.6, 0.0])}
        """
        ...


class KQSelectionFixedQ(SelectionBase):
    """
    Key-Query attention with a fixed gw vector.
    """

    def __init__(self, domain_dim, head_size):
        """
        Args:
            domain_dim (`int`) : dimension of the input dims (assumed to be the same for now)
            head_size (`int`) : dimension of the key and query vectors.
        """
        super().__init__()
        self.head_size = head_size
        self.query_layer = nn.Linear(domain_dim, head_size)
        self.key_layers = nn.ModuleDict(
            {
                "v_latents": nn.Linear(domain_dim, head_size),
                "attr": nn.Linear(domain_dim, head_size),
            }
        )
        self.gw_state: torch.Tensor | None = None

    def update_gw_state(self, gw_state: torch.Tensor) -> None:
        """
        Set the internal copy of the fixed gw state. You're meant to only call this once

        Args:
            gw_state (`torch.Tensor`): the previous GW state
        """
        self.gw_state = gw_state

    def forward(self, domains: LatentsDomainGroupT) -> dict[str, torch.Tensor]:
        """
        Compute keys and queries, match them with dot product and softmax.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.

        Returns:
            `dict[str, torch.Tensor]`: for each domain in the group, the fusion
            coefficient for each item in the batch.
        """

        if self.gw_state is None:
            raise ValueError("GW state has not been initialized.")

        keys = {
            domain: self.key_layers[domain](encoding)
            for domain, encoding in domains.items()
        }

        device = shimmer.utils.group_device(domains)
        query = self.query_layer(self.gw_state.to(device))

        dot_products = {
            domain: torch.bmm(key.unsqueeze(1), query.unsqueeze(2)).squeeze()
            for domain, key in keys.items()
        }

        dot_products_tensor = torch.stack(list(dot_products.values()), dim=1)

        attention_scores = torch.softmax(dot_products_tensor, dim=1)

        attention_dict = {
            domain: attention_scores[:, i : i + 1] for i, domain in enumerate(keys)
        }

        return attention_dict


class RandomSelection(SelectionBase):
    """
    random attention, not learned, with a proportion of binary scaling factors, and a proportion of uniform-then-softmaxed-across-modalities scores.
    this class serves to train broadcast with robustness on linear scaling on prefusion representations.
    """

    def __init__(self, binary_proportion, temperature):
        """
        Args:
            binary_proportion (`float`) : proportion of binary scaling factors returned by forward(). between 0 and 1.
            temperature (`float`) : temperature of the softmax applied to uniform scaling factors.
        """
        super().__init__()
        self.binary_proportion = binary_proportion
        self.temperature = temperature

    def forward(self, domains: LatentsDomainGroupT) -> dict[str, torch.Tensor]:
        """
        randomly draw binary and uniform-then-domain-wise-softmaxed samples according to self.binary_proportion.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations. This is not used in the function.

        Returns:
            `dict[str, torch.Tensor]`: for each domain in the group, the fusion
            coefficient for each item in the batch.
        """
        num_domains = len(domains)
        batch_size = shimmer.utils.group_batch_size(domains)

        # have to add extra binaries when the division's not integer
        total_binary_scores = int(batch_size * self.binary_proportion)
        num_binary_per_domain, extra_binary_scores = divmod(
            total_binary_scores, num_domains
        )

        # Calculate number of uniform scores taking into account extra binary scores
        num_uniform = batch_size - total_binary_scores

        uniform_scores = torch.rand(num_uniform, num_domains)
        softmax_scores = torch.softmax(uniform_scores / self.temperature, dim=1)

        # Generate binary scores, adjusting for any extra binary scores
        binary_scores = []
        for i in range(num_domains):
            binary_score = torch.zeros(
                num_binary_per_domain + (1 if i < extra_binary_scores else 0),
                num_domains,
            )
            binary_score[:, i] = 1
            binary_scores.append(binary_score)
        binary_scores_concat = torch.cat(binary_scores, dim=0)

        all_scores = torch.cat([softmax_scores, binary_scores_concat], dim=0)
        attention_dict = {
            domain: all_scores[:, i : i + 1] for i, domain in enumerate(domains)
        }
        return attention_dict
