from abc import ABC, abstractmethod

import torch
import torch.nn as nn

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


class KQAttentionOnePass(SelectionBase):
    """
    Key-Query attention with a fixed gw vector.
    """


    def __init__(self, domain_dim, head_size):
        super().__init__()
        self.head_size = head_size
        self.query_layer = nn.Linear(domain_dim, head_size)
        self.key_layers = nn.ModuleDict(
            {
                "v_latents": nn.Linear(domain_dim, head_size),
                "attr": nn.Linear(domain_dim, head_size),
            }
        )
        self.gw_state=None

    def update_gw_state(self, gw_state: torch.Tensor) -> None:
        """
        Set the internal copy of the fixed gw state. You're meant to only call this once

        Args:
            gw_state (`torch.Tensor`): the previous GW state
        """
        self.gw_state = gw_state


    def forward(
        self, domains: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
        """
        Compute keys and queries, match them with dot product and softmax.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.

        Returns:
            `dict[str, torch.Tensor]`: for each domain in the group, the fusion
            coefficient for each item in the batch.
        """


        keys = {
            domain: self.key_layers[domain](encoding)
            for domain, encoding in domains.items()
        }

        if self.gw_state is None:
            raise ValueError("GW state has not been initialized.")

        device = next(iter(domains.values())).device
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
    def __init__(self, binary_proportion, temperature):
        super().__init__()
        self.binary_proportion = binary_proportion
        self.temperature = temperature

    def forward(self, domains: LatentsDomainGroupT) -> dict[str, torch.Tensor]:
        num_domains = len(domains)
        batch_size = next(iter(domains.values())).shape[0]

        # Calculate the number of samples for each part
        num_uniform = int(batch_size * (1 - self.binary_proportion))
        num_binary_per_domain = (batch_size - num_uniform) // num_domains

        # Generate uniform scores
        uniform_scores = torch.rand(num_uniform, num_domains)

        # Apply softmax with temperature to uniform scores
        softmax_scores = torch.softmax(uniform_scores / self.temperature, dim=1)

        # Generate binary scores
        binary_scores = torch.cat([
            torch.cat([
                torch.zeros(num_binary_per_domain, i),
                torch.ones(num_binary_per_domain, 1),
                torch.zeros(num_binary_per_domain, num_domains - i - 1)
            ], dim=1) for i in range(num_domains)
        ], dim=0)

        # Concatenate the scores
        all_scores = torch.cat([softmax_scores, binary_scores], dim=0)

        # Ensure the scores shape matches the expected output (batch_size, num_domains)
        if all_scores.shape[0] < batch_size:
            # If there are missing scores due to division, fill with binary scores for the last domain
            missing_scores = batch_size - all_scores.shape[0]
            print("missing scores : ",missing_scores)
            last_domain_scores = torch.cat([
                torch.zeros(missing_scores, num_domains - 1),
                torch.ones(missing_scores, 1)
            ], dim=1)
            all_scores = torch.cat([all_scores, last_domain_scores], dim=0)

        # Convert scores to the expected output format: dict[str, torch.Tensor]
        attention_dict = {domain: all_scores[:, i:i+1] for i, domain in enumerate(domains)}

        return attention_dict
