from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
import torch.nn as nn

from shimmer.types import LatentsDomainGroupT
from shimmer.utils import group_batch_size, group_device


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
    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the selection method.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.

        Returns:
            `torch.Tensor`: for each domain in the group, the fusion
            coefficient for each item in the batch.

        Example:
            >>> SomeSelectionImplementation().forward(
            ...     {"v": torch.randn(3, 4), "t": torch.randn(3, 8)},
            ...     torch.randn(2, 3, 12),
            ... )
            torch.Tensor([[0.0, 0.4, 1.0], [1.0, 0.6, 0.0]])}
        """
        ...

    # This is just for proper auto-completion...
    def __call__(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: torch.Tensor
    ) -> torch.Tensor:
        return super().__call__(domains, encodings_pre_fusion)


class SingleDomainSelection(SelectionBase):
    """
    This selection mechanism handles groups that can have multiple domains, but always
    return a selection of 1 domain from the group with a uniform distribution.

    For example, if the group has 2 domains, there is a 50% chance of selecting each
    domain.
    """

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the module.

        Args:
            domains (`LatentsDomainGroupT`): input unimodal latent representations
            encodings_pre_fusion (`torch.Tensor`): pre-fusion GW representation

        Returns:
            `torch.Tensor`: whether the domain is selected for each input in the batch.
        """
        bs = group_batch_size(domains)
        device = group_device(domains)
        selection = torch.zeros(len(domains), bs, device=device)
        choice = torch.randint(len(domains), size=(bs,), device=device)
        for k in range(selection.size(0)):
            selection[k] = (choice == k).to(torch.float32)
        return selection


class KQFixedQSelection(SelectionBase):
    """
    Key-Query attention with a fixed gw vector.
    """

    def __init__(self, domain_dim: int, head_size: int, domains: Iterable[str]):
        """
        Args:
            domain_dim (`int`) : dimension of the input dims
                (assumed to be the same for now)
            head_size (`int`) : dimension of the key and query vectors.
            domains (`Iterable[str]`) : list of input domains
        """
        super().__init__()
        self.head_size = head_size
        self.query_layer = nn.Linear(domain_dim, head_size)
        self.key_layers = nn.ModuleDict(
            {domain: nn.Linear(domain_dim, head_size) for domain in domains}
        )
        self.gw_state: torch.Tensor | None = None

    def update_gw_state(self, gw_state: torch.Tensor) -> None:
        """
        Set the internal copy of the fixed gw state. You're meant to only call this once

        Args:
            gw_state (`torch.Tensor`): the previous GW state
        """
        self.gw_state = gw_state

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute keys and queries, match them with dot product and softmax.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.
            encodings_pre_fusion (`torch.Tensor`): pre-fusion GW representation

        Returns:
            `torch.Tensor`: for each domain in the group, the fusion
            coefficient for each item in the batch.
        """

        if self.gw_state is None:
            raise ValueError("GW state has not been initialized.")

        keys = {
            domain: self.key_layers[domain](encoding)
            for domain, encoding in domains.items()
        }

        device = group_device(domains)
        query = self.query_layer(self.gw_state.to(device))

        dot_products = {
            domain: torch.bmm(key.unsqueeze(1), query.unsqueeze(2)).squeeze()
            for domain, key in keys.items()
        }

        dot_products_tensor = torch.stack(list(dot_products.values()), dim=0)
        attention_scores = torch.softmax(dot_products_tensor, dim=0)

        return attention_scores


class RandomSelection(SelectionBase):
    """
    Modified random attention to only utilize uniform-softmax scores across modalities.
    This version omits the binary scaling factors and focuses on generating attention
    coefficients using a uniform distribution followed by a domain-wise softmax.
    """

    def __init__(self, temperature: float):
        """
        Args:
            temperature (`float`): Temperature of the softmax applied to uniform
                scaling factors.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate uniform-then-domain-wise-softmaxed samples for each domain.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.
                This is not used in the function directly but determines the structure
                of the returned attention coefficients.
            encodings_pre_fusion (`torch.Tensor`): pre-fusion GW representation

        Returns:
            `torch.Tensor`: For each domain in the group, the fusion
            coefficient for each item in the batch, based solely on
            uniform-softmax scores.
        """
        num_domains = len(domains)
        batch_size = group_batch_size(domains)
        device = group_device(domains)

        # Generate uniform scores
        uniform_scores = torch.rand(num_domains, batch_size, device=device)

        # Apply softmax across domains with temperature scaling
        return torch.softmax(uniform_scores / self.temperature, dim=0)


class DynamicQueryAttention(SelectionBase):
    """
    Key-Query attention with a dynamic gw vector.
    """

    def __init__(
        self, batch_size: int, domain_dim: int, head_size: int, domains: Iterable[str]
    ):
        """
        Args:
            batch_size (`int`) : size of the batch
            domain_dim (`int`) : dimension of the input dims (assumed to be the same
                for now)
            head_size (`int`) : dimension of the key and query vectors
            domains (`Iterable[str]`) : list of input domains
        """
        super().__init__()
        self.batch_size = batch_size
        self.head_size = head_size
        self.query_layer = nn.Linear(domain_dim, head_size)
        self.key_layers = nn.ModuleDict(
            {domain: nn.Linear(domain_dim, head_size) for domain in domains}
        )
        # Start with a random gw state
        self.gw_state = torch.rand(batch_size, domain_dim)

    def calculate_attention_dict(
        self, keys: dict[str, torch.Tensor], query: torch.Tensor
    ) -> torch.Tensor:
        dot_products = {
            domain: torch.bmm(key.unsqueeze(1), query.unsqueeze(2)).squeeze()
            for domain, key in keys.items()
        }

        dot_products_tensor = torch.stack(list(dot_products.values()), dim=0)

        return torch.softmax(dot_products_tensor, dim=0)

    def fuse_weighted_encodings(
        self, encodings: torch.Tensor, attention_dict: torch.Tensor
    ) -> torch.Tensor:
        # Stack the tensors along a new dimension (dimension 0)
        stacked_tensors = attention_dict * encodings
        return torch.sum(stacked_tensors, dim=0)

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute keys and queries, match them with dot product and softmax.
        Does this twice, once with the static query and once with a dynamic query.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.
            encodings_pre_fusion (`torch.Tensor`): Group of pre-fusion encodings.

        Returns:
            `torch.Tensor`: the attention scores for each domain in the group.
        """

        # Encoding with pytorch
        keys = {
            domain: self.key_layers[domain](encoding)
            for domain, encoding in domains.items()
        }

        # This for training (cpu or gpu)
        device = group_device(domains)
        self.gw_state = self.gw_state.to(device)

        # Retrieve query
        query = self.query_layer(self.gw_state)

        # Calculate the attention scores
        static_attention_dict = self.calculate_attention_dict(keys, query)

        # Apply the attention scores to the encodings
        summed_tensor = self.fuse_weighted_encodings(
            encodings_pre_fusion, static_attention_dict
        )

        # Retrieve query (now it is dependent on the new gw state)
        query = self.query_layer(summed_tensor)

        # Calculate the attention scores again
        return self.calculate_attention_dict(keys, query)
