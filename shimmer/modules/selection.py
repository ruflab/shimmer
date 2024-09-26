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
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
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

    # This is just for proper auto-completion...
    def __call__(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
        return super().__call__(domains, encodings_pre_fusion)


class SingleDomainSelection(SelectionBase):
    """
    This selection mechanism handles groups that can have multiple domains, but always
    return a selection of 1 domain from the group with a uniform distribution.

    For example, if the group has 2 domains, there is a 50% chance of selecting each
    domain.
    """

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
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


class FixedSharedSelection(SelectionBase):
    """
    This selection mechanism is deterministic and always shares the weights equally
    between domains.

    For example, if 2 domains, it gives 0.5 for each; 3 domains, 1/3 for each...
    """

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
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
        coef = torch.full((bs,), 1.0 / len(domains), device=group_device(domains))
        for domain in domains:
            selection[domain] = coef.clone()
        return selection


def _calculate_attention_dict(
    domains: LatentsDomainGroupT,
    keys: dict[str, torch.Tensor],
    query: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Args:
        domains (`LatentsDomainGroupT`): Group of unimodal latent representations.
        keys (`dict[str, torch.Tensor]`): The keys for each domain.
        query (`torch.Tensor`): The query tensor.

    Returns:
        `dict[str, torch.Tensor]`: The attention scores for each domain.
    """
    dot_products = {
        domain: torch.bmm(key.unsqueeze(1), query.unsqueeze(2)).squeeze()
        for domain, key in keys.items()
    }

    dot_products_tensor = torch.stack(list(dot_products.values()), dim=1)

    attention_scores = torch.softmax(dot_products_tensor, dim=1)

    attention_dict = {
        domain: attention_scores[:, i] for i, domain in enumerate(domains)
    }
    return attention_dict


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
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
        """
        Generate uniform-then-domain-wise-softmaxed samples for each domain.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.
                This is not used in the function directly but determines the structure
                of the returned attention coefficients.

        Returns:
            `dict[str, torch.Tensor]`: For each domain in the group, the fusion
            coefficient for each item in the batch, based solely on
            uniform-softmax scores.
        """
        num_domains = len(domains)
        batch_size = group_batch_size(domains)
        device = group_device(domains)

        # Generate uniform scores
        uniform_scores = torch.rand(batch_size, num_domains, device=device)

        # Apply softmax across domains with temperature scaling
        softmax_scores = torch.softmax(uniform_scores / self.temperature, dim=1)
        # Create attention dictionary for each domain
        attention_dict = {
            domain: softmax_scores[:, i] for i, domain in enumerate(domains)
        }

        return attention_dict


class DynamicQueryAttention(SelectionBase):
    """
    Key-Query attention with a dynamic gw vector.
    The query is updated based on the scaled gw vector.
    """

    def __init__(
        self,
        head_size: int,
        domain_dim: int,
        domain_names: Iterable[str],
        n_steps: int = 1,
    ):
        """
        Args:
            head_size (`int`) : dimension of the key and query vectors.
            domain_dim (`int`) : dimension of the input dims (assumed to be the same
                for now)
            domain_names  (`Iterable[str]`) : list of input domains
            n_steps (`int`) : number of steps to update the query vector
        """
        super().__init__()
        self.head_size = head_size
        self.query_layer = nn.Linear(domain_dim, head_size)
        self.key_layers = nn.ModuleDict(
            {domain: nn.Linear(domain_dim, head_size) for domain in domain_names}
        )
        self.n_steps = n_steps
        self.step_limit = n_steps  # Default step limit is n_steps
        # Start with a random gw state
        self.register_buffer("initial_gw_state", torch.rand(domain_dim))

    def set_step_limit(self, step_limit: int):
        """
        Sets the step limit for the dynamic attention update loop.

        Args:
            step_limit (`int`): Maximum number of steps to run the loop.
        """
        if step_limit > self.n_steps:
            raise ValueError(f"Step limit cannot exceed the maximum n_steps ({self.n_steps}).")
        self.step_limit = step_limit

    def fuse_weighted_encodings(
        self, encodings: LatentsDomainGroupT, attention_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse the weighted encodings using the attention scores.

        Args:
            encodings (`LatentsDomainGroupT`): Unimodal latent representation
            attention_dict (`dict[str, torch.Tensor]`): The attention scores for each
                domain in the group.

        Returns:
            `torch.Tensor`: The fused tensor.
        """
        # Apply attention scores to the encodings
        weighted_encodings = {}
        for key in attention_dict:
            if key in encodings:
                # Perform element-wise multiplication
                weighted_encodings[key] = (
                    attention_dict[key].unsqueeze(1) * encodings[key]
                )

        # Stack the tensors along a new dimension (dimension 0)
        stacked_tensors = torch.stack(list(weighted_encodings.values()))

        # Apply fusion by summing along the newly created dimension
        summed_tensor = torch.sum(stacked_tensors, dim=0)
        return summed_tensor

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
        """
        Compute keys and queries, match them with dot product and softmax.
        Does this twice, once with the static query and once with a dynamic query.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.
            encodings (`LatentsDomainGroupT`): Group of pre-fusion encodings.

        Returns:
            `dict[str, torch.Tensor]`: the attention scores for each domain in the
            group.
        """

        keys = {
            domain: self.key_layers[domain](encoding)
            for domain, encoding in domains.items()
        }

        batch_size = group_batch_size(domains)

        # Retrieve random query
        query = self.query_layer(self.initial_gw_state.expand(batch_size, -1))

        # Calculate the attention scores
        attention_dict = _calculate_attention_dict(domains, keys, query)

        if self.n_steps > 0:
            # Update the query based on the static attention scores
            for _ in range(min(self.step_limit, self.n_steps)):
                # Apply the attention scores to the encodings
                summed_tensor = self.fuse_weighted_encodings(
                    encodings_pre_fusion, attention_dict
                )

                # Retrieve query (now it is dependent on the new gw state)
                query = self.query_layer(summed_tensor)

                # Calculate the attention scores again
                attention_dict = _calculate_attention_dict(domains, keys, query)

        return attention_dict
