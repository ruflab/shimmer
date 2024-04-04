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


class KQFixedQSelection(SelectionBase):
    """
    Key-Query attention with a fixed gw vector.
    """

    def __init__(self, domain_dim: int, head_size: int):
        """
        Args:
            domain_dim (`int`) : dimension of the input dims
                (assumed to be the same for now)
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

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
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

        device = group_device(domains)
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
    random attention, not learned, with a proportion of binary scaling factors,
    and a proportion of uniform-then-softmaxed-across-modalities scores.
    this class serves to train broadcast with robustness on linear scaling on
    prefusion representations.
    """

    def __init__(self, binary_proportion: float, temperature: float):
        """
        Args:
            binary_proportion (`float`) : proportion of binary scaling factors
                returned by forward(). between 0 and 1.
            temperature (`float`) : temperature of the softmax applied to uniform
                scaling factors.
        """
        super().__init__()
        self.binary_proportion = binary_proportion
        self.temperature = temperature

    def forward(
        self, domains: LatentsDomainGroupT, encodings_pre_fusion: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
        """
        randomly draw binary and uniform-then-domain-wise-softmaxed samples according
        to self.binary_proportion.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.
                This is not used in the function.

        Returns:
            `dict[str, torch.Tensor]`: for each domain in the group, the fusion
            coefficient for each item in the batch.
        """
        num_domains = len(domains)
        batch_size = group_batch_size(domains)

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
            domain_dim (`int`) : dimension of the input dims (assumed to be the same for now)
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
        self, keys: dict, query: torch.Tensor
    ) -> dict[str, torch.Tensor]:
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

    def fuse_weighted_encodings(
        self, encodings: LatentsDomainGroupT, attention_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Apply attention scores to the encodings
        weighted_encodings = {}
        for key in attention_dict:
            if key in encodings:
                # Perform element-wise multiplication and store the result
                weighted_encodings[key] = attention_dict[key] * encodings[key]

        # Stack the tensors along a new dimension (dimension 0)
        stacked_tensors = torch.stack(list(weighted_encodings.values()))

        # Apply fusion by summing along the newly created dimension
        summed_tensor = torch.sum(stacked_tensors, dim=0)

        return summed_tensor

    def forward(
        self, domains: LatentsDomainGroupT, encodings: LatentsDomainGroupT
    ) -> dict[str, torch.Tensor]:
        """
        Compute keys and queries, match them with dot product and softmax.
        Does this twice, once with the static query and once with a dynamic query.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.
            encodings (`LatentsDomainGroupT`): Group of pre-fusion encodings.

        Returns:
            `dict[str, torch.Tensor]`: the attention scores for each domain in the group.
        """

        if self.gw_state is None:
            # tensor with only zeros as default (with specified domain and batch size)
            self.gw_state = torch.zeros(torch.rand(self.batch_size, self.domain_dim))

        # Encoding with pytorch
        keys = {
            domain: self.key_layers[domain](encoding)
            for domain, encoding in domains.items()
        }

        # This for training (cpu or gpu)
        device = group_device(domains)

        # Retrieve query
        query = self.query_layer(self.gw_state.to(device))

        # Calculate the attention scores
        static_attention_dict = self.calculate_attention_dict(keys, query)

        # Apply the attention scores to the encodings
        summed_tensor = self.fuse_weighted_encodings(encodings, static_attention_dict)

        # Retrieve query (now it is dependent on the new gw state)
        query = self.query_layer(summed_tensor.to(device))

        # Calculate the attention scores again
        dynamic_attention_dict = self.calculate_attention_dict(keys, query)

        return dynamic_attention_dict
