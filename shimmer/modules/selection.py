from abc import ABC, abstractmethod

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

    def forward(
        self, domains: dict[str, torch.Tensor], gw_state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
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


class KQDynamicQSelection(SelectionBase):
    """
    Key-Query attention with a dynamic gw vector.
    """

    def __init__(self, batch_size: int, domain_dim: int, head_size: int):
        """
        Args:
            batch_size (`int`) : size of the batch
            domain_dim (`int`) : dimension of the input dims (assumed to be the same for now)
            head_size (`int`) : dimension of the key and query vectors.
        """
        super().__init__()
        self.batch_size = batch_size
        self.head_size = head_size
        self.query_layer = nn.Linear(domain_dim, head_size)
        self.key_layers = nn.ModuleDict(
            {
                "v_latents": nn.Linear(domain_dim, head_size),
                "attr": nn.Linear(domain_dim, head_size),
            }
        )
        # Start with a initial naive gw state
        self.gw_state: torch.Tensor | None = None

    def update_gw_state(self, gw_state: torch.Tensor) -> None:
        """
        Update gw state with an initial or updated value.

        Args:
            gw_state (`torch.Tensor`): the previous GW state
        """
        self.gw_state = gw_state

    # TODO: implement this
    def get_prefusion_encodings(
        self,
        domains: LatentsDomainGroupT,
        random: bool,
    ) -> dict[str, torch.Tensor]:
        """
        Get random prefusion encodings for each domain in the group.

        Args:
            domains (`LatentsDomainGroupT`): Group of unimodal latent representations.

        Returns:
            `dict[str, torch.Tensor]`: for each domain in the group, the prefusion
            encoding for each item in the batch.
        """
        # get domain nam
        if random:
            return {
                domain: torch.rand_like(encoding)
                for domain, encoding in domains.items()
            }
        else:
            # do something else
            pass

        multiple_domain_input = {
            "v_latents": torch.rand(batch_size, domain_dim),
            "attr": torch.rand(batch_size, domain_dim),
        }
        return multiple_domain_input

    def apply_attention_scores(
        self,
        prefusion_encodings: dict[str, torch.Tensor],
        attention_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Apply the attention scores on the prefusion encodings.

        Args:
            prefusion_encodings (`dict[str, torch.Tensor]`): the prefusion encodings for each domain.
            attention_dict (`dict[str, torch.Tensor]`): the attention scores for each domain.

        Returns:
            `dict[str, torch.Tensor]`: the prefusion encodings with the attention scores applied.
        """
        return {
            domain: attention_dict[domain] * prefusion_encodings[domain]
            for domain in prefusion_encodings.keys()
        }

    def apply_fusion(
        self, domains: LatentsDomainGroupT, attention_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Merge function to combine domains.

        Args:
        x (`LatentsDomainGroupT`): the group of latent representation.
            selection_score (`Mapping[str, torch.Tensor]`): attention scores to
                use to encode the reprensetation.
        Returns:
            `torch.Tensor`: The merged representation.
        """
        return torch.sum(
            torch.stack(
                [
                    attention_dict[domain] * domains[domain]
                    for domain in attention_dict.keys()
                ]
            ),
            dim=0,
        )

    def encode(
        self,
        x: LatentsDomainGroupT,
        # selection_scores: Mapping[str, torch.Tensor] | None = None,
    ) -> LatentsDomainGroupT:
        """Encode the unimodal latent representation `x` into the pre-fusion GW
        representations.

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.

        Returns:
            `torch.Tensor`: encoded and fused GW representation.
        """
    
        domains = {}
        bs = group_batch_size(x)
        device = group_device(x)
        # domainmodskeys are names of the domains
        for domain in x.keys():
            if domain in x:
                domains[domain] = x[domain]
            else:
                domains[domain] = torch.zeros(
                    bs, self.domain_mods[domain].latent_dim
                ).to(device)
        return {
            domain_name: self.gw_encoders[domain_name](domain)
            for domain_name, domain in domains.items()
        }

    # def calculate_gw_state_with_attention(self, attention_dict: dict) -> None:
    #     """
    #     Update the internal copy of the previous GW state with the attention scores.

    #     Args:
    #         gw_state (`torch.Tensor`): the previous GW state
    #         attention_dict (`dict`): the attention scores for each domain
    #     """

    #     # Need to check this (need to do something with fusion?)
    #     for domain, attention in attention_dict.items():
    #         new_state = self.gw_state * attention
    #     return new_state

    def forward(self, domains: LatentsDomainGroupT, encodings: ) -> dict[str, torch.Tensor]:
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

        # Encoding with pytorch
        keys = {
            domain: self.key_layers[domain](encoding)
            for domain, encoding in domains.items()
        }
        # print(f"keys: {keys}")

        # This for training (cpu or gpu)
        device = group_device(domains)

        # Retrieve query
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

        # Get the prefusion encodings
        # encoding = self.get_prefusion_encodings(domains, random=True)
        encoding = self.encode(domains)
        print(encoding)

        ## Apply attentionscores on the encodings (maybe this is the same as the fusion?)
        ## Apply fusion
        # apply_fusion(encodings)

        # Calculate the new gw state

        ## Do this again for dynamic attention
        # function: set prefusion encodings (sets gw state)
        # fuse again
        # update state
        # again dot products etc..

        return attention_dict
