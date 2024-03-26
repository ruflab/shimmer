from abc import ABC, abstractmethod

import torch

from shimmer.types import LatentsDomainGroupT


class SelectionBase(torch.nn.Module, ABC):
<<<<<<< HEAD
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

    def forward(
        self, domains: LatentsDomainGroupT, gw_state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        keys = {
            domain: self.key_layers[domain](encoding)
            for domain, encoding in domains.items()
        }

        device = gw_state.device
        query = self.query_layer(gw_state.to(device))

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
=======
    @abstractmethod
    def forward(
        self, domains: LatentsDomainGroupT, gw_state: torch.Tensor
    ) -> dict[str, torch.Tensor]: ...






#todo : make this return a dict of attention scores. and move it into randomattention.
def sample_scaling_factors(
    binary_scaling_prob: float,
    batch_size: int,
    temperature: float,
    device: torch.device,
):
    """
    Args:
        binary_scaling_prob (`float`): Should be between 0 and 1.
        batch_size (`int`):
        temperature (`float`): Should be greater than 0.
        device (`torch.device`):
    """
    assert 0 <= binary_scaling_prob <= 1

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> LatentsDomainGroupDT:
        """Decode the GW representation into given `domains`.

        Args:
            z (`torch.Tensor`): the GW representation.
            domains (`Iterable[str]`): iterable of domains to decode.

        Returns:
            `LatentsDomainGroupDT`: the decoded unimodal representations.
        """
        ...
    binary_mask = torch.rand(batch_size) < binary_scaling_prob

    binary_factors = torch.randint(0, 2, (batch_size,)).float()
    binary_softmax = torch.stack([binary_factors, 1 - binary_factors], dim=1)

    uniform_samples = torch.rand(batch_size)
    uniform_for_softmax = torch.stack([uniform_samples, 1 - uniform_samples], dim=1)

    uniform_softmax = F.softmax(uniform_for_softmax * temperature, dim=1)

    scaling_factors = torch.where(
        binary_mask.unsqueeze(-1), binary_softmax, uniform_softmax
    ).to(device)

    binary_indices = torch.where(binary_mask)[0]
    softmax_indices = torch.where(~binary_mask)[0]

    binary_scaling_factors = scaling_factors[binary_indices]
    softmax_scaling_factors = scaling_factors[softmax_indices]

    return {
        "binary": (
            binary_scaling_factors[:, 0],
            binary_scaling_factors[:, 1],
            binary_indices,
        ),
        "softmax": (
            softmax_scaling_factors[:, 0],
            softmax_scaling_factors[:, 1],
            softmax_indices,
        ),
    }

>>>>>>> 380f5f2 (I've no idea why rebase didn't create selection.py but I'm creating it here)


class RandomSelection(SelectionBase):
    def __init__(self, binary_proportion, temperature):
        super().__init__()
        self.binary_proportion = binary_proportion
        self.temperature = temperature

<<<<<<< HEAD
    def forward(
        self, domains: LatentsDomainGroupT, gw_states: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pass
=======
    def forward(self, domains: LatentsDomainGroupT, gw_states: torch.Tensor) -> dict[str, float]:
        return sample_scaling_factors(self.binary_proportion, gw_states.shape[0], self.temperature, gw_states.device)













class KQAttentionOnePass(SelectionBase):
    def __init__(self, domain_dim, head_size):
        super().__init__()
        self.head_size = head_size
        self.query_layer = nn.Linear(domain_dim, head_size)
        self.key_layers = nn.ModuleDict({
            'v_latents': nn.Linear(domain_dim, head_size),
            'attr': nn.Linear(domain_dim, head_size)
        })
        self.gw_vector = torch.randn(domain_dim)  # Fixed random global workspace vector
        self.attention_scores = None  # Attribute to store the latest attention scores

    def forward(self, domain_encodings: LatentsDomainGroupT, gw_states: torch.Tensor) -> LatentsDomainGroupT:
        #TODO : check if this actually works !

        # Initialize key_layers if not already done
        keys = {domain: self.key_layers[domain](encoding) for domain, encoding in domain_encodings.items()}

        device = next(iter(keys.values())).device

        query = self.query_layer(self.gw_vector.to(device))

        # Calculate dot products for each domain
        dot_products = [torch.sum(key_tensor * query, dim=1) for key_tensor in keys.values()]

        # Organize the dot products into a 2D tensor [number_of_domains, 2]
        dot_products_tensor = torch.stack(dot_products)

        attention_scores = torch.softmax(dot_products_tensor,dim=0)

        # Scale the input latents by attention scores and print before and after scaling for the first two pairs
        weighted_encodings = {}
        for idx, (domain, encoding) in enumerate(domain_encodings.items()):
            # Reshape attention scores to match the encoding shape for broadcasting
            scaled_latent = encoding * attention_scores[idx].unsqueeze(1).expand_as(encoding)
            weighted_encodings[domain] = scaled_latent

        return weighted_encodings




class BinaryAttention(SelectionBase):
    def __init__(self, domain_dim, head_size):
        super().__init__()

        weights = {}
        #unsure what is meant by "1 if domain is here, 0 otherwise" -- how do we get the domain key if it's not there ?
        def forward(self, domain_encodings: LatentsDomainGroupT) -> LatentsDomainGroupT:
            for (domain, encoding) in domain_encodings.items():
                weights[domain] = torch.ones(encoding.shape[0])

            return weights

>>>>>>> 380f5f2 (I've no idea why rebase didn't create selection.py but I'm creating it here)
