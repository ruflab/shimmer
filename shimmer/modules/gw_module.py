from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import cast

import torch
from torch import nn

from shimmer.modules.domain import DomainModule
from shimmer.modules.selection import SelectionBase
from shimmer.types import LatentsDomainGroupDT, LatentsDomainGroupT
from shimmer.utils import group_batch_size


def get_n_layers(n_layers: int, hidden_dim: int) -> list[nn.Module]:
    """
    Makes a list of `n_layers` `nn.Linear` layers with `nn.ReLU`.

    Args:
        n_layers (`int`): number of layers
        hidden_dim (`int`): size of the hidden dimension

    Returns:
        `list[nn.Module]`: list of linear and relu layers.
    """
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return layers


class GWDecoder(nn.Sequential):
    """A Decoder network for GWModules."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        """
        Initializes the decoder.

        Args:
            in_dim (`int`): input dimension
            hidden_dim (`int`): hidden dimension
            out_dim (`int`): output dimension
            n_layers (`int`): number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after).
        """

        self.in_dim = in_dim
        """input dimension"""

        self.hidden_dim = hidden_dim
        """hidden dimension"""

        self.out_dim = out_dim
        """output dimension"""

        self.n_layers = n_layers
        """
        number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after)."""

        super().__init__(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )


class GWEncoder(GWDecoder):
    """
    An Encoder network used in GWModules.

    This is similar to the decoder, but adds a tanh non-linearity at the end.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        """
        Initializes the encoder.

        Args:
            in_dim (`int`): input dimension
            hidden_dim (`int`): hidden dimension
            out_dim (`int`): output dimension
            n_layers (`int`): number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after).
        """
        super().__init__(in_dim, hidden_dim, out_dim, n_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)


class GWEncoderLinear(nn.Linear):
    """A linear Encoder network used in GWModules."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(super().forward(input))


class GWModuleBase(nn.Module, ABC):
    """
    Base class for GWModule.

    GWModule handles encoding, decoding the unimodal representations
    using the `gw_encoders` and`gw_decoders`, and define
    some common operations in GW like cycles and translations.

    This is an abstract class and should be implemented.
    For an implemented interface, see `GWModule`.
    """

    def __init__(
        self,
        domain_mods: Mapping[str, DomainModule],
        workspace_dim: int,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the GWModule.

        Args:
            domain_modules (`Mapping[str, DomainModule]`): the domain modules.
            workspace_dim (`int`): dimension of the GW.
        """
        super().__init__()

        self.domain_mods = domain_mods
        """The unimodal domain modules."""

        self.workspace_dim = workspace_dim
        """Dimension of the GW"""

    @abstractmethod
    def fuse(
        self, x: LatentsDomainGroupT, selection_scores: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Merge function used to combine domains.

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.
            selection_score (`Mapping[str, torch.Tensor]`): attention scores to
                use to encode the reprensetation.
        Returns:
            `torch.Tensor`: The merged representation.
        """
        ...

    @abstractmethod
    def encode(self, x: LatentsDomainGroupT) -> LatentsDomainGroupDT:
        """
        Encode the latent representation infos to the pre-fusion GW representation.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations

        Returns:
            `LatentsDomainGroupT`: pre-fusion GW representations
        """
        ...

    def encode_and_fuse(
        self, x: LatentsDomainGroupT, selection_module: SelectionBase
    ) -> torch.Tensor:
        """
        Encode the latent representation infos to the final GW representation.
        It combines the encode and fuse methods.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations
            selection_score (`Mapping[str, torch.Tensor]`): attention scores to
                use to encode the reprensetation.

        Returns:
            `torch.Tensor`: The merged representation.
        """
        encodings = self.encode(x)
        selection_scores = selection_module(x, encodings)
        return self.fuse(encodings, selection_scores)

    @abstractmethod
    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> LatentsDomainGroupDT:
        """
        Decode the GW representation into given `domains`.

        Args:
            z (`torch.Tensor`): the GW representation.
            domains (`Iterable[str]`): iterable of domains to decode.

        Returns:
            `LatentsDomainGroupDT`: the decoded unimodal representations.
        """
        ...


class GWModule(GWModuleBase):
    """GW nn.Module. Implements `GWModuleBase`."""

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        gw_encoders: Mapping[str, nn.Module],
        gw_decoders: Mapping[str, nn.Module],
    ) -> None:
        """
        Initializes the GWModule.

        Args:
            domain_modules (`Mapping[str, DomainModule]`): the domain modules.
            workspace_dim (`int`): dimension of the GW.
            gw_encoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a an torch.nn.Module class that encodes a
                unimodal latent representations into a GW representation (pre fusion).
            gw_decoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a an torch.nn.Module class that decodes a
                 GW representation to a unimodal latent representation.
        """
        super().__init__(domain_modules, workspace_dim)

        self.gw_encoders = nn.ModuleDict(gw_encoders)
        """The module's encoders"""

        self.gw_decoders = nn.ModuleDict(gw_decoders)
        """The module's decoders"""

    def fuse(
        self,
        x: LatentsDomainGroupT,
        selection_scores: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Merge function used to combine domains.

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.
            selection_score (`Mapping[str, torch.Tensor]`): attention scores to
                use to encode the reprensetation.
        Returns:
            `torch.Tensor`: The merged representation.
        """
        return torch.tanh(
            torch.sum(
                torch.stack(
                    [
                        selection_scores[domain].unsqueeze(1) * x[domain]
                        for domain in selection_scores
                    ]
                ),
                dim=0,
            )
        )

    def encode(self, x: LatentsDomainGroupT) -> LatentsDomainGroupDT:
        """
        Encode the latent representation infos to the pre-fusion GW representation.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations.

        Returns:
            `LatentsDomainGroupT`: pre-fusion representation
        """
        return {
            domain_name: self.gw_encoders[domain_name](domain)
            for domain_name, domain in x.items()
        }

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> LatentsDomainGroupDT:
        """
        Decodes a GW representation to multiple domains.

        Args:
            z (`torch.Tensor`): the GW representation
            domains (`Iterable[str] | None`): the domains to decode to. Defaults to
                use keys in `gw_interfaces` (all domains).
        Returns:
            `LatentsDomainGroupDT`: decoded unimodal representation
        """
        return {
            domain: self.gw_decoders[domain](z)
            for domain in domains or self.gw_decoders.keys()
        }


def compute_fusion_scores(
    score_1: torch.Tensor,
    score_2: torch.Tensor,
    sensitivity_1: float = 1.0,
    sensitivity_2: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Combine precision scores using std summation in quadrature

    The two scores should have the same dimension.

    Args:
        score_1 (`torch.Tensor`): First scores.
        score_2 (`torch.Tensor`): Second scores.
        sensitivity_1 (`float`): sensitivity for the first score
        sensitivity_2 (`float`): sensitivity for the second score
        eps (`float`): a value added to avoid numerical unstability.

    Returns:
        `torch.Tensor`: the combined scores
    """
    total_uncertainty = sensitivity_1 / (eps + score_1) + sensitivity_2 / (
        eps + score_2
    )
    final_scores = 1 / (eps + total_uncertainty)
    return final_scores / final_scores.sum(dim=0, keepdim=True)


class GWModuleBayesian(GWModule):
    """`GWModule` with a Bayesian based uncertainty prediction."""

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        gw_encoders: Mapping[str, nn.Module],
        gw_decoders: Mapping[str, nn.Module],
        sensitivity_selection: float = 1,
        sensitivity_precision: float = 1,
    ) -> None:
        """
        Initializes the GWModuleBayesian.

        Args:
            domain_modules (`Mapping[str, DomainModule]`): the domain modules.
            workspace_dim (`int`): dimension of the GW.
            gw_encoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a an torch.nn.Module class that encodes a
                unimodal latent representations into a GW representation (pre fusion).
            gw_decoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a an torch.nn.Module class that decodes a
                 GW representation to a unimodal latent representation.
            sensitivity_selection (`float`): sensivity coef $c'_1$
            sensitivity_precision (`float`): sensitivity coef $c'_2$
        """
        super().__init__(domain_modules, workspace_dim, gw_encoders, gw_decoders)

        self.precisions = cast(
            dict[str, torch.Tensor],
            nn.ParameterDict(
                {domain: torch.randn(workspace_dim) for domain in gw_encoders}
            ),
        )
        """Precision at the neuron level for every domain."""

        self.sensitivity_selection = sensitivity_selection
        self.sensitivity_precision = sensitivity_precision

    def get_precision(self, x: LatentsDomainGroupT) -> dict[str, torch.Tensor]:
        """
        Get the precision vector of given domain and batch

        Returns:
            `torch.Tensor`: batch of precision
        """
        batch_size = group_batch_size(x)
        precisions = (
            torch.stack([precision for precision in self.precisions.values()], dim=0)
            .softmax(dim=0)
            .unsqueeze(1)
            .expand(-1, batch_size, -1)
        )
        return {domain: precisions[k] for k, domain in enumerate(self.precisions)}

    def fuse(
        self,
        x: LatentsDomainGroupT,
        selection_scores: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Merge function used to combine domains.

        In the following, $D$ is the number of domains, $N$ the batch size, and $d$ the
        dimension of the Global Workspace.

        This function needs to merge two kind of scores:
        * the selection scores $a\\in [0,1]^{D\\times N}$;
        * the precision scores $b \\in [0,1]^{D\\times N \\times d}$.

        .. note::
            The precision score is obtained by predicting logits and using a softmax

        We can obtain associated uncertainties to the scores by introducing a std
        variable and using bayesian integration:

        $$a_k = \\frac{M_1}{\\sigma_k^2}$$
        where $M_1 = \\frac{1}{\\sum_{i=1}^D \\frac{1}{\\sigma_i^2}}$.

        Similarly,
        $$b_k = \\frac{M_2}{\\mu_k^2}$$
        where $M_2 = \\frac{1}{\\sum_{i=1}^D \\frac{1}{\\mu_i^2}}$.

        The we can sum the variances to obtain the final uncertainty (squared) $\\xi$:
        $$\\xi_k^2 = c_1 \\sigma_k^2 + c_2 \\mu_k^2$$

        which, in terms of $a_k$ and $b_k$ yields:
        $$\\xi_k^2 = \\frac{c'_1}{a_k} + \\frac{c'_2}{b_k}$$
        where $c'_1 = c_1 \\cdot M_1$ and $c'_2 = c_2 \\cdot M_2$.

        Finally, the finale combined coefficient is
        $$\\lambda_k = \\frac{M_3}{\\frac{c'_1}{a_k} + \\frac{c'_2}{b_k}}$$
        where
        $$M_3 = \\frac{1}{\\sum_{i=1}^D
            \\frac{1}{\\frac{c'_1}{a_i} + \\frac{c'_2}{b_i}}}$$

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.
            selection_score (`Mapping[str, torch.Tensor]`): attention scores to
                use to encode the reprensetation.
        Returns:
            `torch.Tensor`: The merged representation.
        """
        scores: list[torch.Tensor] = []
        precisions: list[torch.Tensor] = []
        domains: list[torch.Tensor] = []
        precision_scores = self.get_precision(x)
        for domain, score in selection_scores.items():
            scores.append(score)
            precisions.append(precision_scores[domain])
            domains.append(x[domain])
        return torch.tanh(
            torch.sum(
                torch.stack(precisions) * torch.stack(domains),
                dim=0,
            )
        )
