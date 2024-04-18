from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import cast

import torch
from torch import nn

from shimmer.modules.domain import DomainModule
from shimmer.modules.selection import SelectionBase
from shimmer.types import LatentsDomainGroupDT, LatentsDomainGroupT


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


class GWModuleWithConfidence(GWModule):
    """`GWModule` with confidence information."""

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        gw_encoders: Mapping[str, nn.Module],
        gw_decoders: Mapping[str, nn.Module],
    ) -> None:
        """
        Initializes the GWModuleWithConfidence.

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
        super().__init__(domain_modules, workspace_dim, gw_encoders, gw_decoders)

        self.precisions = cast(
            dict[str, torch.Tensor],
            nn.ParameterDict(
                {domain: torch.randn(workspace_dim) for domain in gw_encoders}
            ),
        )
        """Precision at the neuron level for every domain."""

    def get_precision(self, domain: str, x: torch.Tensor) -> torch.Tensor:
        """
        Get the precision vector of given domain and batch

        Args:
            domain (`str`):
            x (`torch.Tensor`): batch of inputs

        Returns:
            `torch.Tensor`: batch of precision
        """
        return self.precisions[domain].unsqueeze(0).expand(x.size(0), -1)

    def _fuse_and_scores(
        self,
        x: LatentsDomainGroupT,
        selection_scores: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Merge function used to combine domains and return fusion scores.

        This is used for testing purposes.

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
        for domain, score in selection_scores.items():
            scores.append(score)
            precisions.append(self.get_precision(domain, x[domain]))
            domains.append(x[domain])
        # Size: D x N x d  (D: number of domains, d: workspace dim)
        precision_scores = torch.softmax(torch.stack(precisions), dim=0)
        scores_ = torch.stack(scores)  # Size: D x N (N: batch size)
        final_scores = scores_.unsqueeze(-1) * precision_scores  # Size D x N x d
        final_scores = final_scores / final_scores.sum(dim=0, keepdim=True)

        return torch.tanh(
            torch.sum(final_scores * torch.stack(domains), dim=0)
        ), final_scores

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
        * the selection scores $s\\in [0,1]^{D\\times N}$;
        * the precision scores $\\mu \\in R_+^{D\\times N \\times d}$.

        To combine this score with the selection scores, we multiply the two
        scores:
        $$\\lambda = \\frac{s * \\text{softmax}(\\mu)}{M} \\in [0,1]^{D \\times d}$$

        And for domain $k$, batch element $i$ and workspace neuron $j$:
        $$S_{k,i,j} =
            \\frac{s_{k,i} e^{\\mu_{k,i,j}}}{M_{i,j}(\\sum_k e^{\\mu_{k,i,j})}$$

        To select the normalization coef $M_{i,j}$, we use the following requirement:
        $$\\sum_k S_{k,i,j} = 1$$

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.
            selection_score (`Mapping[str, torch.Tensor]`): attention scores to
                use to encode the reprensetation.
        Returns:
            `torch.Tensor`: The merged representation.
        """
        return self._fuse_and_scores(x, selection_scores)[0]
