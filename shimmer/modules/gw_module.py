from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

import torch
from torch import nn

from shimmer.modules.domain import DomainModule
from shimmer.modules.selection import SingleDomainSelection
from shimmer.modules.vae import reparameterize
from shimmer.types import LatentsDomainGroupDT, LatentsDomainGroupT
from shimmer.utils import group_batch_size, group_device


def get_n_layers(n_layers: int, hidden_dim: int) -> list[nn.Module]:
    """Makes a list of `n_layers` `nn.Linear` layers with `nn.ReLU`.

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
        """Initializes the decoder.

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
        """number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after)."""

        super().__init__(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )


class GWEncoder(GWDecoder):
    """An Encoder network used in GWModules.

    This is similar to the decoder, but adds a tanh non-linearity at the end.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        """Initializes the encoder.

        Args:
            in_dim (`int`): input dimension
            hidden_dim (`int`): hidden dimension
            out_dim (`int`): output dimension
            n_layers (`int`): number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after).
        """
        super().__init__(in_dim, hidden_dim, out_dim, n_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(super().forward(input))


class GWEncoderLinear(nn.Linear):
    """A linear Encoder network used in GWModules."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.tanh(super().forward(input))


class GWEncoderWithUncertainty(nn.Module):
    """An encoder network with uncertainty."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        """Initializes the encoder.

        Args:
            in_dim (`int`): input dimension
            hidden_dim (`int`): hidden dimension
            out_dim (`int`): output dimension
            n_layers (`int`): number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after).
        """
        super().__init__()

        self.in_dim = in_dim
        """Input dimension"""

        self.hidden_dim = hidden_dim
        """Hidden dimension"""

        self.out_dim = out_dim
        """Output dimension"""

        self.n_layers = n_layers
        """Number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after)."""

        self.layers = GWEncoder(
            self.in_dim, self.hidden_dim, self.out_dim, self.n_layers
        )

        self.log_uncertainty_level = nn.Parameter(torch.zeros(self.out_dim))
        """Log of the uncertainty level of the model."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes the latent unimodal representation into the pre-fusion
        GW representation.

        Args:
            x (`torch.Tensor`): the latent unimodal representation

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: pre-fusion representation and
            uncertainty level.
        """
        return self.layers(x), self.log_uncertainty_level.expand(x.size(0), -1)


class GWModuleBase(nn.Module, ABC):
    """Base class for GWModule.

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
        """Initializes the GWModule.

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
    def encode(self, x: LatentsDomainGroupT) -> LatentsDomainGroupT:
        """Encode the latent representation infos to the pre-fusion GW representation.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations

        Returns:
            `LatentsDomainGroupT`: pre-fusion GW representations
        """
        ...

    def encode_and_fuse(
        self, x: LatentsDomainGroupT, selection_scores: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode the latent representation infos to the final GW representation.
        It combines the encode and fuse methods.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations
            selection_score (`Mapping[str, torch.Tensor]`): attention scores to
                use to encode the reprensetation.

        Returns:
            `torch.Tensor`: The merged representation.
        """
        return self.fuse(self.encode(x), selection_scores)

    @abstractmethod
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


class GWModule(GWModuleBase):
    """GW nn.Module. Implements `GWModuleBase`."""

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        gw_encoders: Mapping[str, nn.Module],
        gw_decoders: Mapping[str, nn.Module],
    ) -> None:
        """Initializes the GWModule.

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
        selection_scores: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Merge function used to combine domains.

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.
            selection_score (`Mapping[str, torch.Tensor] | None`): attention scores to
                use to encode the reprensetation.
        Returns:
            `torch.Tensor`: The merged representation.
        """
        return torch.sum(torch.stack(list(x.values())), dim=0)

    def encode_and_fuse(
        self,
        x: LatentsDomainGroupT,
        selection_scores: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.fuse(self.encode(x), selection_scores)

    def encode(self, x: LatentsDomainGroupT) -> LatentsDomainGroupT:
        """Encode the latent representation infos to the pre-fusion GW representation.

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
        """Decodes a GW representation to multiple domains.

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


class GWModuleWithUncertainty(GWModuleBase):
    """`GWModule` with uncertainty information."""

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        gw_encoders: Mapping[str, nn.Module],
        gw_decoders: Mapping[str, nn.Module],
    ) -> None:
        """Initializes the GWModuleWithUncertainty.

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
        selection_scores: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Fusion of the pre-fusion GW representations.

        Args:
            x (`LatentsDomainGroupT`): pre-fusion GW representations.
            selection_score (`Mapping[str, torch.Tensor] | None`): attention scores to
                use to encode the reprensetation.
        Returns:
            `torch.Tensor`: the merged GW representation.
        """
        return torch.sum(torch.stack(list(x.values())), dim=0)

    def encode(self, x: LatentsDomainGroupT) -> LatentsDomainGroupT:
        """Encode the latent representation infos to the pre-fusion GW representation.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations.

        Returns:
            `LatentsDomainGroupT`: pre-fusion representations
        """
        return {
            domain_name: reparameterize(*self.gw_encoders[domain_name](domain))
            for domain_name, domain in x.items()
        }

    def encode_and_fuse(
        self,
        x: LatentsDomainGroupT,
        selection_scores: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.fuse(self.encode(x), selection_scores)

    def encoded_distribution(
        self, x: LatentsDomainGroupT
    ) -> tuple[LatentsDomainGroupDT, LatentsDomainGroupDT]:
        """Encode a unimodal latent group into a pre-fusion GW distributions.
        The pre-fusion GW representation are the mean of the predicted distribution.

        Args:
            x (`LatentsDomainGroupT`): unimodal latent group

        Returns:
            `tuple[LatentsDomainGroupDT, LatentsDomainGroupDT]`: means and "log
                 uncertainty" of the pre-fusion representations.
        """
        means: LatentsDomainGroupDT = {}
        log_uncertainties: LatentsDomainGroupDT = {}
        for domain in x.keys():
            mean, log_uncertainty = self.gw_encoders[domain](x[domain])
            means[domain] = mean
            log_uncertainties[domain] = log_uncertainty
        return means, log_uncertainties

    def encoded_mean(self, x: LatentsDomainGroupT) -> torch.Tensor:
        """Encodes a unimodal latent group into a GW representation (post-fusion)
        using the mean value of the pre-fusion representations.

        Args:
            x (`LatentsDomainGroupT`): unimodal latent group.
            selection_score (`Mapping[str, torch.Tensor] | None`): attention scores to
                use to encode the reprensetation.

        Returns:
            `torch.Tensor`: GW representation encoded using the mean of pre-fusion GW.
        """
        return self.fuse(self.encoded_distribution(x)[0])

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> LatentsDomainGroupDT:
        """Decodes a GW representation into specified unimodal latent representation.

        Args:
            z (`torch.Tensor`): the GW representation.
            domains (`Iterable[str] | None`): List of domains to decode to. Defaults to
                use keys in `gw_interfaces` (all domains).

        Returns:
            `LatentsDomainGroupDT`: decoded unimodal representations.
        """
        return {
            domain: self.gw_decoders[domain](z)
            for domain in domains or self.gw_decoders.keys()
        }


class GWModuleWithSelection(GWModule):
    """GWModule version that uses the selection module."""

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        selection_mod: SingleDomainSelection,
        gw_encoders: Mapping[str, nn.Module],
        gw_decoders: Mapping[str, nn.Module],
    ) -> None:
        """Initializes the GWModule.

        Args:
            domain_modules (`Mapping[str, DomainModule]`): the domain modules.
            workspace_dim (`int`): dimension of the GW.
            selection_mod (`SingleDomainSelection`): selection module that selects
                only one domain at a time.
            gw_encoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a an torch.nn.Module class that encodes a
                unimodal latent representations into a GW representation (pre fusion).
            gw_decoders (`Mapping[str, torch.nn.Module]`): mapping for each domain
                name to a an torch.nn.Module class that decodes a
                 GW representation to a unimodal latent representation.
        """
        super().__init__(domain_modules, workspace_dim, gw_encoders, gw_decoders)

        self.selection_mod = selection_mod
        """Selection module"""

    def encode(self, x: LatentsDomainGroupT) -> LatentsDomainGroupT:
        """Encode the latent representation infos to the pre-fusion GW representation.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations.

        Returns:
            `LatentsDomainGroupT`: pre-fusion representations
        """
        return {
            domain_name: self.gw_encoders[domain_name](domain)
            for domain_name, domain in x.items()
        }


class GWModuleFusion(GWModuleBase):
    """
    GWModule used for fusion.
    """

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        gw_encoders: Mapping[str, nn.Module],
        gw_decoders: Mapping[str, nn.Module],
    ) -> None:
        """Initializes the GWModule Fusion.

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
        return torch.sum(torch.stack(list(x.values())), dim=0)

    def encode(self, x: LatentsDomainGroupT) -> LatentsDomainGroupT:
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
        for domain in self.domain_mods.keys():
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

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> LatentsDomainGroupDT:
        """Decodes a GW representation to multiple domains.

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
