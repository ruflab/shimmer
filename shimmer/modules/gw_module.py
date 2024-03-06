from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

import torch
from torch import Module, nn

from shimmer.modules.domain import DomainModule
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
    """A Decoder network used in GWInterfaces."""

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
    """An Encoder network used in GWInterfaces.

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


class VariationalGWEncoder(nn.Module):
    """A Variational flavor of encoder network used in GWInterfaces."""

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
        """input dimension"""

        self.hidden_dim = hidden_dim
        """hidden dimension"""

        self.out_dim = out_dim
        """output dimension"""

        self.n_layers = n_layers
        """number of hidden layers. The total number of layers
                will be `n_layers` + 2 (one before, one after)."""

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Tanh(),
        )

        self.uncertainty_level = nn.Parameter(torch.full((self.out_dim,), 3.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers(x), self.uncertainty_level.expand(x.size(0), -1)


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

    def on_before_gw_encode_dcy(self, x: LatentsDomainGroupT) -> LatentsDomainGroupDT:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the demi-cycle loss. Defaults to identity.

        Args:
            x (`LatentsDomainGroupT`): mapping of domain name to
                latent representation.
        Returns:
            `LatentsDomainGroupDT`: the same mapping with updated representations
        """
        return {
            domain: self.domain_mods[domain].on_before_gw_encode_dcy(x[domain])
            for domain in x.keys()
        }

    def on_before_gw_encode_cy(self, x: LatentsDomainGroupT) -> LatentsDomainGroupDT:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the cycle loss. Defaults to identity.

        Args:
            x (`LatentsDomainGroupT`): mapping of domain name to
                latent representation.
        Returns:
            `LatentsDomainGroupDT`: the same mapping with updated representations
        """
        return {
            domain: self.domain_mods[domain].on_before_gw_encode_cy(x[domain])
            for domain in x.keys()
        }

    def on_before_gw_encode_tr(self, x: LatentsDomainGroupT) -> LatentsDomainGroupDT:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the translation loss. Defaults to identity.

        Args:
            x (`LatentsDomainGroupT`): mapping of domain name to
                latent representation.
        Returns:
            `LatentsDomainGroupDT`: the same mapping with updated representations
        """
        return {
            domain: self.domain_mods[domain].on_before_gw_encode_tr(x[domain])
            for domain in x.keys()
        }

    def on_before_gw_encode_cont(self, x: LatentsDomainGroupT) -> LatentsDomainGroupDT:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the contrastive loss. Defaults to identity.

        Args:
            x (`LatentsDomainGroupT`): mapping of domain name to
                latent representation.
        Returns:
            `LatentsDomainGroupDT`: the same mapping with updated representations
        """
        return {
            domain: self.domain_mods[domain].on_before_gw_encode_cont(x[domain])
            for domain in x.keys()
        }

    @abstractmethod
    def encode(self, x: LatentsDomainGroupT) -> torch.Tensor:
        """Encode latent representations into the GW representation.

        Args:
            x (`LatentsDomainGroupT`): the input domain representations.

        Returns:
            `torch.Tensor`: the GW representations.
        """
        ...

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

    @abstractmethod
    def translate(self, x: LatentsDomainGroupT, to: str) -> torch.Tensor:
        """
        Translate from one domain to another.

        Args:
            x (`LatentsDomainGroupT`): mapping of domain name
                to unimodal representation.
            to (`str`): domain to translate to.
        Returns:
            `torch.Tensor`: the unimodal representation of domain given by `to`.
        """
        ...

    @abstractmethod
    def cycle(self, x: LatentsDomainGroupT, through: str) -> LatentsDomainGroupDT:
        """
        Cycle from one domain through another.

        Args:
            x (`LatentsDomainGroupT`): mapping of domain name
                to unimodal representation.
            through (`str`): intermediate domain of the cycle

        Returns:
            `LatentsDomainGroupDT`: the decoded unimodal representations after the
            cycle.
        """
        ...


class GWModule(GWModuleBase):
    """GW Module. Implements `GWModuleBase`."""

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        gw_encoders: Mapping[str, Module],
        gw_decoders: Mapping[str, Module],
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

        self.gw_encoders = nn.ModuleDict(gw_encoders)  # type: ignore
        """The module's encoders"""

        self.gw_decoders = nn.ModuleDict(gw_decoders)  # type: ignore
        """The module's decoders"""

    def fusion_mechanism(self, x: LatentsDomainGroupT) -> torch.Tensor:
        """
        Merge function used to combine domains.

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.
        Returns:
            `torch.Tensor`: The merged representation.
        """
        return torch.mean(torch.stack(list(x.values())), dim=0)

    def encode(self, x: LatentsDomainGroupT) -> torch.Tensor:
        """Encode the unimodal latent representation `x` into the GW representation.

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.

        Returns:
            `torch.Tensor`: encoded and fused GW representation.

        """
        return self.fusion_mechanism(
            {domain: self.gw_encoders[domain](x[domain]) for domain in x.keys()}
        )

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

    def translate(self, x: LatentsDomainGroupT, to: str) -> torch.Tensor:
        """Translate from multiple domains to one domain.

        Args:
            x (`LatentsDomainGroupT`): the group of latent representations
            to (`str`): the domain name to encode to

        Returns:
            `torch.Tensor`: the translated unimodal representation
                of the provided domain.
        """
        return self.decode(self.encode(x), domains={to})[to]

    def cycle(self, x: LatentsDomainGroupT, through: str) -> LatentsDomainGroupDT:
        """Do a full cycle from a group of representation through one domain.

        [Original domains] -> [GW] -> [through] -> [GW] -> [Original domains]

        Args:
            x (`LatentsDomainGroupT`): group of unimodal latent representation
            through (`str`): domain name to cycle through
        Returns:
            `LatentsDomainGroupDT`: group of unimodal latent representation after
                cycling.
        """
        return {
            domain: self.translate({through: self.translate(x, through)}, domain)
            for domain in x.keys()
        }


class VariationalGWModule(GWModuleBase):
    """Variational flavor of `GWModule`."""

    def __init__(
        self,
        domain_modules: Mapping[str, DomainModule],
        workspace_dim: int,
        gw_encoders: Mapping[str, Module],
        gw_decoders: Mapping[str, Module],
    ) -> None:
        """Initializes the VariationalGWModule.

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

        self.gw_encoders = nn.ModuleDict(gw_encoders)  # type: ignore
        """The module's encoders"""

        self.gw_decoders = nn.ModuleDict(gw_decoders)  # type: ignore
        """The module's decoders"""

    def fusion_mechanism(self, x: LatentsDomainGroupT) -> torch.Tensor:
        """Fusion of the pre-fusion GW representations.

        Args:
            x (`LatentsDomainGroupT`): pre-fusion GW representations.
        Returns:
            `torch.Tensor`: the merged GW representation.
        """
        return torch.mean(torch.stack(list(x.values())), dim=0)

    def encode(
        self,
        x: LatentsDomainGroupT,
    ) -> torch.Tensor:
        """Encode a unimodal latent group into a GW representation.
        The pre-fusion GW representation are sampled from the predicted distribution.

        Args:
            x (`LatentsDomainGroupT`): unimodal latent group.

        Returns:
            `torch.Tensor`: GW representation
        """
        latents: LatentsDomainGroupDT = {}
        for domain in x.keys():
            mean, log_uncertainty = self.gw_encoders[domain](x[domain])
            latents[domain] = reparameterize(mean, log_uncertainty)
        return self.fusion_mechanism(latents)

    def encoded_distribution(
        self,
        x: LatentsDomainGroupT,
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

    def encode_mean(
        self,
        x: LatentsDomainGroupT,
    ) -> torch.Tensor:
        """Encodes a unimodal latent group into a GW representation (post-fusion)
        using the mean value of the pre-fusion representations.

        Args:
            x (`LatentsDomainGroupT`): unimodal latent group.

        Returns:
            `torch.Tensor`: GW representation encoded using the mean of pre-fusion
                GW.
        """
        return self.fusion_mechanism(self.encoded_distribution(x)[0])

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

    def translate(self, x: LatentsDomainGroupT, to: str) -> torch.Tensor:
        """Translate a latent representation to a specified domain.

        Args:
            x (`LatentsDomainGroupT`): group of latent representations.
            to (`str`): domain name to translate to.

        Returns:
            `torch.Tensor`: translated unimodal representation in domain given in `to`.
        """
        return self.decode(self.encode_mean(x), domains={to})[to]

    def cycle(self, x: LatentsDomainGroupT, through: str) -> LatentsDomainGroupDT:
        """Do a full cycle from a group of representation through one domain.

        [Original domains] -> [GW] -> [through] -> [GW] -> [Original domains]

        Args:
            x (`LatentsDomainGroupT`): group of unimodal latent representation
            through (`str`): domain name to cycle through
        Returns:
            `LatentsDomainGroupDT`: group of unimodal latent representation after
                cycling.
        """
        return {
            domain: self.translate({through: self.translate(x, through)}, domain)
            for domain in x.keys()
        }


class GWModuleFusion(GWModule):
    def fusion_mechanism(self, x: LatentsDomainGroupT) -> torch.Tensor:
        """
        Merge function used to combine domains.

        Args:
            x (`LatentsDomainGroupT`): the group of latent representation.
        Returns:
            `torch.Tensor`: The merged representation.
        """
        return torch.sum(torch.stack(list(x.values())), dim=0)

    def encode(self, x: LatentsDomainGroupT) -> torch.Tensor:
        """Encode the unimodal latent representation `x` into the GW representation.

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
        return self.fusion_mechanism(
            {
                domain_name: self.gw_encoders[domain_name](domain)
                for domain_name, domain in domains.items()
            }
        )
