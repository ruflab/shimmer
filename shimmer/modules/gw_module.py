from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import cast

import torch
from torch import nn

from shimmer.modules.domain import DomainModule
from shimmer.modules.losses import LatentsDomainGroupT
from shimmer.modules.vae import reparameterize


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(super().forward(x))


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


class GWInterfaceBase(nn.Module, ABC):
    """Base class for GWInterfaces.

    Interfaces encode and decode unimodal representation to the domain's GW
        representation (pre-fusion).

    This is an abstract class and should be implemented.
    For an implemented interface, see `GWInterface`.
    """

    def __init__(self, domain_module: DomainModule, workspace_dim: int) -> None:
        """
        Initialized the interface.

        Args:
            domain_module (`DomainModule`): Domain module to link.
            workspace_dim (`int`): dimension of the GW.
        """
        super().__init__()

        self.domain_module = domain_module
        """Domain module."""

        self.workspace_dim = workspace_dim
        """Dimension of the GW."""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode from the unimodal latent representation to the domain's GW
            representation (pre-fusion).

        Args:
            x (`torch.Tensor`): the domain's unimodal latent representation.

        Returns:
            `torch.Tensor`: the domain's pre-fusion GW representation.
        """
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode from the domain's pre-fusion GW to the unimodal latent representation.

        Args:
            z (`torch.Tensor`): the domain's pre-fusion GW representation.

        Returns:
            `torch.Tensor`: the domain's unimodal latent representation.
        """
        ...


class GWModuleBase(nn.Module, ABC):
    """Base class for GWModule.

    GWModule handle how to merge representations from the Interfaces and define
    some common operations in GW like cycles and translations.

    This is an abstract class and should be implemented.
    For an implemented interface, see `GWModule`.
    """

    def __init__(
        self, gw_interfaces: Mapping[str, GWInterfaceBase], workspace_dim: int
    ) -> None:
        """Initializes the GWModule.

        Args:
            gw_interfaces (`Mapping[str, GWInterfaceBase]`): mapping for each domain
                name to a `GWInterfaceBase` class which role is to encode/decode
                unimodal latent representations into a GW representation (pre fusion).
            workspace_dim (`int`): dimension of the GW.
        """
        super().__init__()

        # casting for LSP autocompletion
        self.gw_interfaces = cast(
            dict[str, GWInterfaceBase], nn.ModuleDict(gw_interfaces)
        )
        """the GWInterface"""

        self.workspace_dim = workspace_dim
        """dimension of the GW"""

    def on_before_gw_encode_dcy(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the demi-cycle loss. Defaults to identity.

        Args:
            x (`Mapping[str, torch.Tensor]`): mapping of domain name to
                latent representation.
        Returns:
            `dict[str, torch.Tensor]`: the same mapping with updated representations
        """
        return {
            domain: self.gw_interfaces[domain].domain_module.on_before_gw_encode_dcy(
                x[domain]
            )
            for domain in x.keys()
        }

    def on_before_gw_encode_cy(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the cycle loss. Defaults to identity.

        Args:
            x (`Mapping[str, torch.Tensor]`): mapping of domain name to
                latent representation.
        Returns:
            `dict[str, torch.Tensor]`: the same mapping with updated representations
        """
        return {
            domain: self.gw_interfaces[domain].domain_module.on_before_gw_encode_cy(
                x[domain]
            )
            for domain in x.keys()
        }

    def on_before_gw_encode_tr(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the translation loss. Defaults to identity.

        Args:
            x (`Mapping[str, torch.Tensor]`): mapping of domain name to
                latent representation.
        Returns:
            `dict[str, torch.Tensor]`: the same mapping with updated representations
        """
        return {
            domain: self.gw_interfaces[domain].domain_module.on_before_gw_encode_tr(
                x[domain]
            )
            for domain in x.keys()
        }

    def on_before_gw_encode_cont(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the contrastive loss. Defaults to identity.

        Args:
            x (`Mapping[str, torch.Tensor]`): mapping of domain name to
                latent representation.
        Returns:
            `dict[str, torch.Tensor]`: the same mapping with updated representations
        """
        return {
            domain: self.gw_interfaces[domain].domain_module.on_before_gw_encode_cont(
                x[domain]
            )
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
    ) -> dict[str, torch.Tensor]:
        """Decode the GW representation into given `domains`.

        Args:
            z (`torch.Tensor`): the GW representation.
            domains (`Iterable[str]`): iterable of domains to decode.

        Returns:
            `dict[str, torch.Tensor]`: the decoded unimodal representations.
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
    def cycle(self, x: LatentsDomainGroupT, through: str) -> dict[str, torch.Tensor]:
        """
        Cycle from one domain through another.

        Args:
            x (`LatentsDomainGroupT`): mapping of domain name
                to unimodal representation.
            through (`str`): intermediate domain of the cycle

        Returns:
            `torch.Tensor`: the unimodal representation of domain given by `to`.
        """
        ...


class GWInterface(GWInterfaceBase):
    """
    A implementation of `GWInterfaceBase` using `GWEncoder`  and `GWDecoder`.
    """

    def __init__(
        self,
        domain_module: DomainModule,
        workspace_dim: int,
        encoder_hidden_dim: int,
        encoder_n_layers: int,
        decoder_hidden_dim: int,
        decoder_n_layers: int,
    ) -> None:
        super().__init__(domain_module, workspace_dim)

        self.encoder = GWEncoder(
            domain_module.latent_dim,
            encoder_hidden_dim,
            workspace_dim,
            encoder_n_layers,
        )
        self.decoder = GWDecoder(
            workspace_dim,
            decoder_hidden_dim,
            domain_module.latent_dim,
            decoder_n_layers,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class GWModule(GWModuleBase):
    def fusion_mechanism(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Merge function used to combine domains.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            The merged representation
        """
        return torch.mean(torch.stack(list(x.values())), dim=0)

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.fusion_mechanism(
            {
                domain: self.gw_interfaces[domain].encode(x[domain])
                for domain in x.keys()
            }
        )

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.gw_interfaces[domain].decode(z)
            for domain in domains or self.gw_interfaces.keys()
        }

    def translate(self, x: Mapping[str, torch.Tensor], to: str) -> torch.Tensor:
        return self.decode(self.encode(x), domains={to})[to]

    def cycle(
        self, x: Mapping[str, torch.Tensor], through: str
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.translate({through: self.translate(x, through)}, domain)
            for domain in x.keys()
        }


class VariationalGWInterface(GWInterfaceBase):
    def __init__(
        self,
        domain_module: DomainModule,
        workspace_dim: int,
        encoder_hidden_dim: int,
        encoder_n_layers: int,
        decoder_hidden_dim: int,
        decoder_n_layers: int,
    ) -> None:
        super().__init__(domain_module, workspace_dim)

        self.encoder = VariationalGWEncoder(
            domain_module.latent_dim,
            encoder_hidden_dim,
            workspace_dim,
            encoder_n_layers,
        )
        self.decoder = GWDecoder(
            workspace_dim,
            decoder_hidden_dim,
            domain_module.latent_dim,
            decoder_n_layers,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class VariationalGWModule(GWModuleBase):
    def fusion_mechanism(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.stack(list(x.values())), dim=0)

    def encode(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        latents: dict[str, torch.Tensor] = {}
        for domain in x.keys():
            mean, log_uncertainty = self.gw_interfaces[domain].encode(x[domain])
            latents[domain] = reparameterize(mean, log_uncertainty)
            latents[domain] = mean
        return self.fusion_mechanism(latents)

    def encoded_distribution(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        means: dict[str, torch.Tensor] = {}
        log_uncertainties: dict[str, torch.Tensor] = {}
        for domain in x.keys():
            mean, log_uncertainty = self.gw_interfaces[domain].encode(x[domain])
            means[domain] = mean
            log_uncertainties[domain] = log_uncertainty
        return means, log_uncertainties

    def encode_mean(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.fusion_mechanism(self.encoded_distribution(x)[0])

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.gw_interfaces[domain].decode(z)
            for domain in domains or self.gw_interfaces.keys()
        }

    def translate(self, x: Mapping[str, torch.Tensor], to: str) -> torch.Tensor:
        return self.decode(self.encode_mean(x), domains={to})[to]

    def cycle(
        self, x: Mapping[str, torch.Tensor], through: str
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.translate({through: self.translate(x, through)}, domain)
            for domain in x.keys()
        }


class GWModuleFusion(GWModule):
    def fusion_mechanism(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Merge function used to combine domains.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            The merged representation
        """
        return torch.sum(torch.stack(list(x.values())), dim=0)

    def get_batch_size(self, x: Mapping[str, torch.Tensor]) -> int:
        for val in x.values():
            return val.size(0)
        raise ValueError("Got empty dict.")

    def get_device(self, x: Mapping[str, torch.Tensor]) -> torch.device:
        for val in x.values():
            return val.device
        raise ValueError("Got empty dict.")

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        domains = {}
        bs = self.get_batch_size(x)
        device = self.get_device(x)
        for domain in self.gw_interfaces.keys():
            if domain in x:
                domains[domain] = x[domain]
            else:
                domains[domain] = torch.zeros(
                    bs, self.gw_interfaces[domain].domain_module.latent_dim
                ).to(device)
        return self.fusion_mechanism(
            {
                domain_name: self.gw_interfaces[domain_name].encode(domain)
                for domain_name, domain in domains.items()
            }
        )
