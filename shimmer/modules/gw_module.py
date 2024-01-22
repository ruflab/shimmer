from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import cast

import torch
from torch import nn

from shimmer.modules.domain import DomainModule
from shimmer.modules.vae import reparameterize


def get_n_layers(n_layers: int, hidden_dim: int):
    layers = []
    for _ in range(n_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return layers


class GWDecoder(nn.Sequential):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.n_layers = n_layers

        super().__init__(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )


class GWEncoder(GWDecoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        super().__init__(in_dim, hidden_dim, out_dim, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(super().forward(x))


class VariationalGWEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

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
    def __init__(
        self, domain_module: DomainModule, workspace_dim: int
    ) -> None:
        super().__init__()
        self.domain_module = domain_module
        self.workspace_dim = workspace_dim

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        ...


class GWModuleBase(nn.Module, ABC):
    def __init__(
        self, gw_interfaces: Mapping[str, GWInterfaceBase], workspace_dim: int
    ) -> None:
        super().__init__()
        # casting for LSP autocompletion
        self.gw_interfaces = cast(
            dict[str, GWInterfaceBase], nn.ModuleDict(gw_interfaces)
        )
        self.workspace_dim = workspace_dim

    def on_before_gw_encode_dcy(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the demi-cycle loss. Defaults to identity.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            the same mapping with updated representations
        """
        return {
            domain: self.gw_interfaces[
                domain
            ].domain_module.on_before_gw_encode_dcy(x[domain])
            for domain in x.keys()
        }

    def on_before_gw_encode_cy(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the cycle loss. Defaults to identity.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            the same mapping with updated representations
        """
        return {
            domain: self.gw_interfaces[
                domain
            ].domain_module.on_before_gw_encode_cy(x[domain])
            for domain in x.keys()
        }

    def on_before_gw_encode_tr(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the translation loss. Defaults to identity.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            the same mapping with updated representations
        """
        return {
            domain: self.gw_interfaces[
                domain
            ].domain_module.on_before_gw_encode_tr(x[domain])
            for domain in x.keys()
        }

    def on_before_gw_encode_cont(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Callback used before projecting the unimodal representations to the GW
        representation when computing the contrastive loss. Defaults to identity.
        Args:
            x: mapping of domain name to latent representation.
        Returns:
            the same mapping with updated representations
        """
        return {
            domain: self.gw_interfaces[
                domain
            ].domain_module.on_before_gw_encode_cont(x[domain])
            for domain in x.keys()
        }

    @abstractmethod
    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode the unimodal representations to the GW representation.
        Args:
            x: mapping of domain name to unimodal representation.
        Returns:
            GW representation
        """
        ...

    @abstractmethod
    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Decode the GW representation to the unimodal representations.
        Args:
            z: GW representation
            domains: iterable of domains to decode to. Defaults to all domains.
        Returns:
            dict of domain name to decoded unimodal representation.
        """
        ...

    @abstractmethod
    def translate(
        self, x: Mapping[str, torch.Tensor], to: str
    ) -> torch.Tensor:
        """
        Translate from one domain to another.
        Args:
            x: mapping of domain name to unimodal representation.
            to: domain to translate to.
        Returns:
            the unimodal representation of domain given by `to`.
        """
        ...

    @abstractmethod
    def cycle(
        self, x: Mapping[str, torch.Tensor], through: str
    ) -> dict[str, torch.Tensor]:
        """
        Cycle from one domain through another.
        Args:
            x: mapping of domain name to unimodal representation.
            through: domain to translate to.
        Returns:
            the unimodal representations cycles through the given domain.
        """
        ...


class GWInterface(GWInterfaceBase):
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

    def translate(
        self, x: Mapping[str, torch.Tensor], to: str
    ) -> torch.Tensor:
        return self.decode(self.encode(x), domains={to})[to]

    def cycle(
        self, x: Mapping[str, torch.Tensor], through: str
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.translate(
                {through: self.translate(x, through)}, domain
            )
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
            mean, log_uncertainty = self.gw_interfaces[domain].encode(
                x[domain]
            )
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
            mean, log_uncertainty = self.gw_interfaces[domain].encode(
                x[domain]
            )
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

    def translate(
        self, x: Mapping[str, torch.Tensor], to: str
    ) -> torch.Tensor:
        return self.decode(self.encode_mean(x), domains={to})[to]

    def cycle(
        self, x: Mapping[str, torch.Tensor], through: str
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.translate(
                {through: self.translate(x, through)}, domain
            )
            for domain in x.keys()
        }
