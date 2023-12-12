from collections.abc import Iterable, Mapping

import torch
from torch import nn

from shimmer.modules.domain import DomainDescription
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


class GWModule(nn.Module):
    domain_descr: Mapping[str, DomainDescription]

    def fusion_mechanism(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def on_before_gw_encode_dcy(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.domain_descr[domain].module.on_before_gw_encode_dcy(
                x[domain]
            )
            for domain in x.keys()
        }

    def on_before_gw_encode_cy(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.domain_descr[domain].module.on_before_gw_encode_cy(
                x[domain]
            )
            for domain in x.keys()
        }

    def on_before_gw_encode_tr(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.domain_descr[domain].module.on_before_gw_encode_tr(
                x[domain]
            )
            for domain in x.keys()
        }

    def on_before_gw_encode_cont(
        self, x: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.domain_descr[domain].module.on_before_gw_encode_cont(
                x[domain]
            )
            for domain in x.keys()
        }

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def translate(
        self, x: Mapping[str, torch.Tensor], to: str
    ) -> torch.Tensor:
        raise NotImplementedError

    def cycle(
        self, x: Mapping[str, torch.Tensor], through: str
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def default_encoders(
    domain_descriptions: Mapping[str, DomainDescription], latent_dim: int
) -> dict[str, GWEncoder]:
    return {
        name: GWEncoder(
            domain.latent_dim,
            domain.encoder_hidden_dim,
            latent_dim,
            domain.encoder_n_layers,
        )
        for name, domain in domain_descriptions.items()
    }


def default_decoders(
    domain_descriptions: Mapping[str, DomainDescription], latent_dim: int
) -> dict[str, GWDecoder]:
    return {
        name: GWDecoder(
            domain.latent_dim,
            domain.decoder_hidden_dim,
            latent_dim,
            domain.decoder_n_layers,
        )
        for name, domain in domain_descriptions.items()
    }


class DeterministicGWModule(GWModule):
    def __init__(
        self,
        domain_descriptions: Mapping[str, DomainDescription],
        latent_dim: int,
        encoders: Mapping[str, nn.Module] | None = None,
        decoders: Mapping[str, nn.Module] | None = None,
    ):
        super().__init__()

        self.domains = set(domain_descriptions.keys())
        self.domain_descr = domain_descriptions
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleDict(
            encoders or default_encoders(domain_descriptions, latent_dim)
        )
        self.decoders = nn.ModuleDict(
            decoders or default_decoders(domain_descriptions, latent_dim)
        )

    def fusion_mechanism(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.stack(list(x.values())), dim=0)

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.fusion_mechanism(
            {domain: self.encoders[domain](x[domain]) for domain in x.keys()}
        )

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.decoders[domain](z)
            for domain in domains or self.domains
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


def default_var_encoders(
    domain_descriptions: Mapping[str, DomainDescription], latent_dim: int
) -> dict[str, VariationalGWEncoder]:
    return {
        name: VariationalGWEncoder(
            domain.latent_dim,
            domain.encoder_hidden_dim,
            latent_dim,
            domain.encoder_n_layers,
        )
        for name, domain in domain_descriptions.items()
    }


class VariationalGWModule(GWModule):
    def __init__(
        self,
        domain_descriptions: Mapping[str, DomainDescription],
        latent_dim: int,
        encoders: Mapping[str, nn.Module] | None = None,
        decoders: Mapping[str, nn.Module] | None = None,
    ):
        super().__init__()

        self.domains = set(domain_descriptions.keys())
        self.domain_descr = domain_descriptions
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleDict(
            encoders or default_var_encoders(domain_descriptions, latent_dim)
        )
        self.decoders = nn.ModuleDict(
            decoders or default_decoders(domain_descriptions, latent_dim)
        )

    def fusion_mechanism(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.stack(list(x.values())), dim=0)

    def encode(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        latents: dict[str, torch.Tensor] = {}
        for domain in x.keys():
            mean, log_uncertainty = self.encoders[domain](x[domain])
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
            mean, log_uncertainty = self.encoders[domain](x[domain])
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
            domain: self.decoders[domain](z)
            for domain in domains or self.domains
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
