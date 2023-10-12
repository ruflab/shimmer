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


class GWEncoder(nn.Sequential):
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

        super(GWEncoder, self).__init__(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )


class VariationalGWEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        super(VariationalGWEncoder, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.confidence_level = nn.Parameter(torch.full((self.out_dim,), 5.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers(x), self.confidence_level


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


class DeterministicGWModule(GWModule):
    def __init__(
        self,
        domain_descriptions: Mapping[str, DomainDescription],
        latent_dim: int,
    ):
        super().__init__()

        self.domains = set(domain_descriptions.keys())
        self.domain_descr = domain_descriptions
        self.latent_dim = latent_dim

        self.input_dim: dict[str, int] = {}
        self.encoder_hidden_dim: dict[str, int] = {}
        self.encoder_n_layers: dict[str, int] = {}
        self.decoder_hidden_dim: dict[str, int] = {}
        self.decoder_n_layers: dict[str, int] = {}

        for name, domain in domain_descriptions.items():
            self.input_dim[name] = domain.latent_dim
            self.encoder_hidden_dim[name] = domain.encoder_hidden_dim
            self.encoder_n_layers[name] = domain.encoder_n_layers
            self.decoder_hidden_dim[name] = domain.decoder_hidden_dim
            self.decoder_n_layers[name] = domain.decoder_n_layers

        self.encoders = nn.ModuleDict(
            {
                domain: GWEncoder(
                    self.input_dim[domain],
                    self.encoder_hidden_dim[domain],
                    self.latent_dim,
                    self.encoder_n_layers[domain],
                )
                for domain in self.domains
            }
        )
        self.decoders = nn.ModuleDict(
            {
                domain: GWEncoder(
                    self.latent_dim,
                    self.decoder_hidden_dim[domain],
                    self.input_dim[domain],
                    self.decoder_n_layers[domain],
                )
                for domain in self.domains
            }
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


class VariationalGWModule(GWModule):
    def __init__(
        self,
        domain_descriptions: Mapping[str, DomainDescription],
        latent_dim: int,
    ):
        super().__init__()

        self.domains = set(domain_descriptions.keys())
        self.domain_descr = domain_descriptions
        self.latent_dim = latent_dim

        self.input_dim: dict[str, int] = {}
        self.encoder_hidden_dim: dict[str, int] = {}
        self.encoder_n_layers: dict[str, int] = {}
        self.decoder_hidden_dim: dict[str, int] = {}
        self.decoder_n_layers: dict[str, int] = {}

        for name, domain in domain_descriptions.items():
            self.input_dim[name] = domain.latent_dim
            self.encoder_hidden_dim[name] = domain.encoder_hidden_dim
            self.encoder_n_layers[name] = domain.encoder_n_layers
            self.decoder_hidden_dim[name] = domain.decoder_hidden_dim
            self.decoder_n_layers[name] = domain.decoder_n_layers

        self.encoders = nn.ModuleDict(
            {
                domain: VariationalGWEncoder(
                    self.input_dim[domain],
                    self.encoder_hidden_dim[domain],
                    self.latent_dim,
                    self.encoder_n_layers[domain],
                )
                for domain in self.domains
            }
        )
        self.decoders = nn.ModuleDict(
            {
                domain: GWEncoder(
                    self.latent_dim,
                    self.decoder_hidden_dim[domain],
                    self.input_dim[domain],
                    self.decoder_n_layers[domain],
                )
                for domain in self.domains
            }
        )

    def fusion_mechanism(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.stack(list(x.values())), dim=0)

    def encode(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        latents: dict[str, torch.Tensor] = {}
        for domain in x.keys():
            mean, logvar = self.encoders[domain](x[domain])
            # latents[domain] = reparameterize(mean, logvar)
            latents[domain] = mean
        return self.fusion_mechanism(latents)

    def encoded_distribution(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        means: dict[str, torch.Tensor] = {}
        logvars: dict[str, torch.Tensor] = {}
        for domain in x.keys():
            mean, logvar = self.encoders[domain](x[domain])
            means[domain] = mean
            logvars[domain] = logvar
        return means, logvars

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
