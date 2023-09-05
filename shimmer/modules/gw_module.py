from collections.abc import Iterable, Mapping

import torch
from torch import nn

from shimmer.modules.domain import DomainDescription
from shimmer.modules.global_workspace import GWEncoder, VariationalGWEncoder
from shimmer.modules.vae import reparameterize


class GWModule(nn.Module):
    def fusion_mechanism(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

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
        self.domains = set(domain_descriptions.keys())
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
        self.domains = set(domain_descriptions.keys())
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
            latents[domain] = reparameterize(mean, logvar)
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
