from collections.abc import Iterable, Mapping

import torch
from torch import nn

from shimmer.modules.vae import reparameterize


def get_n_layers(n_layers: int, hidden_dim: int):
    layers = []
    for _ in range(n_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return layers


class _Encoder(nn.Module):
    pass


class DeterministicEncoder(_Encoder):
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

        super(DeterministicEncoder, self).__init__(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )


class VariationalEncoder(_Encoder):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
    ):
        super(VariationalEncoder, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
        )
        self.mean_layer = nn.Linear(self.hidden_dim, self.out_dim)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.layers(x)
        return self.mean_layer(z), self.logvar_layer(z)


class GlobalWorkspace(nn.Module):
    def __init__(
        self,
        domains: Iterable[str],
        encoder_type: type[_Encoder],
        latent_dim: int,
        input_dim: Mapping[str, int],
        encoder_hidden_dim: Mapping[str, int],
        encoder_n_layers: Mapping[str, int],
        decoder_hidden_dim: Mapping[str, int],
        decoder_n_layers: Mapping[str, int],
    ) -> None:
        super().__init__()

        self.domains = domains
        self.latent_dim = latent_dim

        self.input_dim = input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_n_layers = encoder_n_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_n_layers = decoder_n_layers

        self.encoders = nn.ModuleDict(
            {
                domain: encoder_type(
                    self.input_dim[domain],
                    self.encoder_hidden_dim[domain],
                    self.latent_dim,
                    self.encoder_n_layers[domain],
                )
                for domain in domains
            }
        )
        self.decoders = nn.ModuleDict(
            {
                domain: DeterministicEncoder(
                    self.latent_dim,
                    self.decoder_hidden_dim[domain],
                    self.input_dim[domain],
                    self.decoder_n_layers[domain],
                )
                for domain in domains
            }
        )

    def fusion_mechanism(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.stack(list(x.values())), dim=0)

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def decode(
        self, z: torch.Tensor, domains: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        return {
            domain: self.decoders[domain](z)
            for domain in domains or self.domains
        }

    def translate(self, x: Mapping[str, torch.Tensor], to: str):
        return self.decode(self.encode(x), domains={to})[to]

    def cycle(self, x: Mapping[str, torch.Tensor], through: str):
        return {
            domain: self.translate(
                {through: self.translate(x, through)}, domain
            )
            for domain in x.keys()
        }


class DeterministicGlobalWorkspace(GlobalWorkspace):
    def __init__(
        self,
        domains: Iterable[str],
        latent_dim: int,
        input_dim: Mapping[str, int],
        encoder_hidden_dim: Mapping[str, int],
        encoder_n_layers: Mapping[str, int],
        decoder_hidden_dim: Mapping[str, int],
        decoder_n_layers: Mapping[str, int],
    ) -> None:
        super().__init__(
            domains,
            DeterministicEncoder,
            latent_dim,
            input_dim,
            encoder_hidden_dim,
            encoder_n_layers,
            decoder_hidden_dim,
            decoder_n_layers,
        )

    def encode(self, x: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.fusion_mechanism(
            {domain: self.encoders[domain](x[domain]) for domain in x.keys()}
        )


class VariationalGlobalWorkspace(GlobalWorkspace):
    def __init__(
        self,
        domains: Iterable[str],
        latent_dim: int,
        input_dim: Mapping[str, int],
        encoder_hidden_dim: Mapping[str, int],
        encoder_n_layers: Mapping[str, int],
        decoder_hidden_dim: Mapping[str, int],
        decoder_n_layers: Mapping[str, int],
    ) -> None:
        super().__init__(
            domains,
            VariationalEncoder,
            latent_dim,
            input_dim,
            encoder_hidden_dim,
            encoder_n_layers,
            decoder_hidden_dim,
            decoder_n_layers,
        )

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

    def translate_mean(self, x: Mapping[str, torch.Tensor], to: str):
        mean, _ = self.encoded_distribution(x)
        return self.decode(self.fusion_mechanism(mean), domains={to})[to]

    def cycle_mean(self, x: Mapping[str, torch.Tensor], through: str):
        return {
            domain: self.translate_mean(
                {through: self.translate_mean(x, through)}, domain
            )
            for domain in x.keys()
        }
