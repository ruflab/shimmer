from collections.abc import Mapping

import torch
from torch import nn


def get_n_layers(n_layers: int, hidden_dim: int):
    layers = []
    for _ in range(n_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    return layers


class Encoder(nn.Sequential):
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

        super(Encoder, self).__init__(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            *get_n_layers(n_layers, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.out_dim),
        )


class GlobalWorkspace(nn.Module):
    def __init__(
        self,
        domains: set[str],
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
                domain: Encoder(
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
                domain: Encoder(
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
        return self.fusion_mechanism(
            {domain: self.encoders[domain](x[domain]) for domain in x.keys()}
        )

    def decode(
        self, z: torch.Tensor, domains: set[str] | None = None
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
