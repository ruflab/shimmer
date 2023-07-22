from collections.abc import Iterable, Mapping

import torch
from info_nce import info_nce
from torch import nn
from torch.nn.functional import mse_loss

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
        encoder_type: type[GWEncoder] | type[VariationalGWEncoder],
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
                domain: GWEncoder(
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

    def translate(
        self, x: Mapping[str, torch.Tensor], to: str
    ) -> torch.Tensor:
        raise NotImplementedError

    def cycle(
        self, x: Mapping[str, torch.Tensor], through: str
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def demi_cycle_loss(
        self, x: Mapping[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        domain_name = next(iter(x.keys()))
        x_recons = self.decode(self.encode(x), domains={domain_name})[
            domain_name
        ]
        return mse_loss(x_recons, x[domain_name], **kwargs)

    def cycle_loss(
        self, x: Mapping[str, torch.Tensor], through: str, **kwargs
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        for domain in x.keys():
            x_pred = self.decode(self.encode(x), domains={through})[through]
            x_recons = self.decode(
                self.encode({through: x_pred}), domains={domain}
            )[domain]

            losses[domain] = mse_loss(
                x_recons,
                x[domain],
                **kwargs,
            )
        return losses

    def translation_loss(
        self,
        x: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        to: str,
        **kwargs,
    ) -> torch.Tensor:
        z = self.encode(x)
        prediction = self.decode(z, domains={to})[to]
        return mse_loss(prediction, target, **kwargs)

    def contrastive_losses(
        self,
        x: Mapping[str, torch.Tensor],
        **kwargs,
    ) -> dict[frozenset[str], torch.Tensor]:
        losses: dict[frozenset[str], torch.Tensor] = {}
        for domain1 in x.keys():
            z1 = self.encode({domain1: x[domain1]})
            for domain2 in x.keys():
                key = frozenset([domain1, domain2])
                if domain1 == domain2 or key in losses:
                    continue

                z2 = self.encode({domain2: x[domain2]})
                losses[key] = info_nce(z1, z2, **kwargs)
        return losses


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
            GWEncoder,
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
            VariationalGWEncoder,
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

    def encode_mean(
        self,
        x: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.fusion_mechanism(self.encoded_distribution(x)[0])

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
