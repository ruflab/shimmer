import math
from collections.abc import Sequence

import torch
from torch import nn


def reparameterize(mean, logvar):
    std = logvar.mul(0.5).exp()
    eps = torch.randn_like(std)
    return eps.mul(std).add(mean)


def kl_divergence_loss(
    mean: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return kl


def gaussian_nll(mu, log_sigma, x):
    return (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * math.log(2 * math.pi)
    )


class VAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        beta: float = 1,
    ):
        super().__init__()

        assert beta >= 0

        self.beta = beta

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
        mean_z, _ = self.encoder(x)
        return mean_z

    def decode(self, z: torch.Tensor) -> Sequence[torch.Tensor] | torch.Tensor:
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor | Sequence[torch.Tensor]
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor | Sequence[torch.Tensor],
    ]:
        mean, logvar = self.encoder(x)
        z = reparameterize(mean, logvar)

        x_reconstructed = self.decoder(z)

        return (mean, logvar), x_reconstructed

    def reconstruction_loss(
        self, x_reconstructed: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return gaussian_nll(x_reconstructed, torch.tensor(0), x).sum()

    def get_losses(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (mean, logvar), x_reconstructed = self(x)
        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence = kl_divergence_loss(mean, logvar)
        total_loss = reconstruction_loss + self.beta * kl_divergence
        return reconstruction_loss, kl_divergence, total_loss
