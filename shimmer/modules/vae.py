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

    def encode(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        mean_z, _ = self.encoder(x)
        return mean_z

    def decode(self, z: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.decoder(z)

    def forward(
        self, x: Sequence[torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
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


class RAEEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        ae_dim: int = 1028,
        z_dim: int = 16,
        kernel_dim: int = 4,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.dims = [
            ae_dim // (2**i) for i in reversed(range(4))
        ]  # 1 2 4 8 # 32 64 128 256

        self.kernel_dim = kernel_dim
        self.padding = padding
        self.dims[-1] = ae_dim
        self.use_batchnorm = use_batchnorm

        self.out_dim = self.dims[3] * 2 * 2
        self.z_dim = z_dim

        self.layers = nn.Sequential(
            nn.Conv2d(
                num_channels,
                self.dims[0],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[0],
                self.dims[1],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[1],
                self.dims[2],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[2],
                self.dims[3],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[3])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
        )

        self.q_mean = nn.Linear(self.out_dim, self.z_dim)
        self.q_logvar = nn.Linear(self.out_dim, self.z_dim)

    def forward(self, x):
        out = self.layers(x).view(x.dim(0), -1)
        out = out.view(out.size(0), -1)

        mean_z = self.q_mean(out)
        var_z = self.q_logvar(out)
        return mean_z, var_z


class RAEDecoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        z_dim: int,
        ae_dim: int = 1028,
        kernel_dim: int = 4,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.dims = [ae_dim // (2**i) for i in reversed(range(3))]
        self.dims[-1] = ae_dim

        self.kernel_dim = kernel_dim
        self.padding = padding
        self.use_batchnorm = use_batchnorm

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim,
                self.dims[2],
                kernel_size=8,
                stride=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[2],
                self.dims[1],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[1],
                self.dims[0],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
        )

        self.out_layer = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(
                self.dims[0],
                self.num_channels,
                kernel_size=self.kernel_dim,
                stride=1,
                padding=self.padding,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.out_layer(self.layers(z[:, :, None, None]))
