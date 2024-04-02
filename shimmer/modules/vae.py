import math
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick for VAE

    Args:
        mean (`torch.Tensor`): predicted means
        logvar (`torch.Tensor`): predicted log variance

    Returns:
        `torch.Tensor`: a sample from normal distribution with provided
            parameters, sampled using the reparameterization trick.
    """
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return std * eps + mean


def kl_divergence_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Computes the KL divergence loss used in VAE.

    Args:
        mean (`torch.Tensor`): predicted means
        logvar (`torch.Tensor`): predicted logvars

    Returns:
        `torch.Tensor`: the loss
    """
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return kl


def gaussian_nll(
    mu: torch.Tensor, log_sigma: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """
    Computes gaussian nll loss used in VAE.

    Args:
        mu (`torch.Tensor`): predictions
        log_sigma (`torch.Tensor`): log sigma
        x (`torch.Tensor`): targets

    Returns:
        `torch.Tensor`: the Gaussian NLL loss
    """
    return (
        0.5 * torch.pow((x - mu) / log_sigma.exp(), 2)
        + log_sigma
        + 0.5 * math.log(2 * math.pi)
    )


class VAEEncoder(nn.Module, ABC):
    """
    Base class for a VAE encoder.
    """

    @abstractmethod
    def forward(self, x: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode representation with VAE.


        Args:
            x (`Any`): Some input value

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: the mean and log variance
        """
        ...


class VAEDecoder(nn.Module, ABC):
    """
    Base class for a VAE decoder.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        """
        Decode representation with VAE

        Args:
            x (`torch.Tensor`): VAE latent representation representation

        Returns:
            `Any`: the reconstructed input
        """
        ...


class VAE(nn.Module):
    """VAE module"""

    def __init__(
        self,
        encoder: VAEEncoder,
        decoder: VAEDecoder,
        beta: float = 1,
    ):
        """
        Initializes a VAE.

        Args:
            encoder (`VAEEncoder`): VAE encode
            decoder (`VAEDecoder`): VAE decoder
            beta (`float`): beta value for Beta-VAE. Defaults to 1.
        """
        super().__init__()

        assert beta >= 0

        self.beta = beta
        """Beta value for Beta-VAEs"""

        self.encoder = encoder
        """The encoder"""

        self.decoder = decoder
        """The decoder"""

    def encode(self, x: Any) -> torch.Tensor:
        """
        Encode the representation and returns the mean prediction of VAE.

        Args:
            x (`Any`): Some input value

        Returns:
            `torch.Tensor`: The mean representation.
        """
        mean_z, _ = self.encoder(x)
        return mean_z

    def decode(self, z: torch.Tensor) -> Any:
        """
        Decode the VAE latent representation into input value.

        Args:
            z (`torch.Tensor`): the VAE latent representation.

        Returns:
            `Any`: the reconstructed input.
        """
        return self.decoder(z)

    def forward(self, x: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
        """
        Encode and decodes from x.

        Args:
            x (`Any`): the input data

        Returns:
            `tuple[tuple[torch.Tensor, torch.Tensor], Any]`: The
                first tuple contains the mean and logvar of the encoded input,
                the second item is the reconstructed input.
        """
        mean, logvar = self.encoder(x)
        z = reparameterize(mean, logvar)

        x_reconstructed = self.decoder(z)

        return (mean, logvar), x_reconstructed
