import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import AdamW
from vae import VanillaVAE

import numpy as np

from shimmer import DomainModule, LossOutput

from diffusers.models import AutoencoderKL


class ImageDomain(DomainModule):
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)
        # load the model parameters
        checkpoint_path = "/home/rbertin/pyt_scripts/full_imgnet/full_size/vae_full_withbigger__disc/nearest_lpips_latent_dim=384/vae_model.pth"
        self.vae_model = VanillaVAE(
            in_channels=3, latent_dim=latent_dim, upsampling="nearest", loss_type="lpips"
        )
        self.vae_model.load_state_dict(torch.load(checkpoint_path))
        self.vae_model.eval()
        print(
            "nb params in the vae model : ",
            sum(p.numel() for p in self.vae_model.parameters())
        )


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.eval()

        # Decode using VAE model
        val = self.vae_model.decode(z)
        return val

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        (domain,) = batch
        decoded = self.decode(self.encode(domain))
        loss = F.mse_loss(domain, decoded)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        (domain,) = batch
        decoded = self.decode(self.encode(domain))
        loss = F.mse_loss(domain, decoded)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        # Computes an illustrative loss, can be tailored for specific use cases
        return LossOutput(loss=F.mse_loss(pred, target))



class SDImageDomain(DomainModule):
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)
        if latent_dim != 1024:
            raise ValueError("vision latent_dim must be 1024")

        # load the model parameters
        self.vae_model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

        self.vae_model.eval()
        print(
            "nb params in the vae model : ",
            sum(p.numel() for p in self.vae_model.parameters())
        )


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.eval()
        z = z.reshape(z.shape[0],4,16,16)
        val = self.vae_model.decode(z).sample
        return val

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        (domain,) = batch
        decoded = self.decode(self.encode(domain))
        loss = F.mse_loss(domain, decoded)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        (domain,) = batch
        decoded = self.decode(self.encode(domain))
        loss = F.mse_loss(domain, decoded)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        # Computes an illustrative loss, can be tailored for specific use cases
        return LossOutput(loss=F.mse_loss(pred, target))


class TextDomain(DomainModule):
    def __init__(self, latent_dim: int):
        super().__init__(latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Just pass through the embedding
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Text embeddings do not decode back to text, return input as placeholder
        return z

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        (domain,) = batch
        # No actual decoding is performed, simulate training step
        loss = F.mse_loss(domain, domain)  # No operation loss (placeholder)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        (domain,) = batch
        # No actual decoding is performed, simulate validation step
        loss = F.mse_loss(domain, domain)  # No operation loss (placeholder)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        # Computes an illustrative loss, can be tailored for specific use cases
        return LossOutput(loss=F.mse_loss(pred, target))
