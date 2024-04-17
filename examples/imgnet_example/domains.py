import torch
import torch.nn.functional as F
from torch.nn import Linear

from shimmer import DomainModule, LossOutput

import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch.optim import AdamW
from shimmer import DomainModule, LossOutput

from shimmer.utils import group_device
from vae import VanillaVAE



class ImageDomain(DomainModule):
    def __init__(self, vae_model: Module, latent_dim: int):
        super().__init__(latent_dim)
        # load the model parameters
        checkpoint_path = "vae_model.pth"
        model = VanillaVAE(in_channels=3, latent_dim=384, upsampling='nearest', loss_type='lpips')
        model.load_state_dict(torch.load(checkpoint_path))


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Just pass through the embedding
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Decode using VAE model
        return self.vae_model.decode(z)

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
