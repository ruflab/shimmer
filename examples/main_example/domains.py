import torch
import torch.nn.functional as F
from torch.nn import Linear

from shimmer import DomainModule, LossOutput


class GenericDomain(DomainModule):
    def __init__(self, input_size: int, latent_dim: int) -> None:
        super().__init__(latent_dim)

        self.input_size = input_size
        # after the __init__, self.latent_dim is set
        self.encoder = Linear(self.input_size, self.latent_dim)
        self.decoder = Linear(self.latent_dim, self.input_size)

    # Pytorch Lightning stuff to train the Domain Module

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ):
        # TensorDataset always gives a list of tensor, even if there is only one input.
        (domain,) = batch
        decoded = self.decoder(self.encoder(domain))
        loss = F.mse_loss(domain, decoded)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ):
        # TensorDataset always gives a list of tensor, even if there is only one input.
        (domain,) = batch
        decoded = self.decoder(self.encoder(domain))
        loss = F.mse_loss(domain, decoded)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Define which optimizer to use
        """
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-6)

    # shimmer stuff to train the GW

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode data to the unimodal representation.
        Args:
            x: input data of the domain.
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode data back to the unimodal representation.
        Args:
            z: latent reprensetation of the domain.
        Returns:
            the reconstructed input data
        """
        return self.decoder(z)

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> LossOutput:
        """
        Computes a generic loss in the domain's latent representation.
        This must return a LossOutput object. LossOutput is used to separate
        the loss used for training the model (given to loss parameter), and
        additional metrics that are logged, but not trained on.
        """
        return LossOutput(loss=F.mse_loss(pred, target))
