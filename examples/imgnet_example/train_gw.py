import torch
from imnet_logging import LogGWImagesCallback

from dataset import make_datamodule
from domains import ImageDomain, TextDomain
from lightning.pytorch import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint,LearningRateMonitor

from shimmer import GlobalWorkspace, GWDecoder, GWEncoder, BroadcastLossCoefs
from shimmer.modules.global_workspace import GlobalWorkspaceFusion, SchedulerArgs

from lightning.pytorch.loggers.wandb import WandbLogger

#put in utils later
import torch.nn as nn

def dropout_get_n_layers(n_layers: int, hidden_dim: int, dropout_rate: float = 0.5) -> list[nn.Module]:
    """
    Makes a list of `n_layers` `nn.Linear` layers with `nn.ReLU` and `nn.Dropout`.

    Args:
        n_layers (int): number of layers
        hidden_dim (int): size of the hidden dimension
        dropout_rate (float): dropout rate

    Returns:
        list[nn.Module]: list of linear, relu, and dropout layers.
    """
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
    return layers

class dropout_GWDecoder(nn.Sequential):
    """A Decoder network for GWModules."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        dropout_rate: float = 0.5  # Provide a default value
    ):
        """
        Initializes the decoder with dropout layers.

        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension
            out_dim (int): output dimension
            n_layers (int): number of hidden layers
            dropout_rate (float): dropout rate
        """
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *dropout_get_n_layers(n_layers, hidden_dim, dropout_rate),
            nn.Linear(hidden_dim, out_dim),
        )

class dropout_GWEncoder(dropout_GWDecoder):
    """
    An Encoder network used in GWModules.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        dropout_rate: float = 0.5  # Ensure this parameter is accepted
    ):
        """
        Initializes the encoder.

        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension
            out_dim (int): output dimension
            n_layers (int): number of hidden layers
            dropout_rate (float): dropout rate
        """
        super().__init__(in_dim, hidden_dim, out_dim, n_layers, dropout_rate)

def train_gw():
    train_split=0.8
    # Prepare data modules from the dataset script
    data = make_datamodule(train_split, batch_size=2056)


    # Initialize the domain modules with the specific
    # latent dimensions and model parameters
    image_domain = ImageDomain(latent_dim=384)
    text_domain = TextDomain(latent_dim=384)

    domain_mods = {
        "image_latents": image_domain,
        "caption_embeddings": text_domain,
    }

    workspace_dim = 512  # Define the dimension of the global workspace

    # Define modality encoders and decoders
    gw_encoders = {}
    gw_decoders = {}
    for name, mod in domain_mods.items():
        gw_encoders[name] = dropout_GWEncoder(
            in_dim=mod.latent_dim,
            hidden_dim=1024,
            out_dim=workspace_dim,
            n_layers=4,
            dropout_rate=0.0  # Example dropout rate
        )
        gw_decoders[name] = dropout_GWDecoder(
            in_dim=workspace_dim,
            hidden_dim=1024,
            out_dim=mod.latent_dim,
            n_layers=4,
            dropout_rate=0.0  # Example dropout rate
        )

    # Loss coefficients setup
    loss_coefs: BroadcastLossCoefs = {
        "translations": 2.0,
        "demi_cycles": 1.0,
        "cycles": 1.0,
        "contrastives": .05,
        "fused": 1.0
    }

    n_epochs = 2000  # Number of training epochs

    global_workspace = GlobalWorkspaceFusion(
        domain_mods,
        gw_encoders,
        gw_decoders,
        workspace_dim,
        loss_coefs,
        scheduler_args=SchedulerArgs(
            max_lr=0.0002,
            total_steps=n_epochs
            * len(iter(data.train_dataloader()))
        ),
    )

    wandb_logger = None
    run_name = "new_bigger_gw"
    wandb_logger = WandbLogger(
        save_dir=f"wandb_output_{run_name}",
        project="simple_shapes_fusion",
        entity="vanrullen-lab",
        tags=["train_gw"],
        name=run_name,
    )

    torch.set_float32_matmul_precision("medium")

    train_samples = data.get_samples("train", 32)
    val_samples = data.get_samples("val", 32)


    callbacks: list[Callback] = [
        LearningRateMonitor(logging_interval="step"),
        LogGWImagesCallback(
            val_samples,
            log_key="images/val",
            mode="val",
            every_n_epochs=10,
        ),
        LogGWImagesCallback(
            train_samples,
            log_key="images/train",
            mode="train",
            every_n_epochs=10,
        )
    ]

    # Trainer setup
    trainer = Trainer(
        logger=wandb_logger,
        devices=1,  # assuming training on 1 GPU
        max_epochs=n_epochs,
        log_every_n_steps=100,
        callbacks=callbacks,
    )

    # Run training and validation
    trainer.fit(global_workspace, data)
    trainer.validate(global_workspace, data, "best")

if __name__ == "__main__":
    train_gw()
