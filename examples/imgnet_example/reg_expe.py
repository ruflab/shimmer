import torch
from imnet_logging import LogGWImagesCallback

from dataset import make_datamodule
from domains import ImageDomain, TextDomain
from lightning.pytorch import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint,LearningRateMonitor

from shimmer import GWDecoder, GWEncoder, BroadcastLossCoefs
from shimmer.modules.global_workspace import GlobalWorkspace, SchedulerArgs

from lightning.pytorch.loggers.wandb import WandbLogger

#put in utils later
import torch.nn as nn

def dropout_get_n_layers(n_layers: int, hidden_dim: int, dropout_rate: float = 0.5, use_batch_norm: bool = False) -> list[nn.Module]:
    """
    Makes a list of `n_layers` `nn.Linear` layers with optional `nn.BatchNorm1d`, `nn.ReLU`, and `nn.Dropout`.

    Args:
        n_layers (int): number of layers
        hidden_dim (int): size of the hidden dimension
        dropout_rate (float): dropout rate
        use_batch_norm (bool): whether to use batch normalization

    Returns:
        list[nn.Module]: list of layers.
    """
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    return layers

class dropout_GWDecoder(nn.Sequential):
    """A Decoder network for GWModules."""
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int,
        dropout_rate: float = 0.5,  # Provide a default value
        use_batch_norm: bool = False  # Provide a default value
    ):
        """
        Initializes the decoder with dropout and optional batch normalization layers.

        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension
            out_dim (int): output dimension
            n_layers (int): number of hidden layers
            dropout_rate (float): dropout rate
            use_batch_norm (bool): whether to use batch normalization
        """
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *dropout_get_n_layers(n_layers, hidden_dim, dropout_rate, use_batch_norm),
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
        dropout_rate: float = 0.5,  # Ensure this parameter is accepted
        use_batch_norm: bool = False  # Ensure this parameter is accepted
    ):
        """
        Initializes the encoder.

        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension
            out_dim (int): output dimension
            n_layers (int): number of hidden layers
            dropout_rate (float): dropout rate
            use_batch_norm (bool): whether to use batch normalization
        """
        super().__init__(in_dim, hidden_dim, out_dim, n_layers, dropout_rate, use_batch_norm)

def train_gw(gw_encoders, gw_decoders, domain_mods, run_name, use_weight_decay):
    # Prepare data modules from the dataset script
    data = make_datamodule(batch_size=2056)

    # Initialize the domain modules with the specific
    # latent dimensions and model parameters
    image_domain = ImageDomain(latent_dim=384)
    text_domain = TextDomain(latent_dim=384)

    domain_mods = {
        "image_latents": image_domain,
        "caption_embeddings": text_domain,
    }

    workspace_dim = 512  # Define the dimension of the global workspace


    # Loss coefficients setup
    loss_coefs: BroadcastLossCoefs = {
        "translations": 2.0,
        "demi_cycles": 1.0,
        "cycles": 1.0,
        "contrastives": .01,
        "fused": 1.0
    }

    n_epochs = 300  # Number of training epochs

    global_workspace = GlobalWorkspace(
        domain_mods,
        gw_encoders,
        gw_decoders,
        workspace_dim,
        loss_coefs,
        scheduler_args=SchedulerArgs(
            max_lr=0.0005,
            total_steps=n_epochs
            * len(iter(data.train_dataloader()))
        ),
    )

    wandb_logger = WandbLogger(
        save_dir=f"wandb_output_{run_name}",
        project="simple_shapes_fusion",
        entity="vanrullen-lab",
        tags=["train_gw"],
        name=run_name,
        reinit=True
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
            every_n_epochs=40,
        ),
        LogGWImagesCallback(
            train_samples,
            log_key="images/train",
            mode="train",
            every_n_epochs=40,
        )
    ]

    # Set weight decay if specified
    global_workspace.optim_weight_decay = 0.01 if use_weight_decay else 0.0


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

    # Finish the current wandb run
    import wandb
    wandb.finish()


import itertools
import torch.nn.functional as F
import itertools

# Define possible regularizations
regularizations = {
    'weight_decay': 'weight_decay',
    'batch_norm': 'batch_norm',
    'dropout': 'dropout'
}

# Define the combinations of regularizations
regularization_combinations = []
for r in range(1, len(regularizations) + 1):
    combinations = itertools.combinations(regularizations.items(), r)
    regularization_combinations.extend(combinations)

# Loop over each combination
for regs in regularization_combinations:
    run_name = "_".join(name for name, _ in regs)

    # Determine the regularization parameters
    use_weight_decay = False
    use_batch_norm = False
    dropout_rate = 0

    for name, _ in regs:
        if name == 'weight_decay':
            use_weight_decay = True
        elif name == 'batch_norm':
            use_batch_norm = True
        elif name == 'dropout':
            dropout_rate = 0.5

    # Initialize the domain modules with the specific latent dimensions and model parameters
    image_domain = ImageDomain(latent_dim=384)
    text_domain = TextDomain(latent_dim=384)

    domain_mods = {
        "image_latents": image_domain,
        "caption_embeddings": text_domain,
    }

    workspace_dim = 512  # Define the dimension of the global workspace
    
    # Define encoders and decoders with the regularizations
    gw_encoders = {
        name: dropout_GWEncoder(
            in_dim=mod.latent_dim,
            hidden_dim=1024,
            out_dim=workspace_dim,
            n_layers=4,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        ) for name, mod in domain_mods.items()
    }
    gw_decoders = {
        name: dropout_GWDecoder(
            in_dim=workspace_dim,
            hidden_dim=1024,
            out_dim=mod.latent_dim,
            n_layers=4,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm
        ) for name, mod in domain_mods.items()
    }

    # Run the training function
    train_gw(gw_encoders, gw_decoders, domain_mods, run_name, use_weight_decay)
