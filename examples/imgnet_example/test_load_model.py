import torch
from dataset import make_datamodule
from domains import ImageDomain, TextDomain
from shimmer import GWDecoder, GWEncoder, BroadcastLossCoefs
from shimmer.modules.global_workspace import GlobalWorkspace
from lightning.pytorch import Trainer

# Import necessary libraries for regularization
import torch.nn as nn

# Define the model classes as in the training script

def dropout_get_n_layers(n_layers: int, hidden_dim: int, dropout_rate: float = 0.5, use_batch_norm: bool = False) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    return layers

class dropout_GWDecoder(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout_rate: float = 0.5, use_batch_norm: bool = False):
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *dropout_get_n_layers(n_layers, hidden_dim, dropout_rate, use_batch_norm),
            nn.Linear(hidden_dim, out_dim),
        )

class dropout_GWEncoder(dropout_GWDecoder):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout_rate: float = 0.5, use_batch_norm: bool = False):
        super().__init__(in_dim, hidden_dim, out_dim, n_layers, dropout_rate, use_batch_norm)

def load_model(checkpoint_path, use_weight_decay, use_batch_norm, dropout_rate):
    # Initialize the domain modules
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

    # Loss coefficients setup
    loss_coefs: BroadcastLossCoefs = {
        "translations": 2.0,
        "demi_cycles": 1.0,
        "cycles": 1.0,
        "contrastives": .01,
        "fused": 1.0
    }

    global_workspace = GlobalWorkspace(
        domain_mods,
        gw_encoders,
        gw_decoders,
        workspace_dim,
        loss_coefs,
        scheduler_args=None  # No scheduler needed for inference
    )

    # Load the model checkpoint
    checkpoint = torch.load(checkpoint_path)
    global_workspace.load_state_dict(checkpoint['state_dict'])

    return global_workspace

def main():
    # Configuration
    use_weight_decay = True  # Set to True if weight decay is used
    use_batch_norm = False  # Set to True if batch normalization is used
    dropout_rate = 0.0  # Set to the dropout rate if dropout is used

    # Load the model
    CHECKPOINT_PATH = "/home/rbertin/cleaned/git_synced/shimmer/examples/imgnet_example/wandb_output_weight_decay_batch_norm_dropout/simple_shapes_fusion/g2s0lde9/checkpoints/epoch=299-step=187200.ckpt"

    model = load_model(CHECKPOINT_PATH, use_weight_decay, use_batch_norm, dropout_rate)

    # Prepare data modules
    data = make_datamodule(batch_size=1)  # Batch size for inference

    # Trainer setup for inference
    trainer = Trainer(devices=1, accelerator='gpu', logger=False)  # assuming inference on 1 GPU

    # Run inference
    predictions = trainer.predict(model, data)

    # Process predictions (for example, save them to a file or print them)
    for prediction in predictions:
        print(prediction)

if __name__ == "__main__":
    main()
