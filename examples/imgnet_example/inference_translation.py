
import numpy as np
import os

from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import torch
from imnet_logging import LogGWImagesCallback
from dataset import make_datamodule
from domains import ImageDomain, TextDomain
from lightning.pytorch import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from shimmer import GlobalWorkspace, GWDecoder, GWEncoder, BroadcastLossCoefs
from shimmer.modules.global_workspace import GlobalWorkspaceFusion, SchedulerArgs
from lightning.pytorch.loggers.wandb import WandbLogger
import torch.nn as nn
import numpy as np

# Define dropout layers
def dropout_get_n_layers(n_layers: int, hidden_dim: int, dropout_rate: float = 0.5) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for _ in range(n_layers):
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        ])
    return layers

# Define decoder with dropout layers
class dropout_GWDecoder(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout_rate: float = 0.5):
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            *dropout_get_n_layers(n_layers, hidden_dim, dropout_rate),
            nn.Linear(hidden_dim, out_dim),
        )

# Define encoder with dropout layers
class dropout_GWEncoder(dropout_GWDecoder):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, dropout_rate: float = 0.5):
        super().__init__(in_dim, hidden_dim, out_dim, n_layers, dropout_rate)

# Define inference function
def inference_gw():
    train_split = 0.8
    batch_size = 2056

    # Prepare data modules from the dataset script
    data = make_datamodule(train_split, batch_size)

    # Initialize the domain modules with the specific latent dimensions and model parameters
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
            total_steps=n_epochs * len(iter(data.train_dataloader()))
        ),
    )

    # Set precision for matrix multiplication
    torch.set_float32_matmul_precision("medium")

    # Initialize the BGE model
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


    # Load the model checkpoint
    CHECKPOINT_PATH = "wandb_output_bigger_gw/simple_shapes_fusion/wrx61j5u/checkpoints/epoch=1999-step=312000.ckpt"
    checkpoint = torch.load(CHECKPOINT_PATH)

    # Assuming the model has a method to load from checkpoint state_dict
    global_workspace.load_state_dict(checkpoint['state_dict'])

    # Get train samples
    train_samples = data.get_samples("val", 32)


    # Path to the directory where the embeddings file is stored and the filename
    root_dir = ''  # Change this to your directory path
    embeddings_file = '../../../../../pyt_scripts/BLIP_TEST/get_embed/bge_downsampled_captions.npy'
    file_path = os.path.join(root_dir, embeddings_file)

    # Load embeddings
    embeddings = np.load(file_path)

    # Print original mean and std
    mean_tensor = np.mean(embeddings, axis=0)
    std_tensor = np.std(embeddings, axis=0)

    def normalize_embedding(embedding, mean, std):
        return (embedding - mean) / std



    # Process the first 5 textual train samples through the model
    fig, axes = plt.subplots(1, 5, figsize=(30, 12))
    for i, sample in enumerate(train_samples[frozenset({'caption_embeddings'})]['caption_embeddings'][:5]):
        # Encode the textual sample using the TextDomain module
        text_embedding = bge_model.encode(sample, convert_to_tensor=True, device='cpu').unsqueeze(0)

        # Normalize the text embedding
        normalized_text_embedding = normalize_embedding(text_embedding, mean_tensor, std_tensor)

        # Encode text embedding into global workspace representation
        gw_representation = global_workspace.encode_and_fuse({'caption_embeddings': normalized_text_embedding}, selection_module=global_workspace.selection_mod)

        # Decode global workspace representation to image latents
        decoded_output = global_workspace.decode(gw_representation, domains=['image_latents'])

        # Decode image latents to image space using the VAE model
        image_output = image_domain.decode(decoded_output['image_latents'])

        # Plot the image output tensor
        axes[i].imshow(image_output.squeeze().permute(1, 2, 0).cpu().detach().numpy())
        axes[i].set_title("\n".join(" ".join(sample.split(' ')[j:j+3]) for j in range(0, len(sample.split(' ')), 3)), fontsize=10, pad=10)
        axes[i].axis('off')

    plt.subplots_adjust(top=0.85, wspace=0.4)  # Adjust top margin and width spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect parameter to make space for titles
    plt.savefig("translation_plot_val.png")



if __name__ == "__main__":
    inference_gw()
