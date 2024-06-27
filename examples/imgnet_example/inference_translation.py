import pandas as pd
import numpy as np
import os

from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import torch
from imnet_logging import LogGWImagesCallback
from dataset import make_datamodule
from domains import ImageDomain, TextDomain
from lightning.pytorch import Trainer, Callback
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from shimmer import GWDecoder, GWEncoder, BroadcastLossCoefs
from shimmer.modules.global_workspace import GlobalWorkspace, SchedulerArgs
from lightning.pytorch.loggers.wandb import WandbLogger
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import random

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
    batch_size = 2056

    # Prepare data modules from the dataset script
    data = make_datamodule(batch_size)

    # Initialize the domain modules with the specific latent dimensions and model parameters
    image_domain = ImageDomain(latent_dim=512)
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

    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n_epochs = 2000  # Number of training epochs

    global_workspace = GlobalWorkspace(
        domain_mods,
        gw_encoders,
        gw_decoders,
        workspace_dim,
        loss_coefs,
        scheduler_args=SchedulerArgs(
            max_lr=0.0002,
            total_steps=n_epochs * len(iter(data.train_dataloader()))
        ),
    ).to(device)

    # Set precision for matrix multiplication
    torch.set_float32_matmul_precision("medium")

    # Initialize the BGE model
    bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Load the model checkpoint
    CHECKPOINT_PATH = "/home/rbertin/cleaned/git_synced/shimmer/examples/imgnet_example/wandb_output_bigger_vae/simple_shapes_fusion/7t80u3rm/checkpoints/epoch=499-step=312000.ckpt"
    checkpoint = torch.load(CHECKPOINT_PATH)

    # Assuming the model has a method to load from checkpoint state_dict
    global_workspace.load_state_dict(checkpoint['state_dict'])

    print("got checkpoint :)")
    csv_file = "captions_fullimgnet_val_noshuffle.csv"
    df = pd.read_csv(csv_file)

    # Function to get random samples
    def get_random_samples(df, num_samples=5):
        return df.sample(n=num_samples)

    # Get 5 random samples
    train_samples = get_random_samples(df, num_samples=5)

    print("got samples :)")

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

    # Set up the ImageNet dataset transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    val_dir = os.environ.get('DATASET_DIR', '.') + '/imagenet/train'
    imagenet_val_dataset = ImageFolder(root=val_dir, transform=transform)

    # Get the image indices from the sampled captions
    image_indices = train_samples['Image Index'].tolist()

    # Create a subset of the dataset with the selected image indices
    subset_dataset = Subset(imagenet_val_dataset, image_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=1, shuffle=False)

    # Process the textual train samples through the model
    fig, axes = plt.subplots(2, 5, figsize=(30, 18))

    # Move mean and std tensors to device
    mean_tensor = torch.tensor(mean_tensor).to('cuda')
    std_tensor = torch.tensor(std_tensor).to('cuda')
    # Set the directory for the dataset
    val_dir = os.environ.get('DATASET_DIR', '.') + '/imagenet/train'
    imagenet_val_dataset = ImageFolder(root=val_dir, transform=transform)

    # Randomly select five indices from the length of the dataset
    random_indices = random.sample(range(len(imagenet_val_dataset)), 5)

    # Initialize the DataLoader
    data_loader = DataLoader(imagenet_val_dataset, batch_size=1, shuffle=False)

    # Process the textual train samples through the model
    fig, axes = plt.subplots(2, 5, figsize=(30, 18))

    # Move mean and std tensors to device
    mean_tensor = torch.tensor(mean_tensor).to('cuda')
    std_tensor = torch.tensor(std_tensor).to('cuda')

    IMAGE_LATENTS_PATH_VAL = (
        "/home/rbertin/pyt_scripts/full_imgnet/full_size/vae_bigdisc_goodbeta_50ep"
        "/combined_standardized_embeddings.npy"
    )

    # Load image latents
    image_latents_val = np.load(IMAGE_LATENTS_PATH_VAL)
    image_latents_val = torch.tensor(image_latents_val).to('cuda')

    for i, image_index in tqdm(enumerate(random_indices)):
        # Get the image and target at the selected index
        image, _ = imagenet_val_dataset[image_index]

        # Get the image latents at the same index as the original image in the dataset
        file_latents = image_latents_val[image_index].unsqueeze(0).to('cuda')
        print("file_latents stats before modif : ", file_latents.mean(), file_latents.std())
        file_latents = (file_latents * global_workspace.domain_mods["image_latents"].std.to(file_latents.device)) - global_workspace.domain_mods["image_latents"].mean.to(file_latents.device)

        print("image.shape : ", image.shape)
        encoded = image_domain.vae_model.encode(image.unsqueeze(0).to('cuda'))[0]

        print("difference : ", (file_latents - encoded.flatten(start_dim=1)).mean())
        print("encoded stats : ", encoded.mean(), encoded.std())
        print("file_latents stats : ", file_latents.mean(), file_latents.std())

        # Decode image latents to image space using the VAE model
        image_output_1 = global_workspace.domain_mods["image_latents"].vae_model.decode(file_latents.reshape(-1, 32, 4, 4))
        image_output_2 = global_workspace.domain_mods["image_latents"].vae_model.decode(encoded.reshape(-1, 32, 4, 4))

        # Plot the demi-cycled image output tensor
        axes[0, i].imshow(image_output_1.squeeze().permute(1, 2, 0).cpu().detach().numpy())
        axes[0, i].set_title("Demi-cycled Image", fontsize=10, pad=10)
        axes[0, i].axis('off')

        # Plot the original image
        image = image.to('cuda')
        axes[1, i].imshow(image_output_2.squeeze().permute(1, 2, 0).cpu().detach().numpy())
        axes[1, i].set_title("Original Image", fontsize=10, pad=10)
        axes[1, i].axis('off')

    plt.subplots_adjust(top=0.85, wspace=0.4)  # Adjust top margin and width spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect parameter to make space for titles
    plt.savefig("demi_cycle_plot_val.png")

if __name__ == "__main__":
    inference_gw()
