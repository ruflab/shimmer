import pandas as pd
import numpy as np
import os

from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import torch
from imnet_logging import LogGWImagesCallback
from dataset import make_datamodule
from domains import ImageDomain, TextDomain, SDImageDomain
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

import matplotlib.pyplot as plt
from torchvision.utils import save_image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def text_to_image(text, width=256, height=128, font_size=8):
    """Convert a string to an image."""
    from PIL import Image, ImageDraw, ImageFont

    def wrap_text(text, num_words=6):
        words = text.split()
        lines = [' '.join(words[i:i+num_words]) for i in range(0, len(words), num_words)]
        return '\n'.join(lines)

    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    wrapped_text = wrap_text(text)
    draw.text((10, 10), wrapped_text, font=font, fill=(0, 0, 0))
    return image

def plot_and_save_images(input_tensors, tensors, titles, captions, vae_decoder, folder, final_image_name):
    """Plot and save a single image with all results given a list of tensors and titles."""
    if not os.path.exists(folder):
        os.makedirs(folder)

    num_titles = len(titles)
    fig, axes = plt.subplots(num_titles + 2, 5, figsize=(15, (num_titles + 2) * 3))  # Adjust for captions row
    canvas = FigureCanvas(fig)

    # Add a row for captions
    for j, caption in enumerate(captions):
        caption_img = text_to_image(caption, font_size=6)
        axes[0, j].imshow(caption_img)
        axes[0, j].axis('off')
        if j == 0:
            axes[0, j].set_title('Captions', fontsize=12)

    # Add a row for the input latents decoded directly
    input_images = vae_decoder(input_tensors[frozenset(["image_latents"])]["image_latents"]).cpu().detach()
    for j, img in enumerate(input_images):
        axes[1, j].imshow(img.permute(1, 2, 0).clamp(0, 1))  # Convert from CHW to HWC and clamp values to [0, 1]
        axes[1, j].axis('off')
        if j == 0:
            axes[1, j].set_title('Directly Decoded Input Latents', fontsize=12)

    for i, (tensor, title) in enumerate(zip(tensors, titles)):
        imgs = vae_decoder(tensor).cpu().detach()
        row_index = i + 2  # Adjust row index to account for the new top row

        for j, img in enumerate(imgs):
            axes[row_index, j].imshow(img.permute(1, 2, 0).clamp(0, 1))  # Convert from CHW to HWC and clamp values to [0, 1]
            axes[row_index, j].axis('off')
            if j == 0:
                axes[row_index, j].set_title(title, fontsize=12)

    # Save the final image
    save_path = os.path.join(folder, f"{final_image_name}.png")
    canvas.print_figure(save_path)


def run_inference_and_plot(global_workspace, random_latent_domains, captions):
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global_workspace.to(device)
    
    # Initialize the VAE model
    vae_model = global_workspace.domain_mods["image_latents"]

    # Prepare the input domains for different cases
    # 1. Caption embeddings only
    caption_only_domains = {frozenset(["caption_embeddings"]): random_latent_domains[frozenset(["caption_embeddings"])]}

    # 2. Image latents only
    image_only_domains = {frozenset(["image_latents"]): random_latent_domains[frozenset(["image_latents"])]}

    # 3. Both caption embeddings and image latents
    both_domains = {
        frozenset(["caption_embeddings"]): random_latent_domains[frozenset(["caption_embeddings"])],
        frozenset(["image_latents"]): random_latent_domains[frozenset(["image_latents"])]
    }

    # Perform inference
    caption_only_output = global_workspace.forward(caption_only_domains)
    image_only_output = global_workspace.forward(image_only_domains)
    both_output = global_workspace.forward(both_domains)

    # Collect the results to be plotted, excluding those going to caption embeddings
    results = [
        (image_only_output['broadcasts'][frozenset({'image_latents'})].get('image_latents', None), 'Broadcast - Image Latents from Image Only'),
        (image_only_output['cycles'][frozenset({'image_latents'})].get('image_latents', None), 'Cycle - Image Latents from Image Only'),

        (caption_only_output['broadcasts'][frozenset({'caption_embeddings'})].get('image_latents', None), 'Broadcast - Image Latents from Caption Only'),

        (both_output['broadcasts'][frozenset({'image_latents'})].get('image_latents', None), 'Broadcast - Image Latents from Both'),
    ]

    # Filter out None results and those going to caption embeddings
    filtered_results = [(tensor, title) for tensor, title in results if tensor is not None]

    # Plot and save the results
    folder = "inference_results"
    final_image_name = "final_output_image"
    input_tensors = random_latent_domains
    tensors = [result[0] for result in filtered_results]
    titles = [result[1] for result in filtered_results]
    plot_and_save_images(input_tensors, tensors, titles, captions, vae_model.decode, folder, final_image_name)

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

def print_shapes(nested_dict, indent=0):
    """Recursively print the shapes of tensors in a nested dictionary."""
    for key, value in nested_dict.items():
        print(' ' * indent + str(key) + ':')
        if isinstance(value, dict):
            print_shapes(value, indent + 2)
        else:
            print(' ' * (indent + 2) + str(value.shape))

def inference_gw():
    batch_size = 2056

    # Prepare data modules from the dataset script
    data = make_datamodule(batch_size)

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
    CHECKPOINT_PATH = "/home/rbertin/cleaned/git_synced/shimmer/examples/imgnet_example/wandb_output_weight_decay/simple_shapes_fusion/bqumes0b/checkpoints/epoch=299-step=187200.ckpt"
    checkpoint = torch.load(CHECKPOINT_PATH)


    # Set a fixed seed for reproducibility
    np.random.seed(42)

    # Assuming the model has a method to load from checkpoint state_dict
    def load_state_dict_without_vae(global_workspace, checkpoint):
        
        # Create a new state dictionary excluding keys with "vae"
        filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'vae' not in k}

        # Print the keys that are being loaded
        print("Loading the following keys:")
        for key in filtered_state_dict.keys():
            print(key)

        # Load the filtered state dictionary
        global_workspace.load_state_dict(filtered_state_dict, strict=False)

    # Example usage
    load_state_dict_without_vae(global_workspace, checkpoint)

    print("got checkpoint :)")
    csv_file = "/home/rbertin/pyt_scripts/BLIP_TEST/gemma/captions_gemma_valimgnet.csv"
    df = pd.read_csv(csv_file)

    # Set up the ImageNet dataset transformations
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ])

    val_dir = os.environ.get('DATASET_DIR', '.') + '/imagenet/train'
    imagenet_val_dataset = ImageFolder(root=val_dir, transform=transform)

    # Process the textual train samples through the model
    fig, axes = plt.subplots(2, 5, figsize=(30, 18))

    # Load image latents
    IMAGE_LATENTS_PATH_VAL = "/home/rbertin/pyt_scripts/full_imgnet/full_size/vae_full_withbigger_disc/384_val_combined_standardized_embeddings.npy"
    image_latents_val = np.load(IMAGE_LATENTS_PATH_VAL)
    image_latents_val = torch.tensor(image_latents_val).to('cuda')

    CAPTION_EMBEDDINGS_PATH_VAL = "/home/rbertin/pyt_scripts/BLIP_TEST/gemma/gemma_norm_bge_captions_val.npy"
    caption_embeddings_val = np.load(CAPTION_EMBEDDINGS_PATH_VAL)
    caption_embeddings_val = torch.tensor(caption_embeddings_val).to('cuda')

    # Select five random indices
    random_indices = np.random.choice(image_latents_val.shape[0], 5, replace=False)

    print("distribution : ",image_latents_val.mean(), image_latents_val.std())
    print("distribution : ",caption_embeddings_val.mean(), caption_embeddings_val.std())

    # Create random latents domain groups for testing
    random_latent_domains = {
        frozenset(["image_latents"]): {
            "image_latents": image_latents_val[random_indices]
        },
        frozenset(["caption_embeddings"]): {
            "caption_embeddings": caption_embeddings_val[random_indices]
        }
    }
    captions = df.iloc[random_indices]['Caption'].tolist()

    # Print the shapes of the output tensors
    run_inference_and_plot(global_workspace, random_latent_domains, captions)

if __name__ == "__main__":
    inference_gw()