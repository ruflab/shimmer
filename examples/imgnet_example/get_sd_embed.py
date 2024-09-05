import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
from tqdm import tqdm

# Set the device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("device:", device)

# Load the VAE model
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
vae.eval()

# Set the root directory where ImageNet is located
DATASET_DIR = os.environ.get('DATASET_DIR', '.')  # Get the environment variable, if not set, default to '.'
root_dir = DATASET_DIR

# Define your transformations
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = ImageFolder(root=os.path.join(root_dir, 'imagenet/train'), transform=transform)
val_dataset = ImageFolder(root=os.path.join(root_dir, 'imagenet/val'), transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)

print("Images loaded")

# Check inputs distribution
print("Checking out inputs distribution:\n")
data, _ = next(iter(train_loader))
print("Min:", data.min())
print("Max:", data.max())
print("Mean:", data.mean(dim=[0, 2, 3]))
print("Std:", data.std(dim=[0, 2, 3]))

# Loop over the validation set, get all the embeddings, and save them to .npy format.
all_embeddings = {}
all_indices = {}

for batch_idx, (inputs, _) in enumerate(tqdm(train_loader)):
    inputs = inputs.to(device)
    with torch.no_grad():
        encoded_latents = vae.encode(inputs).latent_dist.sample()
    
    # Get the shape of the current embeddings
    shape = encoded_latents.shape
    shape_str = str(shape)  # Convert shape to a string for dictionary keys
    
    # Check if this shape already exists in the dictionary
    if shape_str not in all_embeddings:
        print("Adding shape:", shape_str)
        all_embeddings[shape_str] = []
        all_indices[shape_str] = []
    
    # Add embeddings and corresponding indices to the respective list
    all_embeddings[shape_str].append(encoded_latents.detach().cpu().numpy())
    all_indices[shape_str].append(batch_idx)

root_dir = "sd_image_embeddings"

# Save each list of embeddings to a separate .npy file
os.makedirs(root_dir, exist_ok=True)

for shape_str, embeddings in all_embeddings.items():
    embeddings = np.concatenate(embeddings)
    np.save(os.path.join(root_dir, f'image_embeddings_{shape_str}_sd.npy'), embeddings)
    print(f"Saved embeddings of shape {shape_str} to", os.path.join(root_dir, f'image_embeddings_{shape_str}_sd.npy'))
    print(f"all_embeddings shape for {shape_str}: ", embeddings.shape)

    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    print(f"embeddings distribution for {shape_str}: mean: {mean}, std: {std}")

# Print the shapes and corresponding indices
for shape_str, indices in all_indices.items():
    print(f"Shape: {shape_str}, Indices: {indices}")
