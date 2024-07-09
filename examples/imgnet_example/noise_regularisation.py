
import random
import pickle
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import lpips

import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torchvision import transforms

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

print("device : ",device)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return F.relu(x + self.conv(x), inplace=True)


class VanillaVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims=None, beta=1.0, upsampling='bilinear', loss_type='lpips'):
        super(VanillaVAE, self).__init__()
        self.lpips_model = lpips.LPIPS(net='vgg', lpips=False) if loss_type == 'lpips' else None
        self.latent_dim = latent_dim
        hidden_dims = hidden_dims or [64, 128, 128, 256, 256, 512]
        self.beta = beta
        self.upsampling = upsampling
        self.loss_type = loss_type

        # Encoder setup
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            modules.append(ResidualBlock(h_dim))  # Add a residual block after each conv layer
            in_channels = h_dim
        modules.append(torch.nn.Tanh())
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hidden_dims[-1]*4*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4*4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4 * 4)

        modules = []
        hidden_dims.reverse()  # Ensure hidden_dims is in the correct order for building up
        for i in range(len(hidden_dims)-1):
            final_layer = i == len(hidden_dims) - 2  # Check if it's the layer before the last
            modules.append(self._deconv_block(hidden_dims[i], hidden_dims[i + 1],final=final_layer))
        
        # Add an explicit Upsample to 224x224 as the last upsampling step
        modules.append(nn.Upsample(size=(224, 224), mode=upsampling))  # Adjust mode as needed
        
        # Final convolution to produce the output image
        modules.append(nn.Sequential(
            nn.Conv2d(hidden_dims[-2], 3, kernel_size=3, padding=1),
            nn.Tanh()
        ))

        self.decoder = nn.Sequential(*modules)

    def _deconv_block(self, in_channels, out_channels, final=False):
        layers = []
        if self.upsampling == 'convtranspose':
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        elif self.upsampling == 'pixelshuffle':
            layers.append(nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(upscale_factor=2))
        else:  # Default upsampling strategy
            layers.append(nn.Upsample(scale_factor=2, mode=self.upsampling))
            if not final:  # Avoid changing the channel size in the final block before the RGB layer
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if not final:
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU())
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)


    def encode(self, input):
        result = torch.flatten(self.encoder(input), start_dim=1)
        return self.fc_mu(result), self.fc_var(result)

    def decode(self, z):
        result = self.decoder_input(z).view(-1, 512, 4,4)
        return self.decoder(result)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return torch.randn_like(std) * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), input, mu, log_var

    def loss_function(self, recons, input, mu, log_var, **kwargs):
        # Reconstruction Loss
        if self.loss_type == 'mse':
            recons_loss = F.mse_loss(recons, input)
        elif self.loss_type == 'lpips':
            recons_loss = self.lpips_model(recons, input).mean()

        # KLD Loss
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1)
        kld_loss = kld_loss.mean()

        # Final VAE Loss
        loss = recons_loss + self.beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}


    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        return self.decode(z)

    def generate(self, x):
        return self.forward(x)[0]

import torch.optim as optim
from tqdm import tqdm
import os

from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def denormalize(tensor, mean, std):
    """
    Denormalizes the image tensor from given mean and std.
    """
    if len(tensor.shape) == 3:  # If tensor is 3D, add a batch dimension
        tensor = tensor.unsqueeze(0)

    for i in range(tensor.size(1)):  # Now tensor shape should be [Batch, Channels, Height, Width]
        tensor[:, i] = tensor[:, i] * std[i] + mean[i]

    if tensor.shape[0] == 1:  # If there was only one image, remove the batch dimension
        tensor = tensor.squeeze(0)

    return tensor

# Set the root directory where ImageNet is located
DATASET_DIR = os.environ.get('DATASET_DIR', '.')  # Get the environment variable, if not set, default to '.'
root_dir = DATASET_DIR

# Define your transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Create datasets
train_dataset = ImageFolder(root=root_dir+'/imagenet/train', transform=transform)
val_dataset = ImageFolder(root=root_dir+'/imagenet/val', transform=val_transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

print("Images loaded")

# Check inputs distribution
print("Checking out inputs distribution:\n")
data, _ = next(iter(train_loader))
print("Min:", data.min())
print("Max:", data.max())
print("Mean:", data.mean(dim=[0, 2, 3]))
print("Std:", data.std(dim=[0, 2, 3]))

# Model parameters
upsampling_methods = ['nearest']
loss_functions = ['lpips']
latent_dims = [512]
root_dir = ''

from tqdm import tqdm

#here load the checkpoint from ./vae_full_withbigger_disc/nearest_lpips_latent_dim=384
checkpoint_path = "/home/rbertin/pyt_scripts/full_imgnet/full_size/vae_bigdisc_goodbeta_50ep/bilinear_lpips_latent_dim=512/vae_model.pth"
model = VanillaVAE(in_channels=3, latent_dim=512, upsampling='bilinear', loss_type='lpips').to(device)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

import random
import numpy as np
from torch.utils.data import Subset

# Number of images to save
num_images_to_save = 5

# Randomly pick indices for images to save
dataset_size = len(train_dataset)
random_indices = random.sample(range(dataset_size), num_images_to_save)

# Create a subset of the dataset with only the selected indices
subset = Subset(train_dataset, random_indices)
subset_loader = DataLoader(subset, batch_size=num_images_to_save, shuffle=False)

# Collect images and their reconstructions
images_to_save = []
noise_levels = np.linspace(0.05, 0.25, 5)

for inputs, _ in subset_loader:
    inputs = inputs.to(device)
    mu, log_var = model.encode(inputs)
    z = model.reparameterize(mu, log_var)
    
    for j in range(inputs.size(0)):
        original = inputs[j].detach().cpu()
        recons_list = [original]  # Include the original image
        
        # Add the original reconstruction (no noise)
        recons = model.decode(z[j].unsqueeze(0))
        recons_list.append(recons.squeeze(0).detach().cpu())
        
        for noise_std in noise_levels:
            noise = torch.randn_like(z[j]) * noise_std
            noisy_latent = z[j] + noise
                
            recons = model.decode(noisy_latent.unsqueeze(0))
            recons_list.append(recons.squeeze(0).detach().cpu())
        
        images_to_save.append(recons_list)

import random
import numpy as np
from torch.utils.data import Subset

# Number of images to save
num_images_to_save = 5

# Randomly pick indices for images to save
dataset_size = len(train_dataset)
random_indices = random.sample(range(dataset_size), num_images_to_save)

# Create a subset of the dataset with only the selected indices
subset = Subset(train_dataset, random_indices)
subset_loader = DataLoader(subset, batch_size=num_images_to_save, shuffle=False)

# Collect images and their reconstructions
images_to_save = []
noise_levels = np.linspace(0.01, 0.05, 5)

for inputs, _ in subset_loader:
    inputs = inputs.to(device)
    mu, log_var = model.encode(inputs)
    z = model.reparameterize(mu, log_var)
    
    for j in range(inputs.size(0)):
        original = inputs[j].detach().cpu()
        recons_list = [original]  # Include the original image
        
        # Add the original reconstruction (no noise)
        recons = model.decode(z[j].unsqueeze(0))
        recons_list.append(recons.squeeze(0).detach().cpu())
        
        for noise_std in noise_levels:
            noise = torch.randn_like(z[j]) * noise_std
            noisy_latent = z[j] + noise
                
            recons = model.decode(noisy_latent.unsqueeze(0))
            recons_list.append(recons.squeeze(0).detach().cpu())
        
        images_to_save.append(recons_list)

# Ensure we have exactly 5 rows and 7 columns in the subplot grid
fig, axes = plt.subplots(num_images_to_save, 7, figsize=(20, 2 * num_images_to_save))

for idx, recons_list in enumerate(images_to_save):
    for noise_idx, recons in enumerate(recons_list):
        # Clip images between 0 and 1
        recons = torch.clamp(recons, 0, 1)
        
        if noise_idx == 0:
            title = "Original"
        elif noise_idx == 1:
            title = "No Noise"
        else:
            title = f"Noise: {noise_levels[noise_idx-2]:.2f}"
        
        axes[idx, noise_idx].imshow(recons.permute(1, 2, 0))
        axes[idx, noise_idx].set_title(title)
        axes[idx, noise_idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(root_dir, 'before_after_autoencoding_with_noise.png'))
plt.show()

print("Saved before and after autoencoding results with noise to", os.path.join(root_dir, 'before_after_autoencoding_with_noise.png'))
