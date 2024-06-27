import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
from PIL import Image
from tqdm.auto import tqdm

# ImageNet DataLoader setup
DATA_DIR = '/shared/datasets/imagenet/train'  # Update this path

# ImageNet DataLoader setup
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
imagenet_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Load existing captions
df_captions = pd.read_csv("captions_fullimgnet.csv")

# Determine the starting index of new captions based on ImageNet dataset length
new_captions_start_index = len(imagenet_dataset)  # Assuming imagenet_dataset is previously loaded

# Filter out the new captions assuming they are appended after the original dataset length
new_captions = df_captions[new_captions_start_index:]

# Save the new captions to a new CSV file
new_captions.to_csv("new_captions_only.csv", index=False)