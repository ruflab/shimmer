import os
import random
import pandas as pd
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

# ImageNet DataLoader setup
imagenet_data_path = '/shared/datasets/imagenet/train'  # Path to the ImageNet dataset

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Load the dataset
imagenet_dataset = datasets.ImageFolder(root=imagenet_data_path, transform=transform)

# Load the no-duplicates csv
captions_df = pd.read_csv('../../../../../pyt_scripts/BLIP_TEST/gemma/no_duplicates_gemma_captions.csv')

# Check if the lengths match
assert len(captions_df) == len(imagenet_dataset), "The lengths of the dataset and captions CSV do not match."

# Pick randomly five indices between 0 and the length of the captions csv
random_indices = random.sample(range(len(captions_df)), 5)

# Plot the images and captions
fig, axes = plt.subplots(5, 1, figsize=(10, 20))

for i, idx in enumerate(random_indices):
    image_path, _ = imagenet_dataset.samples[idx]
    image = Image.open(image_path)
    caption = captions_df.loc[idx, 'Caption']

    axes[i].imshow(image)
    axes[i].axis('off')
    # Split the caption into lines with up to 5 words each
    caption_lines = '\n'.join([' '.join(caption.split()[j:j+5]) for j in range(0, len(caption.split()), 5)])
    axes[i].set_title(caption_lines, fontsize=8)

# Save the plot to a file
plt.tight_layout()
plt.savefig('random_images_with_captions.png')

print("Images and captions plotted and saved to 'random_images_with_captions.png'.")
