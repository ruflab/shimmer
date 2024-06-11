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
import torch
import numpy as np
import random

# Define the load_data function
def load_data(path):
    return np.load(path)

# Paths to data
IMAGE_LATENTS_PATH_TRAIN = (
    "/home/rbertin/pyt_scripts/full_imgnet/full_size/vae_full_withbigger__disc/"
    "val_image_embeddings.npy"
)
IMAGE_LATENTS_PATH_VAL = (
    "/home/rbertin/pyt_scripts/full_imgnet/full_size/vae_full_withbigger__disc/"
    "val_image_embeddings_val.npy"
)
CAPTION_EMBEDDINGS_PATH_TRAIN = (
    "/home/rbertin/cleaned/git_synced/shimmer/examples/imgnet_example/"
    "bge_fullsized_captions_norm_fixed.npy"
)
CAPTION_EMBEDDINGS_PATH_VAL = (
    "/home/rbertin/cleaned/git_synced/shimmer/examples/imgnet_example/"
    "bge_fullsized_captions_norm_val.npy"
)

# Load training data
image_latents_train = load_data(IMAGE_LATENTS_PATH_TRAIN)[:50000]
caption_embeddings_train = load_data(CAPTION_EMBEDDINGS_PATH_TRAIN)[:50000]


# Load validation data
image_latents_val = load_data(IMAGE_LATENTS_PATH_VAL)
caption_embeddings_val = load_data(CAPTION_EMBEDDINGS_PATH_VAL)



print("got this far !")

# Assuming `make_datamodule` and `data_module` are defined somewhere in your code
data = make_datamodule(0.8, batch_size=2056)

print("got this far !")
train_loader = data.train_dataloader()
val_loader = data.val_dataloader()

# Function to get a random pair of matched latents
def get_random_matched_pair(batch):
    indices = list(range(len(batch['image_latents'])))
    random_index = random.choice(indices)
    return batch['image_latents'][random_index], batch['caption_embeddings'][random_index]

# Function to find the index of a latent in a dataset
def find_index(latent, dataset):
    for idx, item in enumerate(dataset):
        if np.allclose(latent, item, atol=1e-15):  # Using a tolerance for floating point comparison
            return idx
    return -1

print("got this far !")

# Perform the comparison 1000 times on the validation set
print("Validation set comparisons:")
for i in range(1000):
    batch = next(iter(val_loader))

    image_latent, text_latent = get_random_matched_pair(batch)
    
    if image_latent is None or text_latent is None:
        print("No matched pair found in the batch.")
        continue

    image_latent_np = image_latent.cpu().numpy()
    text_latent_np = text_latent.cpu().numpy()
    
    image_index = find_index(image_latent_np, image_latents_val)
    text_index = find_index(text_latent_np, caption_embeddings_val)
    
    if image_index == -1 or text_index == -1:
        print("Index not found in the respective dataset.")
        continue
    if image_index != text_index or i%10==0:
        print(f"Image index: {image_index}, Text index: {text_index}, Match: {image_index == text_index}")

# Perform the comparison 1000 times on the training set
print("Training set comparisons:")
for i in range(1000):
    batch = next(iter(train_loader))
    image_latent, text_latent = get_random_matched_pair(batch)


    
    if image_latent is None or text_latent is None:
        print("No matched pair found in the batch.")
        continue

    image_latent_np = image_latent.cpu().numpy()
    text_latent_np = text_latent.cpu().numpy()
    
    image_index = find_index(image_latent_np, image_latents_train)
    text_index = find_index(text_latent_np, caption_embeddings_train)
    
    if image_index == -1 or text_index == -1:
        print("Index not found in the respective dataset.")
        continue
    if image_index != text_index or i%10==0:
        print(f"Image index: {image_index}, Text index: {text_index}, Match: {image_index == text_index}")
