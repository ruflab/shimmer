from collections import defaultdict
import os
from typing import Any, Dict, List, Literal, Mapping, Tuple, Union
import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, default_collate
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset


import pandas as pd

# Paths to data
IMAGE_LATENTS_PATH_TRAIN = (
    "/home/rbertin/pyt_scripts/full_imgnet/full_size/vae_full_withbigger_disc/384_combined_standardized_embeddings.npy"
)
IMAGE_LATENTS_PATH_VAL = (
    "/home/rbertin/pyt_scripts/full_imgnet/full_size/vae_full_withbigger_disc/384_val_combined_standardized_embeddings.npy"
)
CAPTION_EMBEDDINGS_PATH_TRAIN = (
    "/home/rbertin/pyt_scripts/BLIP_TEST/gemma/gemma_norm_bge_captions_train.npy"
)
CAPTION_EMBEDDINGS_PATH_VAL = (
    "/home/rbertin/pyt_scripts/BLIP_TEST/gemma/gemma_norm_bge_captions_val.npy"
)


def load_data(file_path: str) -> torch.Tensor:
    """
    Load data from the specified numpy file path.
    """
    data = torch.from_numpy(np.load(file_path))
    return data


class DomainDataset(Dataset):
    def __init__(self, domain_data: dict[str, torch.Tensor]) -> None:
        """
        Initializes a dataset that takes a mapping from domain names to tensors of data.
        """
        self.domain_data = domain_data

    def __len__(self) -> int:
        return self.domain_data[next(iter(self.domain_data))].size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            domain_name: data[index] for domain_name, data in self.domain_data.items()
        }

def collate_fn(batch: List[Dict[Union[str, frozenset], Dict[str, torch.Tensor]]]) -> Tuple[Dict[Union[str, frozenset], Dict[str, torch.Tensor]], int, int]:
    """
    Custom collate function for training batches.
    Includes both single and combined domains, ensuring single domains are also wrapped in a frozenset.
    """
    collated_batch = {}
    for key in batch[0].keys():
        if isinstance(key, frozenset):
            collated_batch[key] = {
                sub_key: torch.stack([b[key][sub_key] for b in batch])
                for sub_key in batch[0][key].keys()
            }
        else:
            # Convert single domain to frozenset for consistent structure
            domain_key = frozenset([key])
            collated_batch[domain_key] = {key: torch.stack([b[key] for b in batch])} # type: ignore
    return collated_batch, 0, 0

def collate_fn_validation(batch: List[Dict[str, torch.Tensor]]) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """
    Custom collate function for validation batches.
    Excludes combined domains and returns as tuple.
    """
    collated_batch = {key: torch.stack([b[key] for b in batch]) for key in batch[0].keys() if not isinstance(key, frozenset)}
    return collated_batch, 0, 0



class CaptionsDataset():
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.captions = self.df['Caption'].tolist()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions[idx]

class GWDataModule(LightningDataModule):
    def __init__(
        self,
        val_datasets: dict[frozenset[str], DomainDataset],
        train_datasets: dict[frozenset[str], DomainDataset],
        batch_size: int,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size

        self.val_datasets = val_datasets
        self.train_datasets = train_datasets

        # Set up the ImageNet dataset transformations
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Separate datasets for train and validation
        train_dir = os.environ.get('DATASET_DIR', '.') + '/imagenet/train'
        val_dir = os.environ.get('DATASET_DIR', '.') + '/imagenet/val'  # Assuming a separate validation directory

        self.imagenet_train_dataset = ImageFolder(root=train_dir, transform=transform)
        self.imagenet_val_dataset = ImageFolder(root=val_dir, transform=transform)

    def setup_dataloaders(self, datasets):
        dataloaders = {}
        for domain_set, dataset in datasets.items():
            # Each dataset gets its own DataLoader
            dataloaders[domain_set] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
        return dataloaders

    def train_dataloader(self):
        train_loaders = self.setup_dataloaders(self.train_datasets)
        return CombinedLoader(train_loaders, mode="min_size")

    def val_dataloader(self):
        val_loaders = self.setup_dataloaders(self.val_datasets)
        return CombinedLoader(val_loaders, mode="sequential")
    
    def get_samples(self, split: Literal["train", "val"], amount: int) -> dict[frozenset, dict[str, torch.Tensor]]:
        """Fetches a specified number of samples from the specified split ('train' or 'val')."""
        if split == "train":
            data_loader = self.train_dataloader()
        else:
            data_loader = self.val_dataloader()

        for sample_data in data_loader:
            break

        # Initialize the structure to hold samples
        collected_samples = {}

        domain_data = sample_data[0]  # the dataloader is set up to return a dictionary of domain data

        if split == "train":
            # Handle each domain
            for domains, data in domain_data.items():
                if domains not in collected_samples:
                    collected_samples[domains] = {}
                for domain_name, tensors in data.items():
                    if domain_name == "image_latents":
                        # Special handling for image latents to fetch from ImageNet
                        imagenet_loader = DataLoader(self.imagenet_train_dataset if split == "train" else self.imagenet_val_dataset, batch_size=amount, shuffle=True)
                        images_tensor = next(iter(imagenet_loader))[0].to('cuda' if torch.cuda.is_available() else 'cpu')
                        collected_samples[domains][domain_name] = images_tensor
                    else:
                        limited_amount = min(amount, tensors.size(0))
                        collected_samples[domains][domain_name] = tensors[:limited_amount]
        else:
            # This should include both 'image_latents' and 'caption_embeddings'
            combined_domain_set = frozenset(domain_data.keys())
            collected_samples[combined_domain_set] = {}
            
            for domain_name, tensors in domain_data.items():
                # Handle image latents specially for ImageNet
                if domain_name == "image_latents":
                    imagenet_loader = DataLoader(self.imagenet_val_dataset, batch_size=amount, shuffle=True)
                    images_tensor = next(iter(imagenet_loader))[0].to('cuda' if torch.cuda.is_available() else 'cpu')
                    collected_samples[combined_domain_set][domain_name] = images_tensor
                    # Also add image latents under its own key
                    individual_set = frozenset([domain_name])
                    collected_samples[individual_set] = {domain_name: images_tensor}
                else:
                    limited_amount = min(amount, tensors.size(0))
                    collected_samples[combined_domain_set][domain_name] = tensors[:limited_amount]
                    # Also add other domains under their own keys
                    individual_set = frozenset([domain_name])
                    collected_samples[individual_set] = {domain_name: tensors[:limited_amount]}

        """print("Sampled Data Overview:")
        for domain_set, domain_data in collected_samples.items():
            print(f"Domains: {domain_set}")
            for key, tensor in domain_data.items():
                print(f"  {key}: {tensor.shape}")"""
                

        return collected_samples












def make_datasets():

    # Load training data
    image_latents_train = load_data(IMAGE_LATENTS_PATH_TRAIN)
    caption_embeddings_train = load_data(CAPTION_EMBEDDINGS_PATH_TRAIN)

    # Load validation data
    image_latents_val = load_data(IMAGE_LATENTS_PATH_VAL)
    caption_embeddings_val = load_data(CAPTION_EMBEDDINGS_PATH_VAL)

    
    print("image_latents_train : ", image_latents_train.shape)
    print("caption_embeddings_train : ", caption_embeddings_train.shape)

    # Shuffle training data
    train_len = len(caption_embeddings_train)
    train_indices = np.random.permutation(train_len)
    image_latents_train = image_latents_train[train_indices]
    caption_embeddings_train = caption_embeddings_train[train_indices]

    # Shuffle validation data
    val_len = len(caption_embeddings_val)
    val_indices = np.random.permutation(val_len)
    image_latents_val = image_latents_val[val_indices]
    caption_embeddings_val = caption_embeddings_val[val_indices]

    # Create domain-specific and combined datasets
    train_datasets = {
        frozenset(['image_latents']): DomainDataset({'image_latents': image_latents_train}),
        frozenset(['caption_embeddings']): DomainDataset({'caption_embeddings': caption_embeddings_train}),
        frozenset(['image_latents', 'caption_embeddings']): DomainDataset({
            'image_latents': image_latents_train,
            'caption_embeddings': caption_embeddings_train
        })
    }

    val_datasets = {
        frozenset(['image_latents', 'caption_embeddings']): DomainDataset({
            'image_latents': image_latents_val,
            'caption_embeddings': caption_embeddings_val
        })
    }

    return train_datasets, val_datasets


def make_datamodule(batch_size: int = 64) -> GWDataModule:
    train_datasets, val_datasets = make_datasets()
    # Create a single GWDataModule handling both training and validation datasets
    data_module = GWDataModule(train_datasets=train_datasets, val_datasets=val_datasets, batch_size=batch_size)
    return data_module