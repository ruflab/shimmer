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


def load_data(file_path: str) -> torch.Tensor:
    """
    Load data from the specified numpy file path.
    """
    data = torch.from_numpy(np.load(file_path))
    return data


class DomainDataset(Dataset):
    def __init__(self, domain_data: Dict[str, torch.Tensor]) -> None:
        """
        Initializes a dataset that takes a mapping from domain names to tensors of data.
        Handles both individual domains and combinations as frozenset.
        """
        self.domain_data = domain_data

    def __len__(self) -> int:
        return next(iter(self.domain_data.values())).size(0)

    def __getitem__(self, index: int) -> Dict[Union[str, frozenset], Dict[str, torch.Tensor]]:
        """
        Returns a dictionary containing a single item for each domain.
        Includes a combined dictionary for domains specified as a frozenset.
        """
        result = {
            'image_latents': self.domain_data['image_latents'][index],
            'caption_embeddings': self.domain_data['caption_embeddings'][index],
            frozenset(['image_latents', 'caption_embeddings']): {
                'image_latents': self.domain_data['image_latents'][index],
                'caption_embeddings': self.domain_data['caption_embeddings'][index]
            }
        }
        return result

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
            collated_batch[domain_key] = {key: torch.stack([b[key] for b in batch])}
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
        val_dataset,
        train_dataset,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.val_dataset = val_dataset
        self.train_dataset = train_dataset

        # ImageNet dataset transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        # Paths to ImageNet directories
        train_dir = os.environ.get('DATASET_DIR', '.') + '/imagenet/train'
        val_dir = os.environ.get('DATASET_DIR', '.') + '/imagenet/val'
        # Load ImageNet datasets
        self.imagenet_train_dataset = ImageFolder(root=train_dir, transform=self.transform)
        self.imagenet_val_dataset = ImageFolder(root=val_dir, transform=self.transform)

        # Load captions datasets
        train_caption_csv = "new_captions_only.csv"
        val_caption_csv = "captions_fullimgnet_val_noshuffle.csv"
        self.train_captions_dataset = CaptionsDataset(train_caption_csv)
        self.val_captions_dataset = CaptionsDataset(val_caption_csv)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn_validation)


    def get_samples(self, split: Literal["train", "val"], amount: int) -> dict[frozenset, dict[str, torch.Tensor]]:
        """Fetches a specified number of samples from the specified split ('train' or 'val')."""
        if split == "train":
            data_loader = self.train_dataloader()
            caption_dataset = self.train_captions_dataset
        else:
            data_loader = self.val_dataloader()
            caption_dataset = self.val_captions_dataset

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
                        imagenet_loader = DataLoader(self.imagenet_train_dataset, batch_size=amount, shuffle=True)
                        images_tensor = next(iter(imagenet_loader))[0].to('cuda' if torch.cuda.is_available() else 'cpu')
                        collected_samples[domains][domain_name] = images_tensor
                    elif domain_name == "caption_embeddings":
                        # Handle caption embeddings to fetch raw text
                        caption_loader = DataLoader(self.train_captions_dataset, batch_size=amount, shuffle=True)
                        captions = [self.train_captions_dataset[idx] for idx in torch.randint(0, len(self.train_captions_dataset), (amount,))]
                        collected_samples[domains][domain_name] = captions
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
                elif domain_name == "caption_embeddings":
                    # Handle caption embeddings to fetch raw text
                    caption_loader = DataLoader(self.val_captions_dataset, batch_size=amount, shuffle=True)
                    captions = [self.val_captions_dataset[idx] for idx in torch.randint(0, len(self.val_captions_dataset), (amount,))]
                    collected_samples[combined_domain_set][domain_name] = captions
                    # Also add caption embeddings under their own key
                    individual_set = frozenset([domain_name])
                    collected_samples[individual_set] = {domain_name: captions}
                else:
                    limited_amount = min(amount, tensors.size(0))
                    collected_samples[combined_domain_set][domain_name] = tensors[:limited_amount]
                    # Also add other domains under their own keys
                    individual_set = frozenset([domain_name])
                    collected_samples[individual_set] = {domain_name: tensors[:limited_amount]}

        return collected_samples












def make_datasets(train_split: float = 0.8):
    """
    Create training and validation datasets with image and text data.
    """
    # Load training data
    image_latents_train = load_data(IMAGE_LATENTS_PATH_TRAIN)
    caption_embeddings_train = load_data(CAPTION_EMBEDDINGS_PATH_TRAIN)

    # Load validation data
    image_latents_val = load_data(IMAGE_LATENTS_PATH_VAL)
    caption_embeddings_val = load_data(CAPTION_EMBEDDINGS_PATH_VAL)

    # Initialize DomainDataset
    train_data = {'image_latents': image_latents_train, 'caption_embeddings': caption_embeddings_train}
    val_data = {'image_latents': image_latents_val, 'caption_embeddings': caption_embeddings_val}
    
    train_dataset = DomainDataset(train_data)
    val_dataset = DomainDataset(val_data)

    return train_dataset, val_dataset

def make_datamodule(train_split=0.8, batch_size: int = 64) -> GWDataModule:
    train_dataset, val_dataset = make_datasets()
    # Create a single GWDataModule handling both training and validation datasets
    data_module = GWDataModule(val_dataset=val_dataset, train_dataset=train_dataset, batch_size=batch_size)
    return data_module