import numpy as np
import torch
from torch.utils.data import DataLoader

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

def collate_fn(batch: List[Dict[Union[str, frozenset], Dict[str, torch.Tensor]]]) -> Dict[Union[str, frozenset], Dict[str, torch.Tensor]]:
    """
    Custom collate function to handle nested dictionary structures in the batch.
    """
    collated_batch = {}
    for key in batch[0].keys():
        if isinstance(key, frozenset):
            collated_batch[key] = {sub_key: torch.stack([b[key][sub_key] for b in batch]) for sub_key in batch[0][key].keys()}
        else:
            collated_batch[key] = torch.stack([b[key] for b in batch])
    return collated_batch

def test_domain_dataset():
    # Load datasets
    image_latents_train = load_data(IMAGE_LATENTS_PATH_TRAIN)
    caption_embeddings_train = load_data(CAPTION_EMBEDDINGS_PATH_TRAIN)
    image_latents_val = load_data(IMAGE_LATENTS_PATH_VAL)
    caption_embeddings_val = load_data(CAPTION_EMBEDDINGS_PATH_VAL)

    # Initialize DomainDataset
    train_data = {'image_latents': image_latents_train, 'caption_embeddings': caption_embeddings_train}
    val_data = {'image_latents': image_latents_val, 'caption_embeddings': caption_embeddings_val}
    
    train_dataset = DomainDataset(train_data)
    val_dataset = DomainDataset(val_data)

    # Check individual item retrieval
    train_item = train_dataset[0]
    val_item = val_dataset[0]

    assert 'image_latents' in train_item, "Missing image_latents in train item."
    assert 'caption_embeddings' in train_item, "Missing caption_embeddings in train item."
    assert frozenset(['image_latents', 'caption_embeddings']) in train_item, "Missing combined domains in train item."

    print("Training data item:", train_item)
    print("Validation data item:", val_item)

    # Initialize DataLoader with the custom collate function
    train_loader = DataLoader(train_dataset, batch_size=100, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=100, collate_fn=collate_fn)

    # Test batch retrieval
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    # Print concise summaries of the batches
    print("Training batch:")
    for key, value in train_batch.items():
        if isinstance(key, frozenset):
            print(f"  Combined domain {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value[:3,:3]}")
        else:
            print(f"  {key}: {value[:3,:3]}")

    print("Validation batch:")
    for key, value in val_batch.items():
        if isinstance(key, frozenset):
            print(f"  Combined domain {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value[:3,:3]}")
        else:
            print(f"  {key}: {value[:3,:3]}")


    # Assertions to ensure batch structure
    assert 'image_latents' in train_batch, "Missing image_latents in train batch."
    assert 'caption_embeddings' in train_batch, "Missing caption_embeddings in train batch."
    assert frozenset(['image_latents', 'caption_embeddings']) in train_batch, "Missing combined domains in train batch."

    print("Dataset and DataLoader tests passed successfully!")

# Execute the test
test_domain_dataset()
