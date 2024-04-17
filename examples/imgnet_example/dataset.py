from collections.abc import Mapping
from typing import Literal

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, TensorDataset

from shimmer import RepeatedDataset

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Literal



# Paths to data
IMAGE_LATENTS_PATH = "/home/rbertin/pyt_scripts/full_imgnet/full_size/vae_full_withbigger__disc/image_embeddings.npy"
CAPTION_EMBEDDINGS_PATH = "/home/rbertin/pyt_scripts/BLIP_TEST/instruct/full_imgnet/bge_fullsized_captions.npy"

def load_data(file_path: str) -> torch.Tensor:
    """
    Load data from the specified numpy file path.
    """
    data = torch.from_numpy(np.load(file_path))
    return data

class DomainDataModule(LightningDataModule):
    def __init__(self, val_dataset: torch.Tensor, train_dataset: torch.Tensor, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.val_dataset = TensorDataset(val_dataset)
        self.train_dataset = TensorDataset(train_dataset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

class DomainDataset(Dataset):
    def __init__(self, domain_data: dict[str, torch.Tensor]) -> None:
        """
        Initializes a dataset that takes a mapping from domain names to tensors of data.
        """
        self.domain_data = domain_data

    def __len__(self) -> int:
        return self.domain_data[next(iter(self.domain_data))].size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {domain_name: data[index] for domain_name, data in self.domain_data.items()}

class GWDataModule(LightningDataModule):
    def __init__(self, train_datasets: dict[frozenset[str], Dataset], val_datasets: dict[frozenset[str], Dataset], batch_size: int):
        super().__init__()
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return {domains: DataLoader(dataset, batch_size=self.batch_size, shuffle=True) for domains, dataset in self.train_datasets.items()}

    def val_dataloader(self) -> DataLoader:
        return {domains: DataLoader(dataset, batch_size=self.batch_size) for domains, dataset in self.val_datasets.items()}

def prepare_datasets():
    # Load data
    image_latents = load_data(IMAGE_LATENTS_PATH)
    caption_embeddings = load_data(CAPTION_EMBEDDINGS_PATH)

    # Assuming the data is already split into training and validation sets
    # Here we are simply slicing the tensors for example purposes
    train_images = image_latents[:1000000]
    val_images = image_latents[1000000:]
    train_captions = caption_embeddings[:1000000]
    val_captions = caption_embeddings[1000000:]

    # Create datasets
    train_dataset = DomainDataset({'image_latents': train_images, 'caption_embeddings': train_captions})
    val_dataset = DomainDataset({'image_latents': val_images, 'caption_embeddings': val_captions})

    # Create data modules
    batch_size = 64
    gw_train_data_module = GWDataModule({frozenset(['image_latents', 'caption_embeddings']): train_dataset}, batch_size=batch_size)
    gw_val_data_module = GWDataModule({frozenset(['image_latents', 'caption_embeddings']): val_dataset}, batch_size=batch_size)

    return gw_train_data_module, gw_val_data_module

if __name__ == "__main__":
    train_data_module, val_data_module = prepare_datasets()
    # Use `train_data_module` and `val_data_module` for training and validation respectively
