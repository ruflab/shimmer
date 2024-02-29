from collections.abc import Mapping
from typing import Literal

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import DataLoader, Dataset, TensorDataset

from shimmer import RepeatedDataset


def get_domain_data(
    domain_name: Literal["domain1", "domain2"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load data from a domain.
    Here we randomly create it
    """
    n_train = 256
    n_val = 128
    if domain_name == "domain1":
        train_data = torch.randn(n_train, 8)
        val_data = torch.randn(n_val, 8)
        return train_data, val_data
    if domain_name == "domain2":
        train_data = torch.randn(n_train, 16)
        val_data = torch.randn(n_val, 16)
        return train_data, val_data


class DomainDataModule(LightningDataModule):
    def __init__(
        self,
        val_dataset: torch.Tensor,
        train_dataset: torch.Tensor,
        batch_size: int,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size

        self.n_train = 128
        self.n_val = 128
        self.n_paired = 64

        self.val_dataset = TensorDataset(val_dataset)
        self.train_dataset = TensorDataset(train_dataset)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class DomainDataset(Dataset):
    def __init__(self, domain_data: Mapping[str, torch.Tensor]) -> None:
        """
        Creates a dataset that takes a mapping from a domain name to a tensor of data.
        """
        self.domain_data = domain_data

    def __len__(self) -> int:
        # All domains should have the same length in one dataset!
        for domain in self.domain_data.values():
            return domain.size(0)
        return 0

    def __getitem__(self, k: int) -> dict[str, torch.Tensor]:
        # There can be several domains to return,
        # so we always return a dict with the item.
        return {domain_name: data[k] for domain_name, data in self.domain_data.items()}


def make_datasets(
    domain_data: Mapping[str, torch.Tensor],
    paired_items: list[int] | None = None,
    add_unpaired_dataset: bool = False,
) -> dict[frozenset[str], DomainDataset]:
    """
    This will create a dataset for each domain, and one for paired items.
    Args:
        domain_data: a mapping from domain name to data
        paired_items: a list of data index corresponding to paired items. If None
            everything is paired
        add_unpaired_dataset: whether we add datasets with only unpaired data
    Returns:
        a dict of frozenset containing the domains in the dataset to a dataset
        (frozenset can be used for dict keys but not sets because they are
        immutable)
    """
    datasets: dict[frozenset[str], DomainDataset] = {}

    # First, we make the paired dataset
    paired_data = {
        domain_name: data[paired_items] for domain_name, data in domain_data.items()
    }
    datasets[frozenset(domain_data.keys())] = DomainDataset(paired_data)

    if add_unpaired_dataset:
        # Then, we create unpaired dataset that only contain one domain
        for domain_name, data in domain_data.items():
            datasets[frozenset([domain_name])] = DomainDataset({domain_name: data})
    return datasets


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

    def train_dataloader(self) -> CombinedLoader:
        assert self.train_datasets is not None

        dataloaders: dict[frozenset[str], DataLoader] = {}
        max_sized_dataset = max(
            len(dataset) for dataset in self.train_datasets.values()
        )
        for domain, dataset in self.train_datasets.items():
            dataloaders[domain] = DataLoader(
                # RepeatedDataset will artificially extend the dataset
                # to be of size "max_sized_dataset" by cycling over its element.
                # This is done so that every dataset has the same length.
                RepeatedDataset(dataset, max_sized_dataset, drop_last=False),
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=True,
                shuffle=True,
                drop_last=True,
            )
        # This Loader will retrieve a batch from each dataloader at each batch.
        # Epoch will end when the smallest dataloader is consumed.
        return CombinedLoader(dataloaders, mode="min_size")

    def val_dataloader(self) -> CombinedLoader:
        assert self.val_datasets is not None

        dataloaders: dict[frozenset[str], DataLoader] = {}
        for domain, dataset in self.val_datasets.items():
            dataloaders[domain] = DataLoader(
                dataset,
                pin_memory=True,
                batch_size=self.batch_size,
                num_workers=0,
            )
        # This time we use the sequential mode which train with a batch
        # of each dataloader one after the other.
        # `dataloaders` should only have one element anyway.
        return CombinedLoader(dataloaders, mode="sequential")
