from pathlib import Path
from typing import NamedTuple

import torch
import torch.utils.data

from shimmer.modules import DomainModule

PROJECT_DIR = Path(__file__).resolve().parent.parent


class DummyData(NamedTuple):
    vec: torch.Tensor


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size: int, domains: list[str]):
        self.size = size
        self.domains = domains

        self.data = torch.randn(size, len(self.domains), 128)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            domain: DummyData(vec=self.data[idx, i])
            for i, domain in enumerate(self.domains)
        }


class DummyDomainModule(DomainModule):
    def encode(self, x: DummyData) -> torch.Tensor:
        return x.vec

    def decode(self, z: torch.Tensor) -> DummyData:
        return DummyData(vec=z)
