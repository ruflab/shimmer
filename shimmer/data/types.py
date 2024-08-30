from collections.abc import Sized
from dataclasses import dataclass
from typing import Protocol


class SizedDataset(Sized, Protocol):
    def __getitem__(self, index): ...


@dataclass(frozen=True)
class DomainDesc:
    base: str
    kind: str
