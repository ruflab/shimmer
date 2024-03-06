from typing import Any, Protocol

from torch.utils.data import Dataset


class _SizedDataset(Protocol):
    def __getitem__(self, k: int) -> Any: ...

    def __len__(self) -> int: ...


class RepeatedDataset(Dataset):
    """
    Dataset that cycles through its items to have a size of at least min size.
    If drop_last is True, the size will be exaclty min_size. If drop_last is False,
    the min_size ≤ size < min_size + len(dataset).
    """

    def __init__(self, dataset: _SizedDataset, min_size: int, drop_last: bool = False):
        """
        Args:
            dataset (SizedDataset): dataset to repeat. The dataset should have a size
                (where `__len__` is defined).
            min_size (int): minimum size of the final dataset
            drop_last (bool): whether to remove overflow when repeating the
                dataset.
        """
        self.dataset = dataset
        assert min_size >= len(self.dataset)
        self.dataset_size = len(self.dataset)
        if drop_last:
            self.total_size = min_size
        else:
            self.total_size = (
                min_size // self.dataset_size + int(min_size % self.dataset_size > 0)
            ) * self.dataset_size

    def __len__(self) -> int:
        """
        Size of the dataset. Will be min_size if drop_last is True.
        Otherwise, min_size ≤ size < min_size + len(dataset).
        """
        return self.total_size

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % self.dataset_size]
