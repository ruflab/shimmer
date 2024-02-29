from typing import Any, Protocol

from torch.utils.data import Dataset


class _SizedDataset(Protocol):
    def __getitem__(self, k: int) -> Any: ...

    def __len__(self) -> int: ...


class RepeatedDataset(Dataset):
    """
    Repeats a dataset to have at least a minimum size.
    """

    def __init__(
        self,
        dataset: _SizedDataset,
        min_size: int,
        drop_last: bool = False,
    ):
        """
        Params:
            dataset: dataset to repeat
            min_size (int): minimum amount of element in the final dataset
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
        return self.total_size

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % self.dataset_size]
