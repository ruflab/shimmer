from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

# TODO: Consider handling CPU usage
# with a workaround in:
# https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662


_T = TypeVar("_T")


class DataDomain(ABC, Generic[_T]):
    """
    Base class for a domain of the SimpleShapesDataset.
    All domains extend this base class and implement the
    __getitem__ and __len__ methods.
    """

    @abstractmethod
    def __init__(
        self,
        dataset_path: str | Path,
        split: str,
        transform: Callable[[Any], _T] | None = None,
        additional_args: dict[str, Any] | None = None,
    ) -> None:
        """
        Params:
            dataset_path (str | pathlib.Path): Path to the dataset.
            split (str): The split of the dataset to use. One of "train", "val", "test".
            transform (Any -> Any): Optional transform to apply to the data.
            additional_args (dict[str, Any]): Optional additional arguments to pass
                to the domain.
        """
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> _T: ...
