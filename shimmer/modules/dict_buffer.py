from collections.abc import Iterator
from typing import Any

import torch
from torch import nn


class DictBuffer(nn.Module):
    def __init__(
        self, buffer_dict: dict[str, torch.Tensor], persistent: bool = True
    ):
        super().__init__()

        self._buffer_keys = set()
        for key, value in buffer_dict.items():
            assert isinstance(value, torch.Tensor)
            self._buffer_keys.add(key)
            self.register_buffer(f"buffer_{key}", value, persistent)

    def __getitem__(self, item):
        return getattr(self, f"buffer_{item}")

    def __len__(self) -> int:
        return len(self._buffer_keys)

    def __iter__(self) -> Iterator[Any]:
        return self.keys()

    def items(self):
        for key in self._buffer_keys:
            yield key, self[key]

    def keys(self):
        yield from iter(self._buffer_keys)

    def values(self):
        for key in self._buffer_keys:
            yield self[key]

    def __contains__(self, item):
        return item in self._buffer_keys
