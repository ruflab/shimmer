from collections.abc import Mapping
from typing import Any, Literal

import torch

RawDomainGroupT = Mapping[str, Any]
"""
Matched raw unimodal data from multiple domains.
Keys of the mapping are domains names and values are the domain data.

All values in the mapping should be matched and represent the same information.

Example:
    ```python
    def fun(domain_group: RawDomainGroupT): ...


    x = {
        "vision": PIL.Image.Image("path/to/dog/picture.png"),
        "language": "This is a picture of a dog.",
    }

    fun(x)
    ```

Note:
    This type uses `collections.abc.Mapping` and is used for functions' inputs.
    Use `RawDomainGroupDT` for functions' outputs.

    This allows to be more generic and allow passing other mappings.
"""

RawDomainGroupDT = dict[str, Any]
"""
Output type version of `RawDomainGroupT`.
Matched raw unimodal data from multiple domains.
Keys of the mapping are domains names and values are the domain data.

Example:
    ```python
    def fun() -> RawDomainGroupDT:
        return {
            "vision": PIL.Image.Image("path/to/dog/picture.png"),
            "language": "This is a picture of a dog.",
        }

    ```

Note:
    This type uses `dict`s and is used for functions' outputs.
    Use `RawDomainGroupT` for functions' inputs.

"""

LatentsDomainGroupT = Mapping[str, torch.Tensor]
"""
Matched unimodal latent representations from multiple domains.
Keys of the mapping are domains names and values are `torch.Tensor` latent
representation of the domain.

Example:
    ```python
    def fun(domain_group: LatentsDomainGroupT): ...


    x = {
        "vision": torch.Tensor([0.0, 1.0, 0.0, ...]),
        "language": torch.Tensor([0.0, 0.3, 0.2, ...]),
    }

    fun(x)
    ```

Note:
    This type uses `collections.abc.Mapping` and is used for functions' inputs.
    Use `LatentsDomainGroupDT` for functions' outputs.

    This allows to be more generic and allow passing other mappings.
"""

LatentsDomainGroupDT = dict[str, torch.Tensor]
"""
Matched unimodal latent representations from multiple domains.
Keys of the dict are domains names and values are `torch.Tensor` latent
representation of the domain.

Example:
    ```python
    def fun() -> LatentsDomainGroupDT:
        return {
            "vision": torch.Tensor([0.0, 1.0, 0.0, ...]),
            "language": torch.Tensor([0.0, 0.3, 0.2, ...]),
        }

    ```

Note:
    This type uses `dict`s and is used for functions' outputs.
    Use `LatentsDomainGroupT` for functions' inputs.
"""

RawDomainGroupsT = Mapping[frozenset[str], RawDomainGroupT]
"""
Mapping of `RawDomainGroupT`. Keys are frozenset of domains matched in the group.
Each group is independent and contains different data (unpaired).

Example:
    ```python
    def fun() -> RawDomainGroupsDT:
        return {
            frozenset(["vision"]): {
                "vision": PIL.Image.Image("path/to/cat/picture.png"),
            },
            frozenset(["language"]): {
                "language": "This is a picture of a rabbit.",
            },
            frozenset(["vision", "language"]): {
                "vision": PIL.Image.Image("path/to/dog/picture.png"),
                "language": "This is a picture of a dog.",
            },
        }

    ```

Note:
    This type uses `dict`s and is used for functions' outputs.
    Use `RawDomainGroupsT` for functions' inputs.
"""

RawDomainGroupsDT = dict[frozenset[str], RawDomainGroupDT]
"""
Mapping of `RawDomainGroupT`. Keys are frozenset of domains matched in the group.
Each group is independent and contains different data (unpaired).

Example:
    ```python
    def fun() -> RawDomainGroupsDT:
        return {
            frozenset(["vision"]): {
                "vision": PIL.Image.Image("path/to/cat/picture.png"),
            },
            frozenset(["language"]): {
                "language": "This is a picture of a rabbit.",
            },
            frozenset(["vision", "language"]): {
                "vision": PIL.Image.Image("path/to/dog/picture.png"),
                "language": "This is a picture of a dog.",
            },
        }

    ```

Note:
    This type uses `dict`s and is used for functions' outputs.
    Use `RawDomainGroupsT` for functions' inputs.
"""

LatentsDomainGroupsT = Mapping[frozenset[str], LatentsDomainGroupT]
"""
Mapping of `LatentsDomainGroupT`. Keys are frozenset of domains matched in the group.
Each group is independent and contains different data (unpaired).

Example:
    ```python
    def fun(domain_group: LatentsDomainGroupsT): ...


    x = {
        frozenset(["vision"]): {
            "vision": torch.Tensor([1.0, 0.0, 0.3, ...]),
        },
        frozenset(["language"]): {
            "language": torch.Tensor([1.0, 0.2, 0.9, ...]),
        },
        frozenset(["vision", "language"]): {
            "vision": torch.Tensor([0.0, 1.0, 0.0, ...]),
            "language": torch.Tensor([0.0, 0.3, 0.2, ...]),
        },
    }

    fun(x)
    ```
Note:
    This type uses `collections.abc.Mapping` and is used for functions' inputs.
    Use `LatentsDomainGroupsDT` for functions' outputs.

    This allows to be more generic and allow passing other mappings.

"""

LatentsDomainGroupsDT = dict[frozenset[str], LatentsDomainGroupDT]
"""
Mapping of `LatentsDomainGroupDT`.
Keys are frozenset of domains matched in the group.
Each group is independent and contains different data (unpaired).

Example:
    ```python
    def fun() -> LatentsDomainGroupsDT:
        return {
            frozenset(["vision"]): {
                "vision": torch.Tensor([1.0, 0.0, 0.3, ...]),
            },
            frozenset(["language"]): {
                "language": torch.Tensor([1.0, 0.2, 0.9, ...]),
            },
            frozenset(["vision", "language"]): {
                "vision": torch.Tensor([0.0, 1.0, 0.0, ...]),
                "language": torch.Tensor([0.0, 0.3, 0.2, ...]),
            },
        }

    ```

Note:
    This type uses `dict`s and is used for functions' outputs.
    Use `LatentsDomainGroupT` for functions' inputs.
"""


ModelModeT = Literal["train", "val", "test", "val/ood", "test/ood"]
"""
Mode used by pytorch lightning (train/val, ...).

When validating or testing in out-of-distribution data, "val/ood" or "test/ood" mode is
used.
"""
