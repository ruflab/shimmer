from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import torch

RawDomainGroupT = Mapping[str, Any]
"""
Matched raw unimodal data from multiple domains.
Keys of the mapping are domains names.
"""

RawDomainGroupDT = dict[str, Any]
"""
Matched raw unimodal data from multiple domains.
Keys of the dict are domains names.

This is a more specific version of `RawDomainGroupT` used in method's outputs.
"""

LatentsDomainGroupT = Mapping[str, torch.Tensor]
"""
Matched unimodal latent representations from multiple domains.
Keys of the mapping are domains names.
"""

LatentsDomainGroupDT = dict[str, torch.Tensor]
"""
Matched unimodal latent representations from multiple domains.
Keys of the dict are domains names.

This is a more specific version of `LatentsDomainGroupT` used in method's outputs.
"""

LatentsDomainGroupsT = Mapping[frozenset[str], LatentsDomainGroupT]
"""
Mapping of `LatentsDomainGroupT`. Keys are frozenset of domains matched in the group.
Each group is independent and contains different data (unpaired).
"""

LatentsDomainGroupsDT = dict[frozenset[str], LatentsDomainGroupDT]
"""
Mapping of `LatentsDomainGroupDT`.
Keys are frozenset of domains matched in the group.
Each group is independent and contains different data (unpaired).

This is a more specific version of `LatentsDomainGroupsT` used in method's outputs.
"""

RawDomainGroupsT = Mapping[frozenset[str], RawDomainGroupT]
"""
Mapping of `RawDomainGroupT`. Keys are frozenset of domains matched in the group.
Each group is independent and contains different data (unpaired).
"""

RawDomainGroupsDT = dict[frozenset[str], RawDomainGroupDT]
"""
Mapping of `RawDomainGroupT`. Keys are frozenset of domains matched in the group.
Each group is independent and contains different data (unpaired).

This is a more specific version of `RawDomainGroupsT` used in method's outputs.
"""

ModelModeT = Literal["train", "val", "test", "val/ood", "test/ood"]
"""Mode used by pytorch lightning.
"""
