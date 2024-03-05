import torch

from shimmer.types import LatentsDomainGroupT


def group_batch_size(x: LatentsDomainGroupT) -> int:
    for val in x.values():
        return val.size(0)
    raise ValueError("Got empty group.")


def group_device(x: LatentsDomainGroupT) -> torch.device:
    for val in x.values():
        return val.device
    raise ValueError("Got empty group.")
