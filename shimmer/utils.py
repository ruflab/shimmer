import torch

from shimmer.types import LatentsDomainGroupsT, LatentsDomainGroupT


def group_batch_size(x: LatentsDomainGroupT) -> int:
    for val in x.values():
        return val.size(0)
    raise ValueError("Got empty group.")


def groups_batch_size(domain_latents: LatentsDomainGroupsT) -> int:
    """Get the batch size of the batch.

    Args:
        domain_latents (`LatentsDomainGroupsT`): the batch of groups.

    Returns:
        int: the batch size.
    """
    for data in domain_latents.values():
        for tensor in data.values():
            return tensor.size(0)
    raise ValueError("Empty batch.")


def group_device(x: LatentsDomainGroupT) -> torch.device:
    for val in x.values():
        return val.device
    raise ValueError("Got empty group.")
