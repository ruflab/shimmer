from os import PathLike
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from migrate_ckpt import (
    ckpt_migration_key,
    get_folder_migrations,
    migrate_from_folder,
)

from shimmer.types import LatentsDomainGroupsT, LatentsDomainGroupT

MIGRATION_DIR = Path(__file__).parent / "ckpt_migrations"


def group_batch_size(x: LatentsDomainGroupT) -> int:
    for val in x.values():
        return val.size(0)
    raise ValueError("Got empty group.")


def groups_batch_size(domain_latents: LatentsDomainGroupsT) -> int:
    """
    Get the batch size of the batch.

    Args:
        domain_latents (`LatentsDomainGroupsT`): the batch of groups.

    Returns:
        int: the batch size.
    """
    for data in domain_latents.values():
        for tensor in data.values():
            return tensor.size(0)
    raise ValueError("Empty batch.")


def groups_device(domain_latents: LatentsDomainGroupsT) -> int:
    """
    Get the batch size of the batch.

    Args:
        domain_latents (`LatentsDomainGroupsT`): the batch of groups.

    Returns:
        int: the batch size.
    """
    for data in domain_latents.values():
        for tensor in data.values():
            return tensor.device
    raise ValueError("Empty batch.")


def group_device(x: LatentsDomainGroupT) -> torch.device:
    for val in x.values():
        return val.device
    raise ValueError("Got empty group.")


def migrate_model(ckpt_path: str | PathLike, **torch_load_kwargs):
    """
    Migrates a model checkpoint

    After the migration, the given checkpoint will be migrated.
    Other versions of the checkpoint will be saved under the stem-version.suffix.

    Args:
        ckpt_path (`str | PathLike`):  path to checkpoint
        torch_load_kwargs: additional args given to torch.load.
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, **torch_load_kwargs)
    new_ckpt, done_migrations = migrate_from_folder(ckpt, MIGRATION_DIR)
    done_migration_log = ", ".join(map(lambda x: x.name, done_migrations))
    print(f"Migrating: {done_migration_log}")
    if len(done_migrations) or ckpt_migration_key not in ckpt:
        version = 0
        if ckpt_migration_key in ckpt:
            version = len(ckpt[ckpt_migration_key])
        torch.save(ckpt, ckpt_path.with_stem(f"{ckpt_path.stem}-{version}"))
        torch.save(new_ckpt, ckpt_path)


class SaveMigrations(Callback):
    def __init__(self):
        self.migrations = get_folder_migrations(MIGRATION_DIR)

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ):
        checkpoint[ckpt_migration_key] = [mig.name for mig in self.migrations]
