from collections.abc import Sequence
from pathlib import Path

import click
import torch
from migrate_ckpt import (
    ckpt_migration_key,
    migrate_from_folder,
)

from shimmer import MIGRATION_DIR


@click.command("migrate-ckpt")
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path, file_okay=True, dir_okay=False),
)
def migrate_ckpt(paths: Sequence[Path]):
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        new_ckpt, done_migrations = migrate_from_folder(ckpt, MIGRATION_DIR)
        done_migration_log = ", ".join(map(lambda x: x.name, done_migrations))
        print(f"Migrating: {done_migration_log}")
        if len(done_migrations) or ckpt_migration_key not in ckpt:
            version = 0
            if ckpt_migration_key in ckpt:
                version = len(ckpt[ckpt_migration_key])
            torch.save(ckpt, path.with_stem(f"{path.stem}-{version}"))
            torch.save(new_ckpt, path)
