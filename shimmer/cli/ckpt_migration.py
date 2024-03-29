from collections.abc import Sequence
from pathlib import Path

import click

from shimmer.utils import migrate_model


@click.command("migrate-ckpt")
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path, file_okay=True, dir_okay=False),
)
def migrate_ckpt(paths: Sequence[Path]):
    """
    Script to migrate a list of checkpoints.
    This can be called with:
    ```sh
    shimmer migrate-ckpt PATH_1 PATH_2 ... PATH_N
    ```
    where paths point to checkpoints.

    Internally, this calls `shimmer.utils.migrate_model` for each of the given paths.
    """
    for path in paths:
        migrate_model(path)
