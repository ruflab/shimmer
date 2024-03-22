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
    for path in paths:
        migrate_model(path)
