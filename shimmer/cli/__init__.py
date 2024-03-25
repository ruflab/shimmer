import click

from .ckpt_migration import migrate_ckpt


@click.group()
def cli():
    pass


cli.add_command(migrate_ckpt)
