from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from shimmer.version import __version__

Config = DictConfig | ListConfig


@dataclass
class ShimmerInfoConfig:
    version: str
    debug: bool
    cli: Any


def load_config(
    path: str | Path,
    load_dirs: list[str] | None = None,
    structure: Any = None,
    use_cli: bool = True,
    debug_mode: bool = False,
) -> Config:
    config_path = Path(path)
    if not config_path.is_dir():
        raise FileNotFoundError(f"Config path {config_path} does not exist.")

    if not (config_path / "local").is_dir():
        (config_path / "local").mkdir(exist_ok=True)

    if not (config_path / "debug").is_dir():
        (config_path / "debug").mkdir(exist_ok=True)

    if load_dirs is not None:
        load_dirs = ["default"] + load_dirs + ["local"]
    else:
        load_dirs = ["default", "local"]

    if debug_mode:
        load_dirs.append("debug")

    configs: list[Config] = []
    if structure is not None:
        configs.append(OmegaConf.structured(structure))

    for dir in load_dirs:
        path_dir = config_path / dir
        if not path_dir.exists():
            raise FileNotFoundError(
                f"Config directory {path_dir} does not exist."
            )
        for config in path_dir.iterdir():
            if config.is_file() and config.suffix == ".yaml":
                configs.append(OmegaConf.load(config.resolve()))

    if use_cli:
        cli_conf = OmegaConf.from_cli()
        configs.append(cli_conf)
    else:
        cli_conf = OmegaConf.create()

    configs.append(
        OmegaConf.create(
            {
                "__shimmer__": {
                    "version": __version__,
                    "debug": debug_mode,
                    "cli": cli_conf,
                }
            }
        )
    )

    config = OmegaConf.merge(*configs)

    return config
