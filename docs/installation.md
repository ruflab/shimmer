# Installation

You can directly use shimmer as a dependency to your project.
However, there is no PyPI entry yet, so you will need to install it as a git dependency:

```sh
pip install git+https://github.com/ruflab/shimmer.git@version
```

You can choose to use a specific version 
(look at the latest releases: https://github.com/ruflab/shimmer/releases) or
use the main branch for latest development:

```sh
pip install git+https://github.com/ruflab/shimmer.git@main
```

You can add the dependency directly in your `requirements.txt`:
```
shimmer@git+https://github.com/ruflab/shimmer.git@main
```

Or in your `pyproject.toml` following your package manager instructions.
This project uses [poetry](https://python-poetry.org/).

Using poetry, you can do:
```
poetry add git@github.com:ruflab/shimmer.git
```

# For contributing
If you want to contribute for bug correction or new features, follow instructions in [CONTRIBUTING.md](../CONTRIBUTING.md).

Otherwise, submit an issue.
