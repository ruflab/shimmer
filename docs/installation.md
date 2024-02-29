# Installation

You can directly use shimmer as a dependency to your project.
However, there is no PyPI entry yet, so you will need to install it as a git dependency:

```sh
pip install git+https://github.com/bdvllrs/shimmer.git@version
```

You can choose to use a specific version 
(look at the latest releases: https://github.com/bdvllrs/shimmer/releases) or
use the main branch for latest development:

```sh
pip install git+https://github.com/bdvllrs/shimmer.git@main
```

You can add the dependency directly in your `requirements.txt`:
```
shimmer@git+https://github.com/bdvllrs/shimmer.git@main
```

Or in your `pyproject.toml` following your package manager instructions.
This project uses [poetry](https://python-poetry.org/).

Using poetry, you can add:
```toml
shimmer = {git = "git@github.com:bdvllrs/shimmer.git", rev = "main"}
```

in the `tool.poetry.dependencies` section.

# For contributing
If you want to contribute for bug correction or new features, follow instructions in [CONTRIBUTING.md](CONTRIBUTING.md).

Otherwise, submit an issue.
