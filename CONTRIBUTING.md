# Contributing Code to Shimmer

The best way to make a contribution is to first fork the repository on github, 
create a new branch with an explicit name for your change, and then create a pull request with your changes.

The project requires Python 3.11. Make sure that you have a compatible version.
Using [Pyenv](https://github.com/pyenv/pyenv) is an easy way to get the right
python version.

We use [poetry](https://python-poetry.org/) (version >= 2) as our package manager,
so make sure it is installed and you have run `poetry install` in your directory.

When you add features or change APIs, update `CHANGELOG.md` to track new changes.

With each commit, the github workflow is executed and will check for:
- code quality;
- unit tests.

## Code quality workflow
This workflow makes sure that the provided code follows correct code formatting
and linting using [ruff](https://github.com/astral-sh/ruff),
and type issues with [mypy](https://github.com/python/mypy).

This project is fully typed, so any contribution should provide typing annotations.

To run the tools locally, make sure that you have installed dependencies with dev group:
```sh
poetry sync --with dev
```

Then you can run:
```sh
poetry run ruff check --fix  # lint the project (and fix errors if ruff can)
```
```sh
poetry run ruff format  # reformat the project
```

There is a [pre-commit](https://pre-commit.com/) configuration set to lint and
reformat using ruff. Set it up to run all automated checks before each commit:

To install the pre-commit hooks:
```sh
poetry run pre-commit install
```
After this command, the ruff format and checks will be automatically done before
each commit.

You can test the hook by running it explicitely:
```sh
poetry run pre-commit run --all-files
```

> [!NOTE]
> mypy can be long to execute, so it is not ran as a pre-commit hook.
> you can run it manually, or let github run it for you when you push.

To check type issues with mypy:
```sh
poetry run mypy --install-types .
```

## Tests
We use [pytest](https://github.com/pytest-dev/pytest/).

```sh
poetry run pytest tests/
```

### Checkpoint Migration
Note that among the tests, we check that old GW checkpoints can still be migrated
automatically. If you plan on making some changes that would break old checkpoints,
please also provide a migration script in `shimmer/ckpt_migrations`.

You just need to create a new file `N_what_I_changed.py` (see
the [store migrations in a folder](https://github.com/bdvllrs/migrate-ckpt?tab=readme-ov-file#store-migrations-in-a-folder) section).
