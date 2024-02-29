# Contributing Code to Shimmer

The best way to make a contribution is to first fork the repository on github, 
create a new branch with an explicit name for your change, and then create a pull request with your changes.

We use [poetry](https://python-poetry.org/) as our package manager, 
so make sure it is installed and you have run `poetry install` in your directory.

When you add features or change APIs, update `CHANGELOG.md` to track new changes.

With each commit, the github workflow is executed and will check for:
- code quality;
- unit tests.

## Code quality workflow
This workflow makes sure that the provided code follows correct code formatting
(using [isort](https://github.com/PyCQA/isort) and [black](https://github.com/psf/black)),
[flake8](https://github.com/PyCQA/flake8), and type issues with [mypy](https://github.com/python/mypy).

To run the tools locally, make sure that you have installed dependencies with dev group:
```sh
poetry install --with=dev
```

Then you can run:
```sh
poetry run isort .
```
```sh
poetry run black .
```
```sh
poetry run flake8 .
```
```sh
poetry run mypy .
```

## Tests
We use [pytest](https://github.com/pytest-dev/pytest/).

Make sure you installed with test dependencies:
```sh
poetry install --with=test
```

Then:
```sh
poetry run pytest tests/
```
