# Contributing

We in the Perceptural Reasoning and Interaction Research (PRIOR) group at the
 Allen Institute for AI (AI2, @allenai) welcome contributions from the greater community. If
 you would like to make such a contributions we recommend first submitting an 
 [issue](https://github.com/allenai/embodied-rl/issues) describing your proposed improvement.
 Doing so can ensure we can validate your suggestions before you spend a great deal of time
 upon them. Small (or validated) improvements and bug fixes should be made via a pull request
 from your fork of the repository at [https://github.com/allenai/embodied-rl](https://github.com/allenai/embodied-rl).
 
All code in pull requests should adhere to the following guidelines.

## Found a bug or want to suggest an enhancement?

Please submit an [issue](https://github.com/allenai/embodied-rl/issues) in which you note the steps
to reproduce the bug or in which you detail the enhancement.

## Making a pull request?

When making a pull request we require that any code respects several guidelines detailed below.

### Auto-formatting

All python code in this repository should be formatted using [black](https://black.readthedocs.io/en/stable/).
To use `black` auto-formatting across all files, simply run
```bash
bash scripts/auto_format.sh
``` 
which will run `black` auto-formatting as well as [docformatter](https://pypi.org/project/docformatter/) (used
to auto-format documentation strings).

### Type-checking

Our code makes liberal use of type hints. If you have not had experience with type hinting in python we recommend
reading the [documentation](https://docs.python.org/3/library/typing.html) of the `typing` python module or the 
simplified introduction to type hints found [here](https://www.python.org/dev/peps/pep-0483/). All methods should
have typed arguments and output. Furthermore we use [mypy](https://mypy.readthedocs.io/en/stable/) to perform 
basic static type checking. Before making a pull request, there should be no warnings or errors when running
```bash
dmypy run -- --follow-imports=skip .
```
Explicitly ignoring type checking (for instance using `# type: ignore`) should be only be done when it would otherwise
be an extensive burden.

### Updating, adding, or removing packages?

We recommend using [pipenv](https://pipenv.kennethreitz.org/en/latest/) to keep track
of dependencies, ensure reproducibility, and keep things synchronized. If you are
doing so and have modified any installed packages please run:
```bash
pipenv-setup sync --pipfile # Syncs packages to setup.py
pip freeze > requirements.txt # Syncs packages to requirements.py
``` 
before submitting a pull request. If you are not using `pipenv`, you are still
required to update the file `Pipfile` with newly installed or modified packages. Moreover
you must manually update the `install_requires` field of the `setup.py` file. 

### Setting up pre-commit hooks (optional)

Pre-commit hooks check that, when you attempt to commit changes, your code adheres a number of
formatting and type-checking guidelines. Pull requests containing code not adhering to these 
guidelines will not be accepted and thus we recommend installing these pre-commit hooks. Assuming you have 
installed all of the project requirements, you can install our recommended
pre-commit hooks by running (from this project's root directory)
```bash
pre-commit install
```
After running the above, each time you run `git commit ...` a set of pre-commit checks will
be run.