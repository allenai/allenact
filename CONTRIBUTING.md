# Contributing

## Development workflow

### Setting up pre-commit hooks

Assuming you have installed all of the above requirements, you can install our recommended
pre-commit hooks by running (from this project's root directory)
```
pre-commit install
```
After running the above, each time you run `git commit ...` a set of pre-commit checks will
be run. These include basic type-checking and auto-formatting (via
[black](https://black.readthedocs.io/en/stable/)). If your code passes these checks, the 
commit will succeed, otherwise it will provide error messages directing you to problem
areas. 

### Updating, adding, or removing packages

We recommend using [pipenv](https://pipenv.kennethreitz.org/en/latest/) to keep track
of dependencies, ensure reproducibility, and keep things synchronized. If you have
modified any installed packages please run:
```bash
pipenv-setup sync --pipfile # Syncs packages to setup.py
pip freeze > requirements.txt # Syncs packages to requirements.py
``` 
before submitting a pull request.