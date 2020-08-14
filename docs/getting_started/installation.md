# Installation

Clone this repository to your local machine and move into the top-level directory

```bash
git clone git@github.com:allenai/allenact.git
cd allenact
```

**Note:** This library has been tested *only in python 3.6*. The following assumes you have a working
version of *python 3.6* installed locally. 

In order to install requirements we recommend using [`pipenv`](https://pipenv.kennethreitz.org/en/latest/) but also include instructions if
you would prefer to install things directly using `pip`.

### Installing requirements with `pipenv` (*recommended*)

If you have already installed [`pipenv`](https://pipenv.kennethreitz.org/en/latest/), you may
run the following to install all requirements.

```bash
pipenv install --skip-lock --dev
```

### Installing requirements with `pip`

Note: *do not* run the following if you have already installed requirements with `pipenv`
as above. If you prefer using `pip`, you may install all requirements as follows

```bash
pip install -r requirements.txt
```

Depending on your machine configuration, you may need to use `pip3` instead of `pip` in the
above.
