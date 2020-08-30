# Installation of AllenAct

Clone the repository to your local machine and move into the top-level directory

```bash
git clone git@github.com:allenai/allenact.git
cd allenact
```

**Note 1:** This library has been tested *only in python 3.6*. The following assumes you have a working
version of *python 3.6* installed locally. 

**Note 2:** If you are installing `allenact` intending to use a GPU for training/inference and your
current machine uses an older version of CUDA you may need to manually install the version of 
PyTorch that supports your CUDA version. In such a case, after installing the below requirements, you
should follow the directions for installing PyTorch with older
versions of CUDA available on the [PyTorch homepage](https://pytorch.org/).

In order to install requirements we recommend creating a new
[python virtual environment](https://docs.python.org/3/tutorial/venv.html) and installing all
of the below requirements into this virtual environment. Several tools exist to help manage
virtual environments, we have had success in using [`pipenv`](https://pipenv.kennethreitz.org/en/latest/)
and so provide instructions for installing the requirements using `pipenv`
but also include instructions if you would prefer to install everything directly using `pip`.

### Installing requirements with `pipenv`

If you have already installed [`pipenv`](https://pipenv.kennethreitz.org/en/latest/), you may
run the following to install all requirements.

```bash
pipenv install --skip-lock --dev
```

Please see the documentation of `pipenv` to understand how to use the newly created virtual environment.

### Installing requirements with `pip`

Note: *do not* run the following if you have already installed requirements with `pipenv`
as above. If you prefer managing your environment manually, all requirements may be installed using
 `pip` by running the following command:

```bash
pip install -r requirements.txt
```

Depending on your machine configuration, you may need to use `pip3` instead of `pip` in the
above.

### Installing supported environments

We also provide installation instructions for the environments supported in AllenAct [here](../installation/installation-framework.md).