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
virtual environments, we have had success in using [Anaconda](https://docs.conda.io) and
[`pipenv`](https://pipenv.kennethreitz.org/en/latest/)

### Installing requirements with `pip`

Note: *do not* run the following if you have already installed requirements with `pipenv`
as above. If you prefer managing your environment manually, all requirements may be installed using
 `pip` by running the following command:

```bash
pip install -r requirements.txt
```

Depending on your machine configuration, you may need to use `pip3` instead of `pip` in the
above.

### Installing requirements with `pipenv`

If you have already installed [`pipenv`](https://pipenv.kennethreitz.org/en/latest/), you may
run the following to install all requirements.

```bash
pipenv install --skip-lock -r requirements.txt
```

Please see the documentation of `pipenv` to understand how to use the newly created virtual environment.

### Installing requirements with `conda`

The `conda` folder contains YAML files specifying [Conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
compatible with AllenAct. These environment files include: 

* `environment-base.yml` - A base environment file to be used on machines where GPU support is not needed (everything
 will be run on the CPU).
* `environment-<CUDA_VERSION>.yml` - where `<CUDA_VERSION>` is the CUDA version used on your machine (if you are using linux, you can generally find this version by running `/usr/local/cuda/bin/nvcc --version`).

#### Installing a Conda environment (experimental)

If you are unfamiliar with Conda, please familiarize yourself with their [introductory documentation](https://docs.conda.io/projects/conda/en/latest/).
If you have not already, you will need to first [install Conda (i.e. Anaconda or Miniconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
on your machine. We suggest installing [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary)
as it's relatively lightweight.

For the moment let's assume you're using `environment-base.yml` above. To install a conda environment with name `allenact`
 using this file you can simply run the following (*this will take a few minutes*):

```bash
conda env create --file ./conda/environment-base.yml --name allenact
``` 
The above is very simple but has the side effect of creating a new `src` directory where it will
place some of AllenAct's dependencies. To get around this, instead of running the above you can instead
run the commands:

```bash
export MY_ENV_NAME=allenact
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc" conda env create --file ./conda/environment-base.yml --name $MY_ENV_NAME
``` 

These additional commands tell conda to place these dependencies under the `${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc` directory rather
than under `src`, this is more in line with where we'd expect dependencies to be placed when running `pip install ...`.

##### Using the Conda environment

Now that you've installed the conda environment as above, you can activate it by running:

```bash
conda activate allenact
```

after which you can run everything as you would normally.

### Installing supported environments

We also provide installation instructions for the environments supported in AllenAct [here](../installation/installation-framework.md).