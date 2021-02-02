# Installation of AllenAct

**Note 1:** This library has been tested *only in python 3.6*. The following assumes you have a working
version of *python 3.6* installed locally. 

**Note 2:** If you are installing `allenact` intending to use a GPU for training/inference and your
current machine uses an older version of CUDA you may need to manually install the version of 
PyTorch that supports your CUDA version. In such a case, after installing the below requirements, you
should follow the directions for installing PyTorch with older
versions of CUDA available on the [PyTorch homepage](https://pytorch.org/).

In order to install `allenact` and/or its requirements we recommend creating a new
[python virtual environment](https://docs.python.org/3/tutorial/venv.html) and installing all
of the below requirements into this virtual environment.

### Installing requirements with `pip`

There are three main installation paths regarding whether you are interested in installing

1. Just the main `allenact` framework (without any plugin).
1. The main framework and one or more of the available plugins.
1. The entire library, e.g. for development of more advanced features like a new training algorithm.

Additionally, we also provide instructions to pip install from github.

#### Standalone framework

```bash
pip install allenact
```

#### Framework and plugins

To install `allenact` and all available plugins, run

```bash
pip install allenact allenact_plugins[all]
```

or, for a specific plugin, like for example `ithor_plugin`:

```bash
pip install allenact allenact_plugins[ithor]
```

#### Full library

Clone the repository to your local machine and move into the top-level directory

```bash
git clone git@github.com:allenai/allenact.git
cd allenact
```

All requirements may be installed by running the following command:

```bash
pip install -r requirements.txt; pip install -r dev_requirements.txt
```

or, alternatively, using the experimental `conda` setup described below.

##### Plugins extra requirements

To install the specific requirements of each plugin, we need to additionally call

```bash
pip install -r allenact_plugins/<PLUGIN_NAME>_plugin/extra_requirements.txt
```

from the top-level directory.

#### Bleeding edge pip install

To install the latest `allenact` framework, you can use

```bash
pip install -e "git+https://github.com/allenai/allenact.git@master#egg=allenact&subdirectory=allenact"
```

and, similarly, you can also use

```bash
pip install -e "git+https://github.com/allenai/allenact.git@master#egg=allenact_plugins[all]&subdirectory=allenact_plugins"
```

to install all plugins.

Depending on your machine configuration, you may need to use `pip3` instead of `pip` in the commands
above.

### Installing requirements with `pipenv`

If you have already installed [`pipenv`](https://pipenv.kennethreitz.org/en/latest/), you may
run the following to install all requirements.

```bash
pipenv install --skip-lock -r requirements.txt
```

Please see the documentation of `pipenv` to understand how to use the newly created virtual environment.

### Installing requirements with `conda`

Clone the repository to your local machine and move into the top-level directory

```bash
git clone git@github.com:allenai/allenact.git
cd allenact
```

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