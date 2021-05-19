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

Alternatively, we also document how to [install a conda environment](#installing-a-conda-environment)
with all the requirements, which is especially useful if you plan to train models in [Habitat](https://aihabitat.org/).

## Different ways to use `allenact`

There are three main installation paths depending on how you wish to use `allenact`.

1. You want to use the `allenact` abstractions and training engine for your own task/environment and don't really 
care about using any of our plugins that offer additional support (in the form of models, sensors, task samplers, etc.)
for select tasks/environments like AI2-THOR, Habitat, and MiniGrid.
    - If this sounds like you, install the [standalone framework](#standalone-framework).
1. You want to use `allenact` as above but would also like to use some of our additional plugins.
    - If this sounds like you, install the [framework and plugins](#framework-and-plugins).
1. You want full access to everything in `allenact` (including all plugins and all of our projects and baselines)
   and want to have the option to edit the internal implementation of `allenact` to suit your desire. 
    - If this sounds like you, install the [full library](#full-library).   


## Standalone framework

You can install `allenact` easily using pip:

```bash
pip install allenact
```

If you'd like to install the latest development version of `allenact` (possibly unstable) directly from GitHub see the
next section.

### Bleeding edge pip install

To install the latest `allenact` framework, you can use

```bash
pip install -e "git+https://github.com/allenai/allenact.git@main#egg=allenact&subdirectory=allenact"
```

and, similarly, you can also use

```bash
pip install -e "git+https://github.com/allenai/allenact.git@main#egg=allenact_plugins[all]&subdirectory=allenact_plugins"
```

to install all plugins.

Depending on your machine configuration, you may need to use `pip3` instead of `pip` in the commands
above.

## Framework and plugins

To install `allenact` and all available plugins, run

```bash
pip install allenact allenact_plugins[all]
```

which will install `allenact` and `allenact_plugins` packages along with the requirements for _all_
of the plugins (when possible). If you only want to install the requirements for some subset of plugins, you can
specify these plugins with the `allenact_plugins[plugin1,plugin2]` notation. For instance, to install requirements
for the `ithor_plugin` and the `minigrid_plugin`, simply run:

```bash
pip install allenact allenact_plugins[ithor,minigrid]
```

A list of all available plugins can be found [here](https://github.com/allenai/allenact/tree/master/allenact_plugins).

## Full library

Clone the `allenact` repository to your local machine and move into the top-level directory

```bash
git clone git@github.com:allenai/allenact.git
cd allenact
```

Below we describe two alternative ways to install all dependencies via `pip` or `conda`.

### Installing requirements with `pip`

All requirements for `allenact` (not including plugin requirements) may be installed by running the following command:

```bash
pip install -r requirements.txt; pip install -r dev_requirements.txt
```

To install plugin requirements, see below.

#### Plugins extra requirements

To install the specific requirements of each plugin, we need to additionally call

```bash
pip install -r allenact_plugins/<PLUGIN_NAME>_plugin/extra_requirements.txt
```

from the top-level directory.

### Installing a `conda` environment

_If you are unfamiliar with Conda, please familiarize yourself with their [introductory documentation](https://docs.conda.io/projects/conda/en/latest/).
If you have not already, you will need to first [install Conda (i.e. Anaconda or Miniconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
on your machine. We suggest installing [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary)
as it's relatively lightweight._

The `conda` folder contains YAML files specifying [Conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
compatible with AllenAct. These environment files include: 

* `environment-base.yml` - A base environment file to be used on all machines (it includes
[PyTorch](https://pytorch.org/) with the latest `cudatoolkit`).
* `environment-dev.yml` - Additional dev dependencies.
* `environment-<CUDA_VERSION>.yml` - Additional dependencies, where `<CUDA_VERSION>` is the CUDA version used on your
machine (if you are using linux, you might find this version by running `/usr/local/cuda/bin/nvcc --version`).
* `environment-cpu.yml` - Additional dependencies to be used on machines where GPU support is not needed (everything
 will be run on the CPU).
 

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

If needed, you can use one of the `environment-<CUDA_VERSION>.yml` environment files to install the proper version of
the `cudatoolkit` by running:

```bash
conda env update --file ./conda/environment-<CUDA_VERSION>.yml --name allenact
```
or the CPU-only version:
```bash
conda env update --file ./conda/environment-cpu.yml --name allenact
```

#### Using the `conda` environment

Now that you've installed the conda environment as above, you can activate it by running:

```bash
conda activate allenact
```

after which you can run everything as you would normally.


#### Installing supported environments with `conda`

Each supported plugin contains a YAML environment file that can be applied upon the existing `allenact` environment. To
install the specific requirements of each plugin, we need to additionally call

```bash
PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc" conda env update --file allenact_plugins/<PLUGIN_NAME>_plugin/extra_environment.yml --name $MY_ENV_NAME
```

from the top-level directory.

**Habitat:** Note that, for habitat, we provide two environment types, regarding whether our machine is connected to a
display. More details can be found [here](../installation/installation-framework.md#installation-of-habitat). 
