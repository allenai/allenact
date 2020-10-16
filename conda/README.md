# Conda envionment files (experimental)

This below is currently experimental and is provided without an guaranteed continued support.

This folder contains YAML files specifying [Conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
compatible with AllenAct. These environment files include: 

* `environment-base.yml` - A base environment file to be used on machines where GPU support is not needed (everything
 will be run on the CPU).
* `environment-<CUDA_VERSION>.yml` - where `<CUDA_VERSION>` is the CUDA version used on your machine.

## Installing a Conda environment

If you are unfamiliar with Conda, please familiarize yourself with their [introductory documentation](https://docs.conda.io/projects/conda/en/latest/).
If you have not already, you will need to first [install Conda (i.e. Anaconda or Miniconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
on your machine. We suggest installing [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary)
as it's relatively lightweight.

For the moment let's assume you're using `environment-base.yml` above. To install a conda environment with name `allenact`
 using this file you can simply run the following (this will take a few minutes):

```bash
conda env create --file ./conda/environment-base.yml --name allenact
``` 
The above is very simple but has the side effect of creating a new `src` directory where it will
place some of AllenAct's dependencies. To get around this, instead of running the above you can instead
run the commands:

```bash
export MY_ENV_NAME=allenact
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc" conda env create --file conda/environment-base.yml --name $MY_ENV_NAME
``` 

These additional commands tell conda to place these dependencies under the `${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc` directory rather
than under `src`, this is more in line with where we'd expect dependencies to be placed when running `pip install ...`.

## Using the Conda environment

Now that you've installed the conda environment as above, you can activate it by running:

```bash
conda activate allenact
```

after which you can run everything as you would normally.