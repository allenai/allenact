# Installation of supported environments

In general, each supported environment can be installed by just following the instructions to
[install the full library and specific requirements of every plugin](../installation/installation-allenact.md#full-library)
either [via pip](../installation/installation-allenact.md#installing-requirements-with-pip) or
[via Conda](../installation/installation-allenact.md#installing-a-conda-environment).

Below we provide additional installation instructions for a number of environments that we support and
provide some guidance for problems commonly experienced when using these environments.

## Installation of iTHOR (`ithor` plugin)

The first time you will run an experiment with `iTHOR` (or any script that uses `ai2thor`)
the library will download all of the assets it requires to render the scenes automatically.
However, the datasets must be manually downloaded as described [here](../installation/download-datasets.md).

**Trying to use `iTHOR` on a machine without an attached display?** 

**Note:** These instructions assume you have
[installed the full library](../installation/installation-allenact.md#full-library).

If you wish to run `iTHOR` on a machine without an attached display (for instance, a remote server such as an AWS
 machine) you will also need to run a script that launches `xserver` processes on your GPUs. This can be done
 with the following command:

```bash
sudo python scripts/startx.py &
```

Notice that you need to run the command with `sudo` (i.e. administrator privileges). If you do not have `sudo` 
access (for example if you are running this on a shared university machine) you
can ask your administrator to run it for you. You only need to run it once (as
long as you do not turn off your machine).

## Installation of RoboTHOR (`robothor` plugin)

`RoboTHOR` is installed in the same way as `iTHOR`. For more information see the above section on installing `iTHOR`. 

## Installation of Habitat

Installing habitat requires 

1. Installing the `habitat-lab` and `habitat-sim` packages.
   - This may be done by either following the [directions provided by Habitat themselves](https://github.com/facebookresearch/habitat-lab#installation)
or by using our `conda` installation instructions below. 
1. Downloading the scene assets (i.e. the Gibson or Matterport scene files) relevant to whichever task you're interested in.
   - Unfortunately we cannot legally distribute these files to you directly. Instead you will need to download these
     yourself. See [here](https://github.com/facebookresearch/habitat-lab#Gibson) for how you can download 
     the Gibson files and [here](https://github.com/facebookresearch/habitat-lab#matterport3d) for directions on
     how to download the Matterport flies.
1. Downloading the dataset files for the task you're interested in (e.g. PointNav, ObjectNav, etc).
   - See [here](https://github.com/facebookresearch/habitat-lab#task-datasets) for links to these dataset files.
 
<!--
### Using Docker

To run experiments using Habitat please use our docker image using the following command:

```bash
docker pull allenact/allenact:latest
```

This container includes the 0.1.0 release of `allenact`, the 0.1.5 release of `habitat` as well
as the `Gibson` point navigation dataset. This dataset consists of a set of start and goal positions provided by habitat.
You then need to launch the container and attach into it:

```bash
docker run --runtime=nvidia -it allenact/allenact
```
If you are running the container on a machine without an Nvidia GPU, omit the `--runtime=nvidia` flag.

Once inside the container simply `cd` into the `allenact` directory where all the allenact and habitat code should be stored:
 
Unfortunately we cannot legally redistribute the Gibson scenes by including them in the above container.
Instead you will need to download these yourself by filling out 
[this form](https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform)
and downloading the `gibson_habitat_trainval` data. Extract the scene assets (`.glb` files) into `habitat-lab/data/scene_datasets/` 
within the above container. You can then proceed to run your experiments using `allenact` as you normally would.
-->

### Using `conda`

Habitat has recently released the option to install their simulator using `conda` which avoids having
to manually build dependencies or use Docker. This does not guarantee that the installation process
is completely painless (it is difficult to avoid all possible build issues) but we've found it
to be a nice alternative to using Docker. To use this installation option please first
install an AllenAct `conda` environment using the instructions available [here](../installation/installation-allenact.md#installing-a-conda-environment).
After installing this environment, you can then install `habitat-sim` and `habitat-lab` by running:

If you are on a machine with an attached display:
```bash
export MY_ENV_NAME=allenact
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc" conda env update --file allenact_plugins/habitat_plugin/extra_environment.yml --name $MY_ENV_NAME
```

If you are on a machine without an attached display (e.g. a server), replace the last command by:
```bash
PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc" conda env update --file allenact_plugins/habitat_plugin/extra_environment_headless.yml --name $MY_ENV_NAME
```

After these steps, feel free to proceed to download the required scene assets and task-specific dataset files as
described above.

<!--
#### Installing a Conda environment

_If you are unfamiliar with Conda, please familiarize yourself with their [introductory documentation](https://docs.conda.io/projects/conda/en/latest/).
If you have not already, you will need to first [install Conda (i.e. Anaconda or Miniconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
on your machine. We suggest installing [Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary)
as it's relatively lightweight._

Clone the `allenact` repository to your local machine and move into the top-level directory

```bash
git clone git@github.com:allenai/allenact.git
cd allenact
```

The `conda` folder contains YAML files specifying [Conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
compatible with AllenAct. These environment files include: 

* `environment-base.yml` - A base environment file to be used on machines where the version of CUDA on your machine
matches the one of the latest `cudatoolkit` in conda.
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
export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc"
conda env create --file ./conda/environment-base.yml --name $MY_ENV_NAME
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

##### Using the Conda environment

Now that you've installed the conda environment as above, you can activate it by running:

```bash
conda activate allenact
```

after which you can run everything as you would normally.
-->