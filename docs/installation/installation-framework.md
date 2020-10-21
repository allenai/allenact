# Installation of supported environments

Below we provide installation instructions for a number of environments that we support.

## Installation of MiniGrid

MiniGrid will automatically be installed when installing `allenact` and so nothing additional needs to be done.
 If you wish to (re)install it manually using `pip`, simply run the command:

```bash
pip install gym-minigrid
```

## Installation of iTHOR

`iTHOR` will automatically be installed when installing `allenact` and so, if you have installed `allenact`
 on a machine with an attached display, nothing additional needs to be done. If you wish to (re)install it manually 
 using `pip`, simply run the command:

```bash
pip install ai2thor
```

The first time you will run an experiment with `iTHOR` (or any script that uses `ai2thor`)
the library will download all of the assets it requires to render the scenes automatically.

**Trying to use `iTHOR` on a machine without an attached display?** 

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

## Installation of RoboTHOR

`RoboTHOR` is installed in tandem with `iTHOR` when installing the `ai2thor` library. For more information
see the above section on installing `iTHOR`. 

## Installation of Habitat

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

### Using `conda` (experimental)

The following is experimental, we do not guarantee that AllenAct will continue to support this
installation procedure in future releases.

Habitat has recently released the option to install their simulator using `conda` which avoids having
to manually build dependencies or use Docker. This does not guarantee that the installation process
is completely painless (it is difficult to avoid all possible build issues) but we've found it
to be a nice alternative to using Docker. To use this installation option please first
install an AllenAct `conda` environment using the instructions available under the _Installing a Conda environment (experimental)_
section [here](installation-allenact.md). After installing this environment, you can then install
`habitat-sim` by running:

If you are on a machine with an attached display:
```bash
conda install habitat-sim=0.1.5 -c conda-forge -c aihabitat --name allenact
```

If you are on a machine without an attached display (e.g. a server):
```bash
conda install habitat-sim=0.1.5 headless -c conda-forge -c aihabitat --name allenact
```

