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

To run experiments using Habitat please use our docker image using the following command:

```bash
docker pull klemenkotar/allenact-habitat:latest
```

This container includes the 0.1.0 release of `allenact`, the 0.1.5 release of `habitat` as well
as the `Gibson` point navigation dataset. This dataset consists of a set of start and goal positions provided by habitat.
You then need to launch the container and attach into it:

```bash
docker run klemenkotar/allenact-habitat --runtime=nvidia -it
```

Once inside the container activate the conda environment:

```bash
conda activate allenact
```
 
Unfortunately we cannot legally redistribute the Gibson scenes by including them, by default, within the above
container. Instead you will need to download these yourself using the
 [instructions provided by the authors](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md#download).
These scene assets should placed into the `dataset` within the above container.
You can then proceed to run your experiments using `allenact` as you normally would.
