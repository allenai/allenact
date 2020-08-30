# Installation of supported environments

Below we provide installation instruction for a number of environments that we support.

## Installation of Minigrid

```bash
pip install gym-minigrid
```

Note that `gym-minigrid` is listed a dependency of `allenact` and it will be automatically installed
along with the framework.

## Installation of iTHOR
To install `iTHOR` for use with a machine with a screen you simply need to run:

```bash
pip install ai2thor
```

Note that `ai2thor` is listed a dependency of `allenact` and it will be automatically installed
along with the framework. The first time you will run an experiment with `iTHOR` (or any script that uses `ai2thor`)
the library will download all of the assets it requires to render the scenes 

To install `iTHOR` for use with a machine without a screen (such as a remote server) you will also need to
run a script that launches `xserver` with the following command:

```bash
sudo python scripts/startx.py
```

Notice that you need to run the command with `sudo`. If you do not have `sudo` 
access (for example if you are running this on a shared university machine) you
can ask your administrator to run it for you. You only need to run it once (as
long as you do not turn off your machine).

## Installation of RoboTHOR
To install `RoboTHOR` for use with a machine with a screen you simply need to run:

```bash
pip install ai2thor
```

Note that `ai2thor` is listed a dependency of `allenact` and it will be automatically installed
along with the framework. The first time you will run an experiment with `RoboTHOR` (or any script that uses `ai2thor`)
the library will download all of the assets it requires to render the scenes 

To install `RoboTHOR` for use with a machine without a screen (such as a remote server) you will also need to
run a script that launches `xserver` with the following command:

```bash
sudo python scripts/startx.py
```

Notice that you need to run the command with `sudo`. If you do not have `sudo` 
access (for example if you are running this on a shared university machine) you
can ask your administrator to run it for you. You only need to run it once (as
long as you do not turn off your machine).

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
 
Unfortunately we cannot legally redistribute the Gibson scene by including them, by default, within the above
container. Instead you will need to download these yourself using the
 [instructions provided by the authors](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md#download).
These scene assets should placed into the the `dataset` within the above container.
You can then proceed to run your experiments using `allenact` as you normally would.
