# Installation of supported environments

Below we provide installation instruction for a number of environments that we support.

## Installation of Minigrid
`pip install gym-minigrid`

## Installation of iTHOR
To install `iTHOR` for use with a machine with a screen you simply need to run:

`pip install ai2thor`

Note that first time you will run an experiment with `iTHOR` (or any script that uses `ai2thor`)
the library will download all of the assets it requires to render the scenes 

To install `iTHOR` for use with a machine without a screen (such as a remote server) you will also need to
run a script that launches `xserver` with the following command:

`sudo python scripts/startx.py`

Notice that you need to run the command with `sudo`. If you do not have `sudo` 
access (for example if you are running this on a shared university machine) you
can ask your administrator to run it for you. You only need to run it once (as
long as you do not turn off your machine).

## Installation of RoboTHOR
To install `RoboTHOR` for use with a machine with a screen you simply need to run:

`pip install ai2thor`

Note that first time you will run an experiment with `RoboTHOR` (or any script that uses `ai2thor`)
the library will download all of the assets it requires to render the scenes 

To install `RoboTHOR` for use with a machine without a screen (such as a remote server) you will also need to
run a script that launches `xserver` with the following command:

`sudo python scripts/startx.py`

Notice that you need to run the command with `sudo`. If you do not have `sudo` 
access (for example if you are running this on a shared university machine) you
can ask your administrator to run it for you. You only need to run it once (as
long as you do not turn off your machine).

## Installation of Habitat

To run experiments using Habitat please use our docker image using the following command:

`docker pull klemenkotar/allenact-habitat:latest`

This container includes the 0.1 release of `allenact`, the 0.15 release of `habitat` as well
as the `Gibson` point navigation dataset and scene assets. You then need to launch the container 
and attach into it:

`docker run klemenkotar/allenact-habitat --runtime=nvidia -it`

Once inside the container activate the conda environment:

`conda activate allenact`
 
The proceed to run your experiments using `allenact` as you normally would.
