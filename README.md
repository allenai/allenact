# EmbodiedRL

EmbodiedRL is a library designed for research in embodied reinforcement learning with
a focus on modularity, flexibility, and low cognitive overhead. This work builds upon
the [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) 
library of Ilya Kostrikov and uses some data structures from FAIR's 
[habitat-api](https://github.com/facebookresearch/habitat-api).

Currently, two RL algorithms are implemented

* Advantage Actor Critic (A2C) - synchronized [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)

## Table of contents

   1. [Why?](#motivation)
   1. [Installation](#installation)
   1. [Contributions](#contributions)
   1. [Citiation](#citation)

## Why?



## Installation

Begin by cloning this repository to your local machine and moving into the top-level directory

```bash
git clone git@github.com:allenai/embodied-rl.git
cd embodied-rl
```

This library has been tested only in python 3.6, the following assumes you have a working
version of python 3.6 installed locally. In order to install requirements we recommend
using [pipenv](https://pipenv.kennethreitz.org/en/latest/) but also include instructions if
you would prefer to simply install things directly using pip.

### Installing requirements with `pipenv` (*recommended*)

If you have already installed [pipenv](https://pipenv.kennethreitz.org/en/latest/), you may
simply run the following command to install all requirements.

```bash
pipenv install --skip-lock --dev
```

### Installing requirements with `pip`

Note: *do not* run the following if you have already installed requirements with `pipenv`
as above. If you prefer using `pip`, you may install all requirements as follows

```bash
pip install -r requirements.txt
```

Depending on your machine configuration, you may need to use `pip3` instead of `pip` in the
above.


## Contributions

We in the Perceptural Reasoning and Interaction Research (PRIOR) group at the
 Allen Institute for AI (AI2, @allenai) welcome contributions from the greater community. If
 you would like to make such a contributions we recommend first submitting an 
 [issue](https://github.com/allenai/embodied-rl/issues) describing your proposed improvement.
 Doing so can ensure we can validate your suggestions before you spend a great deal of time
 upon them. Small (or validated) improvements and bug fixes should be made via a pull request
 from your fork of this repository.
 
All code in this repository is subject to formatting, documentation, and type-annotation
guidelines. For more details, please see the our [contribution guidelines](./CONTRIBUTING.md).   
  
## Citiation