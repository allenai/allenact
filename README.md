# EmbodiedRL

EmbodiedRL is a library designed for research in embodied reinforcement learning with
a focus on modularity, flexibility, and low cognitive overhead. This work builds upon
the [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) 
library of Ilya Kostrikov and uses some data structures from FAIR's 
[habitat-api](https://github.com/facebookresearch/habitat-api).

Currently, two RL algorithms are implemented
* Advantage Actor Critic (A2C) - synchronized [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)

## Requirements

This library has been tested only in Python 3.6. In order to install requirements, we
 recommend creating a new virtual environment and then running the following command:

```bash
pip install -r requirements.txt
```

Alternatively (*recommended*), if you use [pipenv](https://pipenv.kennethreitz.org/en/latest/):

```bash
pipenv install --skip-lock --dev
```

## Contributions

### Updating, adding, or removing packages

We recommend using [pipenv](https://pipenv.kennethreitz.org/en/latest/) to keep track
of dependencies, ensure reproducibility, and keep things synchronized. If you have
modified any installed packages please run:
```bash
pipenv run pipenv-setup sync --pipfile # Syncs packages to setup.py
pipenv run pip freeze > requirements.txt # Syncs packages to requirements.py
``` 
before submitting a pull request.