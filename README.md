# EmbodiedRL

## Please use hyper parameters from this readme. With other hyper parameters things might not work (it's RL after all)!

EmbodiedRL is a library designed for research in embodied reinforcement learning with
a focus on modularity, flexibility, and low cognitive overhead. This work builds upon
the [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) 
library of Ilya Kostrikov and uses some data structures from FAIR's 
[habitat-api](https://github.com/facebookresearch/habitat-api).

Currently, three RL algorithms are implemented
* Advantage Actor Critic (A2C), see [A3C](https://arxiv.org/pdf/1602.01783v1.pdf)
* [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)

Also see the OpenAI posts: [A2C/ACKTR](https://blog.openai.com/baselines-acktr-a2c/) and [PPO](https://blog.openai.com/openai-baselines-ppo/) for more information.

This implementation is inspired by the OpenAI baselines for [A2C](https://github.com/openai/baselines/tree/master/baselines/a2c), [ACKTR](https://github.com/openai/baselines/tree/master/baselines/acktr) and [PPO](https://github.com/openai/baselines/tree/master/baselines/ppo1). It uses the same hyper parameters and the model since they were well tuned for Atari games.

## Requirements

This library has been tested only in Python 3.6. In order to install requirements, we recommend creating a new virutal environment and then run the following command

```bash
pip install -r requirements.txt
```

Alternatively, if you use [pipenv](https://pipenv.kennethreitz.org/en/latest/):

```bash
pipenv install --skip-lock
```

## Contributions
