# Baseline models Gym (for MuJoCo environments)

This project contains the code for training baseline models for the tasks under the [MuJoCo](https://gym.openai.com/envs/#mujoco) group of Gym environments, included ["Ant-v2"](https://gym.openai.com/envs/Ant-v2/), ["HalfCheetah-v2"](https://gym.openai.com/envs/HalfCheetah-v2/), ["Hopper-v2"](https://gym.openai.com/envs/Hopper-v2/), ["Humanoid-v2"](https://gym.openai.com/envs/Humanoid-v2/), ["InvertedDoublePendulum-v2"](https://gym.openai.com/envs/InvertedDoublePendulum-v2/), ["InvertedPendulum-v2"](https://gym.openai.com/envs/InvertedPendulum-v2/), [Reacher-v2](https://gym.openai.com/envs/InvertedDoublePendulum-v2/), ["Swimmer-v2"](https://gym.openai.com/envs/Swimmer-v2/), and [Walker2d-v2"](https://gym.openai.com/envs/Walker2d-v2/).

Provided are experiment configs for training a lightweight implementation with separate MLPs for actors and critic, [MemorylessActorCritic](https://allenact.org/api/allenact_plugins/gym_plugin/gym_models/#memorylessactorcritic), with a [Gaussian distribution](https://allenact.org/api/allenact_plugins/gym_plugin/gym_distributions/#gaussiandistr) to sample actions for all continuous-control environments under the `MuJoCo` group of `Gym` environments. 

The experiments are set up to train models using the [DD-PPO](https://arxiv.org/pdf/1911.00357.pdf)
Reinforcement Learning Algorithm. 

To train an experiment run the following command from the `allenact` root directory:

```bash
python main.py <PATH_TO_EXPERIMENT_CONFIG> -o <PATH_TO_OUTPUT> -c
```

Where `<PATH_TO_OUTPUT>` is the path of the directory where we want the model weights
and logs to be stored and `<PATH_TO_EXPERIMENT_CONFIG>` is the path to the python file containing
the experiment configuration. An example usage of this command would be:

```bash
python main.py projects/gym_baselines/experiments/mujoco/gym_mujoco_ant_ddppo.py -o /Users/charleszhang/Desktop/aaaaaaa/gym_mujoco_ant_ddppo
```

This trains a lightweight implementation with separate MLPs for actors and critic with a Gaussian distribution to sample actions in the "Ant-v2" environment, and stores the model weights and logs
to `storage/gym_mujoco_ant_ddppo`.

