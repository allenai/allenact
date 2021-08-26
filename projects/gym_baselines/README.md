# Baseline models Gym (for MuJoCo environments)

This project contains the code for training baseline models for the tasks under the [MuJoCo](https://gym.openai.com/envs/#mujoco) group of Gym environments, included ["Ant-v2"](https://gym.openai.com/envs/Ant-v2/), ["HalfCheetah-v2"](https://gym.openai.com/envs/HalfCheetah-v2/), ["Hopper-v2"](https://gym.openai.com/envs/Hopper-v2/), ["Humanoid-v2"](https://gym.openai.com/envs/Humanoid-v2/), ["InvertedDoublePendulum-v2"](https://gym.openai.com/envs/InvertedDoublePendulum-v2/), ["InvertedPendulum-v2"](https://gym.openai.com/envs/InvertedPendulum-v2/), [Reacher-v2](https://gym.openai.com/envs/InvertedDoublePendulum-v2/), ["Swimmer-v2"](https://gym.openai.com/envs/Swimmer-v2/), and [Walker2d-v2"](https://gym.openai.com/envs/Walker2d-v2/).

Provided are experiment configs for training a lightweight implementation with separate MLPs for actors and critic, [MemorylessActorCritic](https://allenact.org/api/allenact_plugins/gym_plugin/gym_models/#memorylessactorcritic), with a [Gaussian distribution](https://allenact.org/api/allenact_plugins/gym_plugin/gym_distributions/#gaussiandistr) to sample actions for all continuous-control environments under the `MuJoCo` group of `Gym` environments. 

The experiments are set up to train models using the [DD-PPO](https://arxiv.org/pdf/1911.00357.pdf)
Reinforcement Learning Algorithm. 

To train an experiment run the following command from the `allenact` root directory:

```bash
python main.py <PATH_TO_EXPERIMENT_CONFIG> -o <PATH_TO_OUTPUT>
```

Where `<PATH_TO_OUTPUT>` is the path of the directory where we want the model weights
and logs to be stored and `<PATH_TO_EXPERIMENT_CONFIG>` is the path to the python file containing
the experiment configuration. An example usage of this command would be:

```bash
python main.py projects/gym_baselines/experiments/mujoco/gym_mujoco_ant_ddppo.py -o /YOUR/DESIRED/MUJOCO/OUTPUT/SAVE/PATH/gym_mujoco_ant_ddppo
```

This trains a lightweight implementation with separate MLPs for actors and critic with a Gaussian distribution to sample actions in the "Ant-v2" environment, and stores the model weights and logs
to `/YOUR/DESIRED/MUJOCO/OUTPUT/SAVE/PATH/gym_mujoco_ant_ddppo`.

## Results

In our experiments, the rewards for MuJoCo environments we obtained after training using PPO are similar to those reported by OpenAI Gym Baselines(1M steps). The Humanoid environment is compared with the original PPO paper where training 50M steps using PPO. Due to the time constraint, we only tested our baseline across two seeds so far. 


| Environment           | Gym Baseline Reward | Ours Reward |
| -----------           | ------------------- | ----------- |
|[Ant-v2](https://gym.openai.com/envs/Ant-v2/)| 1083.2 |1098.6(reached 4719 in 25M steps)  | 
| [HalfCheetah-v2](https://gym.openai.com/envs/HalfCheetah-v2/) | 1795.43             |  1741(reached 4019 in 18M steps)           |
|[Hopper-v2](https://gym.openai.com/envs/Hopper-v2/)|2316.16|2266|
|[Humanoid-v2](https://gym.openai.com/envs/Humanoid-v2/)|4000+|4500+(reached 6500 in 70M steps)|
| [InvertedPendulum-v2](https://gym.openai.com/envs/InvertedPendulum-v2/) | 809.43              |  1000       |
|[Reacher-v2](https://gym.openai.com/envs/Reacher-v2/)|-6.71|-7.045|
|[Swimmer-v2](https://gym.openai.com/envs/Swimmer-v2/)|111.19|124.7|
|[Walker2d](https://gym.openai.com/envs/Walker2d-v2/)|3424.95|2723 in 10M steps|
