# Defining an  experiment

Let's look at an example experiment configuration for an object navigation example with an actor-critic agent observing
RGB images from the environment and target object classes from the task. This is a simplified example where the 
agent is confined to a single `iTHOR` scene (`FloorPlan1`) and needs to find a single object (a tomato). To see how one
might running a "full"/"hard" version of navigation within AI2-THOR, see our tutorials
 [PointNav in RoboTHOR](../tutorials/training-a-pointnav-model.md) and 
 [Swapping in a new environment](../tutorials/transfering-to-a-different-environment-framework.md).

The interface to be implemented by the experiment specification is defined in
[allenact.base_abstractions.experiment_config](/api/allenact/base_abstractions/experiment_config#experimentconfig). If you'd
like to skip ahead and see the finished configuration, [see here](https://github.com/allenai/allenact/blob/master/projects/tutorials/object_nav_ithor_ppo_one_object.py).
We begin by making the following imports:

```python
from math import ceil
from typing import Dict, Any, List, Optional

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.sensor import SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact_plugins.ithor_plugin.ithor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from allenact_plugins.ithor_plugin.ithor_task_samplers import ObjectNavTaskSampler
from allenact_plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask
from projects.objectnav_baselines.models.object_nav_models import (
 ObjectNavBaselineActorCritic,
)
from allenact.utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay
```

Now first method to implement is `tag`, which provides a string identifying the experiment:

```python
class ObjectNavThorPPOExperimentConfig(ExperimentConfig):
    ...
    @classmethod
    def tag(cls):
        return "ObjectNavThorPPO"
    ...
```

## Model creation

Next, `create_model` will be used to instantiate an
[baseline object navigation actor-critic model](/api/projects/objectnav_baselines/models/object_nav_models#ObjectNavBaselineActorCritic):

```python
class ObjectNavThorExperimentConfig(ExperimentConfig):
    ...

    # A simple setting, train/valid/test are all the same single scene
    # and we're looking for a single object
    OBJECT_TYPES = ["Tomato"]
    TRAIN_SCENES = ["FloorPlan1_physics"]
    VALID_SCENES = ["FloorPlan1_physics"]
    TEST_SCENES = ["FloorPlan1_physics"]

    # Setting up sensors and basic environment details
    SCREEN_SIZE = 224
    SENSORS = [
        RGBSensorThor(
            height=SCREEN_SIZE, width=SCREEN_SIZE, use_resnet_normalization=True,
        ),
        GoalObjectTypeThorSensor(object_types=OBJECT_TYPES),
    ]
    
    ...
    
    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObjectNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNaviThorGridTask.class_action_names())),
            observation_space=SensorSuite(cls.SENSORS).observation_spaces,
            rgb_uuid=cls.SENSORS[0].uuid,
            depth_uuid=None,
            goal_sensor_uuid="goal_object_type_ind",
            hidden_size=512,
            object_type_embedding_dim=8,
        )
    ...
```

## Training pipeline

We now implement a training pipeline which trains with a single stage using PPO.

In the below we use [Builder](/api/allenact/utils/experiment_utils#builder) objects, which allow us to defer the instantiation
of objects of the class passed as their first argument while allowing passing additional keyword arguments to their
initializers. This is necessary when instantiating things like PyTorch optimizers who take as input the list of
parameters associated with our agent's model (something we can't know until the `create_model` function has been called).
 
```python
class ObjectNavThorPPOExperimentConfig(ExperimentConfig):
    ...
    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(1e6)
        lr = 2.5e-4
        num_mini_batch = 2 if not torch.cuda.is_available() else 6
        update_repeats = 4
        num_steps = 128
        metric_accumulate_interval = cls.MAX_STEPS * 10  # Log every 10 max length tasks
        save_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={
                "ppo_loss": PPO(clip_decay=LinearDecay(ppo_steps), **PPOConfig),
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps,),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )
    ...
```

Alternatively, we could use a more sophisticated pipeline that begins training with dataset aggregation
([DAgger](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)) before moving to training
with PPO. This requires the existence of an
expert (implemented in the task definition) that provides optimal actions to agents. We have implemented such a 
pipeline by extending the above configuration as follows

```python
class ObjectNavThorDaggerThenPPOExperimentConfig(ObjectNavThorPPOExperimentConfig):
    ...
    SENSORS = [
        RGBSensorThor(
            height=SCREEN_SIZE, width=SCREEN_SIZE, use_resnet_normalization=True,
        ),
        GoalObjectTypeThorSensor(object_types=OBJECT_TYPES),
        ExpertActionSensor(nactions=6), # Notice that we have added an expert action sensor.
    ]
    ...
    @classmethod
    def training_pipeline(cls, **kwargs):
        dagger_steps = int(1e4) # Much smaller number of steps as we're using imitation learning
        ppo_steps = int(1e6)
        lr = 2.5e-4
        num_mini_batch = 1 if not torch.cuda.is_available() else 6
        update_repeats = 4
        num_steps = 128
        metric_accumulate_interval = cls.MAX_STEPS * 10  # Log every 10 max length tasks
        save_interval = 10000
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5

        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=metric_accumulate_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={
                "ppo_loss": PPO(clip_decay=LinearDecay(ppo_steps), **PPOConfig),
                "imitation_loss": Imitation(), # We add an imitation loss.
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[ # The pipeline now has two stages, in the first
                              # we use DAgger (imitation loss + teacher forcing).
                              # In the second stage we no longer use teacher
                              # forcing and add in the ppo loss.
                PipelineStage(
                    loss_names=["imitation_loss"],
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=dagger_steps,
                    ),
                    max_stage_steps=dagger_steps,
                ),
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps,),
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )
``` 

A version of our experiment config file for which we have implemented this two-stage training
can be found [here](https://github.com/allenai/allenact/blob/master/projects/tutorials/object_nav_ithor_dagger_then_ppo_one_object.py).
This two-stage configuration `ObjectNavThorDaggerThenPPOExperimentConfig` is actually implemented _as a subclass of `ObjectNavThorPPOExperimentConfig`_.
This is a common pattern used in AllenAct and lets one skip a great deal of boilerplate when defining a new
experiment as a slight modification of an old one. Of course one must then be careful: changes to the superclass
configuration will propagate to all subclassed configurations. 

## Machine configuration

In `machine_params` we define machine configuration parameters that will be used for training, validation and test:
```python
class ObjectNavThorPPOExperimentConfig(allenact.base_abstractions.experiment_config.ExperimentConfig):
    ...
    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        num_gpus = torch.cuda.device_count()
        has_gpu = num_gpus != 0 

        if mode == "train":
            nprocesses = 20 if has_gpu else 4
            gpu_ids = [0] if has_gpu else []
        elif mode == "valid":
            nprocesses = 1
            gpu_ids = [1 % num_gpus] if has_gpu else []
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [0] if has_gpu else []
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        return {"nprocesses": nprocesses, "gpu_ids": gpu_ids}
    ...
```
In the above we use the availability of cuda (`torch.cuda.device_count() !=  0`) to determine whether
we should use parameters appropriate for local machines or for a server. We might optionally add a list of
`sampler_devices` to assign devices (likely those not used for running our agent) to task sampling workers.

## Task sampling

The above has defined the model we'd like to use, the types of losses we wish to use during training,
and the machine specific parameters that should be used during training. Critically we have not yet
defined which task we wish to train our agent to complete. This is done by implementing the 
`ExperimentConfig.make_sampler_fn` function
```python
class ObjectNavThorPPOExperimentConfig(ExperimentConfig):
    ...
    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavTaskSampler(**kwargs)
    ...
```
Now, before training starts, our trainer will know to generate a collection of task
samplers using `make_sampler_fn` for training (and possibly validation or testing).
The `kwargs` parameters used in the above function call can be different for each
training process, we implement such differences using the
`ExperimentConfig.train_task_sampler_args` function
```python
class ObjectNavThorPPOExperimentConfig(ExperimentConfig):
    ...
    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.TRAIN_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = "manual"
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res
    ...
```
Now training process `i` out of `n` total processes will be instantiated with the parameters
`ObjectNavThorPPOExperimentConfig.train_task_sampler_args(i, n, ...)`. Similar functions
 (`valid_task_sampler_args` and `test_task_sampler_args`) exist for generating validation
 and test parameters. Note also that with this function we can assign devices to run
 our environment for each worker. See the documentation of `ExperimentConfig` for more information.
 

## Running the experiment

We are now in the position to run the experiment (with seed 12345) using the command
```bash
python main.py object_nav_ithor_ppo_one_object -b projects/tutorials -s 12345
```
