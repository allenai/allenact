# Experiment specification

Let's look at an example experiment configuration for an object navigation example with an actor-critic agent observing
RGB images from the environment, target object classes from the task and expert actions.

The interface to be implemented by the experiment specification is defined in
[rl_base.experiment_config](/api/rl_base/experiment_config#experimentconfig). The first method to implement is `tag`,
which provides a string identifying the experiment:
```python
class ObjectNavThorExperimentConfig(rl_base.experiment_config.ExperimentConfig):
    @classmethod
    def tag(cls):
        return "ObjectNavThor"
    ...
```

## Model creation

Next, `create_model` will be used to instantiate
[object navigation baseline actor-critic models](/api/models/object_nav_models#ObjectNavBaselineActorCritic):
```python
class ObjectNavThorExperimentConfig(rl_base.experiment_config.ExperimentConfig):
    ...
    SCREEN_SIZE = 224
    OBJECT_TYPES = sorted(["Tomato"])
    SENSORS = [
        rl_ai2thor.ai2thor_sensors.RGBSensorThor(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        rl_ai2thor.ai2thor_sensors.GoalObjectTypeThorSensor(
            {"object_types": OBJECT_TYPES}
        ),
        rl_base.sensor.ExpertActionSensor({"nactions": 6}),
    ]
    
    @classmethod
    def create_model(cls, **kwargs) -> torch.nn.Module:
        return models.object_nav_models.ObjectNavBaselineActorCritic(
            action_space=gym.spaces.Discrete(
                len(rl_ai2thor.object_nav.tasks.ObjectNavTask.action_names())
            ),
            observation_space=rl_base.sensor.SensorSuite(
                cls.SENSORS
            ).observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            hidden_size=512,
            object_type_embedding_dim=8,
        )
    ...
```

## Training pipeline

We can implement a training pipeline including dataset aggregation
([DAGGER](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)), by assuming the existence of an
expert (implemented in the task definition) that provides optimal actions to agents. Different stages can combine in
multiple ways PPO and imitation losses:
```python
class ObjectNavThorExperimentConfig(rl_base.experiment_config.ExperimentConfig):
    ...
    @classmethod
    def training_pipeline(cls, **kwargs):
        dagger_steps = int(3e4)
        ppo_steps = int(1e6)
        lr = 2.5e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 16
        log_interval = 100 * num_steps
        save_interval = 10 * log_interval
        gamma = 0.99
        use_gae = True
        gae_lambda = 1.0
        max_grad_norm = 0.5
        return utils.experiment_utils.TrainingPipeline(
            named_losses={
                "imitation_loss": utils.experiment_utils.Builder(
                    onpolicy_sync.losses.imitation.Imitation,
                ),
                "ppo_loss": utils.experiment_utils.Builder(
                    onpolicy_sync.losses.ppo.PPO,
                    default=onpolicy_sync.losses.ppo.PPOConfig,
                ),
            },
            optimizer=utils.experiment_utils.Builder(
                torch.optim.Adam, dict(lr=lr)
            ),
            save_interval=save_interval,
            log_interval=log_interval,
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            num_steps=num_steps,
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            max_grad_norm=max_grad_norm,
            pipeline_stages=[
                utils.experiment_utils.PipelineStage(
                    loss_names=["imitation_loss", "ppo_loss"],
                    teacher_forcing=utils.experiment_utils.LinearDecay(
                        startp=1.0, endp=0.0, steps=dagger_steps,
                    ),
                    end_criterion=dagger_steps,
                ),
                utils.experiment_utils.PipelineStage(
                    loss_names=["ppo_loss", "imitation_loss"],
                    end_criterion=ppo_steps
                ),
            ],
    ...
```

## Machine configuration

In `machine_params` we define machine configuration parameters that will be used for training, validation and test:
```python
class ObjectNavThorExperimentConfig(rl_base.experiment_config.ExperimentConfig):
    ...
    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        if mode == "train":
            nprocesses = 3
            gpu_ids = [] if not torch.cuda.is_available() else [0]
        elif mode in ["valid", "test"]:
            nprocesses = 1
            gpu_ids = [] if not torch.cuda.is_available() else [0]
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        return {"nprocesses": nprocesses, "gpu_ids": gpu_ids}
    ...
```

## Task sampling

TODO!!!!!
