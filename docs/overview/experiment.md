# Experiment specification

Let's look at an example experiment configuration for an object navigation example with an actor-critic agent observing
RGB images from the environment and target object classes from the task.

The interface to be implemented by the experiment specification is defined in
[rl_base.experiment_config](/api/rl_base/experiment_config#experimentconfig). The first method to implement is `tag`,
which provides a string identifying the experiment:
```python
class ObjectNavThorPPOExperimentConfig(rl_base.experiment_config.ExperimentConfig):
    @classmethod
    def tag(cls):
        return "ObjectNavThorPPO"
    ...
```

## Model creation

Next, `create_model` will be used to instantiate
[object navigation baseline actor-critic models](/api/models/object_nav_models#ObjectNavBaselineActorCritic):
```python
class ObjectNavThorExperimentConfig(rl_base.experiment_config.ExperimentConfig):
    ...
    SCREEN_SIZE = 224
    ...
    OBJECT_TYPES = sorted(["Tomato"])
    ...

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

In this section we use [Builder](/api/utils/experiment_utils#builder) objects, which allow us to defer the instantiation
of objects of the class passed as their first argument while allowing passing additional keyword arguments to their
initializers. 

We can implement a training pipeline which trains with a single stage using PPO:
```python
class ObjectNavThorPPOExperimentConfig(rl_base.experiment_config.ExperimentConfig):
    ...
    @classmethod
    def training_pipeline(cls, **kwargs):
        return utils.experiment_utils.TrainingPipeline(
            named_losses={
                "ppo_loss": utils.experiment_utils.Builder(
                    onpolicy_sync.losses.ppo.PPO,
                    default=onpolicy_sync.losses.ppo.PPOConfig,
                ),
            },
            optimizer=utils.experiment_utils.Builder(
                torch.optim.Adam, dict(lr=2.5e-4)
            ),
            save_interval=10000,  # Save every 10000 steps (approximately)
            log_interval=cls.MAX_STEPS * 10,  # Log every 10 max length tasks
            num_mini_batch=1,
            update_repeats=4,
            num_steps=128,
            gamma=0.99,
            use_gae=True,
            gae_lambda=1.0,
            max_grad_norm=0.5,
            pipeline_stages=[
                utils.experiment_utils.PipelineStage(
                    loss_names=["ppo_loss"],
                    end_criterion=int(1e6)
                ),
            ],
        )
    ...
```

Alternatively, we could use a more complicated pipeline that includes dataset aggregation
([DAgger](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)). This requires the existence of an
expert (implemented in the task definition) that provides optimal actions to agents. We have implemented such a 
pipeline by extending the above configuration as follows

```python
class ObjectNavThorPPOExperimentConfig(ObjectNavThorPPOExperimentConfig):
    ...
    SENSORS = [
        RGBSensorThor(
            {
                "height": SCREEN_SIZE,
                "width": SCREEN_SIZE,
                "use_resnet_normalization": True,
            }
        ),
        GoalObjectTypeThorSensor({"object_types": OBJECT_TYPES}),
        ExpertActionSensor({"nactions": 6}), # Notice that we have added
                                             # an expert action sensor.
    ]
    ...
    @classmethod
    def training_pipeline(cls, **kwargs):
        dagger_steps = int(3e4)
        return TrainingPipeline(
            save_interval=10000,  # Save every 10000 steps (approximately)
            log_interval=cls.MAX_STEPS * 10,  # Log every 10 max length tasks
            optimizer=Builder(optim.Adam, dict(lr=2.5e-4)),
            num_mini_batch=6 if not torch.cuda.is_available() else 30,
            update_repeats=4,
            num_steps=128,
            named_losses={
                "imitation_loss": Builder(Imitation,), # We add an imitation loss.
                "ppo_loss": Builder(PPO, default=PPOConfig,),
            },
            gamma=0.99,
            use_gae=True,
            gae_lambda=1.0,
            max_grad_norm=0.5,
            pipeline_stages=[ # The pipeline now has two stages, in the first
                              # we use DAgger (imitation loss + teacher forcing).
                              # In the second stage we no longer use teacher
                              # forcing and add in the ppo loss.
                PipelineStage(
                    loss_names=["imitation_loss"],
                    teacher_forcing=LinearDecay(
                        startp=1.0, endp=0.0, steps=dagger_steps,
                    ),
                    end_criterion=dagger_steps,
                ),
                PipelineStage(
                    loss_names=["ppo_loss", "imitation_loss"],
                    end_criterion=int(3e4)
                ),
            ],
        )
``` 
Note that, in order for the saved configs in the experiment output folder to be fully usable, we currently need
to import the module with the parent experiment config relative to the current location. 

## Machine configuration

In `machine_params` we define machine configuration parameters that will be used for training, validation and test:
```python
class ObjectNavThorPPOExperimentConfig(rl_base.experiment_config.ExperimentConfig):
    ...
    @classmethod
    def machine_params(cls, mode="train", **kwargs):
        on_server = torch.cuda.is_available()
        if mode == "train":
            nprocesses = 6 if not on_server else 20
            gpu_ids = [] if not on_server else [0]
        elif mode == "valid":
            nprocesses = 0
            gpu_ids = [] if not on_server else [1]
        elif mode == "test":
            nprocesses = 1
            gpu_ids = [] if not on_server else [0]
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        return {"nprocesses": nprocesses, "gpu_ids": gpu_ids}
    ...
```
In the above we use the availability of cuda (`torch.cuda.is_available()`) to determine whether
we should use parameters appropriate for local machines or for a server. We might optionally add a list of
`sampler_devices` to assign devices (likely those not used for running our agent) to task sampling workers.

## Task sampling

The above has defined the model we'd like to use, the types of losses we wish to use during training,
and the machine specific parameters that should be used during training. Critically we have not yet
defined which task we wish to train our agent to complete. This is done by implementing the 
`ExperimentConfig.make_sampler_fn` function
```python
class ObjectNavThorPPOExperimentConfig(rl_base.experiment_config.ExperimentConfig):
    ...
    @staticmethod
    def make_sampler_fn(**kwargs) -> rl_base.task.TaskSampler:
        return rl_ai2thor.object_nav.task_samplers.ObjectNavTaskSampler(**kwargs)
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
        devices: typing.Optional[typing.List[int]] = None,
        seeds: typing.Optional[typing.List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> typing.Dict[str, typing.Any]:
        res = self._get_sampler_args_for_scene_split(
            self.TRAIN_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = self.SCENE_PERIOD
        res["env_args"]["x_display"] = None
        return res
    ...
```
Now training process `i` out of `n` total processes will be instantiated with the parameters
`ObjectNavThorPPOExperimentConfig.train_task_sampler_args(i, n, ...)`. Similar functions
 (`valid_task_sampler_args` and `test_task_sampler_args`) exist for generating validation
 and test parameters. Note also that with this function we can assign devices to run
 our environment for each worker. See the documentation of `ExperimentConfig` for more information.