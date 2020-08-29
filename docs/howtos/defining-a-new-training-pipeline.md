# Defining a new training pipeline

Defining a new training pipeline, or even new learning algorithms, is straightforward with the modular design in
`AllenAct`.

A convenience [Builder](/api/utils/experiment_utils#builder) object allows us to defer the instantiation
of objects of the class passed as their first argument while allowing passing additional keyword arguments to their
initializers.

## On-policy

We can implement a training pipeline which trains with a single stage using PPO:
```python
class ObjectNavThorPPOExperimentConfig(ExperimentConfig):
    ...
    @classmethod
    def training_pipeline(cls, **kwargs):
        return TrainingPipeline(
            named_losses={
                "ppo_loss": PPO(**PPOConfig),
            },
            optimizer_builder=Builder(
                optim.Adam, dict(lr=2.5e-4)
            ),
            save_interval=10000,  # Save every 10000 steps (approximately)
            metric_accumulate_interval=1,  # Log after every step (in practice after every rollout)
            num_mini_batch=6,
            update_repeats=4,
            num_steps=128,
            gamma=0.99,
            use_gae=True,
            gae_lambda=1.0,
            max_grad_norm=0.5,
            advance_scene_rollout_period=None,
            pipeline_stages=[
                PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=int(1e6)
                ),
            ],
        )
    ...
```

Alternatively, we could use a more complex pipeline that includes dataset aggregation
([DAgger](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)). This requires the existence of an
expert (implemented in the task definition) that provides optimal actions to agents. We have implemented such a 
pipeline by extending the above configuration as follows:
```python
class ObjectNavThorDaggerPPOExperimentConfig(ExperimentConfig):
    ...
    SENSORS = [
        ...
        ExpertActionSensor({"nactions": 6}), # Notice that we have added
                                             # an expert action sensor.
    ]
    ...
    @classmethod
    def training_pipeline(cls, **kwargs):
        dagger_steps = int(3e4)
        return TrainingPipeline(
            save_interval=10000,  # Save every 10000 steps (approximately)
            metric_accumulate_interval=1,
            optimizer_builder=Builder(optim.Adam, dict(lr=2.5e-4)),
            num_mini_batch=6,
            update_repeats=4,
            num_steps=128,
            named_losses={
                "imitation_loss": Imitation(), # We add an imitation loss.
                "ppo_loss": PPO(**PPOConfig),
            },
            gamma=0.99,
            use_gae=True,
            gae_lambda=1.0,
            max_grad_norm=0.5,
            advance_scene_rollout_period=None,
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
                PipelineStage(
                    loss_names=["ppo_loss", "imitation_loss"],
                    max_stage_steps=int(3e5)
                ),
            ],
        )
``` 

## Off-policy

We can also define off-policy stages where an external dataset is used, in this case, for Behavior Cloning: 

```python
class BCOffPolicyBabyAIGoToLocalExperimentConfig(ExperimentConfig):
    ...
    @classmethod
    def training_pipeline(cls, **kwargs):
        total_train_steps = int(1e7)
        num_steps=128
        return TrainingPipeline(
            save_interval=10000,  # Save every 10000 steps (approximately)
            metric_accumulate_interval=1,
            optimizer_builder=Builder(optim.Adam, dict(lr=2.5e-4)),
            num_mini_batch=0,  # no on-policy training
            update_repeats=0,  # no on-policy training
            num_steps=num_steps // 4,  # rollouts from environment tasks
            named_losses={
                "offpolicy_expert_ce_loss": MiniGridOffPolicyExpertCELoss(
                    total_episodes_in_epoch=int(1e6)  # dataset contains 1M episodes
                ),
            },
            gamma=0.99,
            use_gae=True,
            gae_lambda=1.0,
            max_grad_norm=0.5,
            advance_scene_rollout_period=None,
            pipeline_stages=[
                PipelineStage(
                    loss_names=[],  # no on-policy losses
                    max_stage_steps=total_train_steps,
                    # We only train from off-policy data:
                    offpolicy_component=OffPolicyPipelineComponent(
                        data_iterator_builder=lambda **kwargs: create_minigrid_offpolicy_data_iterator(
                            path=DATASET_PATH,  # external dataset
                            nrollouts=128,  # per trainer batch size
                            rollout_len=num_steps,  # For truncated-BTT
                            instr_len=5,
                            **kwargs,
                        ),
                        loss_names=["offpolicy_expert_ce_loss"],  # off-policy losses
                        updates=16,  # 16 batches per rollout
                    ),
                ),
            ],
        )
```

Note that, in this example, `128 / 4 = 32` steps will be sampled from tasks in a MiniGrid environment (which can be
useful to track the agent's performance), while a subgraph of the model (in this case the entire Actor) is
trained from batches of 128-step truncated episodes sampled from an offline dataset stored under `DATASET_PATH`.
