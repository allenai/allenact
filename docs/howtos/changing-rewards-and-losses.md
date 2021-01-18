# Changing rewards and losses

In order to train actor-critic agents, we need to specify

* `rewards` at the task level, and
* `losses` at the training pipeline level. 

## Rewards

We will use the [object navigation task in `iTHOR`](/api/plugins/ithor_plugin/ithor_tasks/#objectnavtask) as a 
running example. We can see how the `ObjectNaviThorGridTask._step(self, action: int) -> RLStepResult` method computes the reward for the latest 
action by invoking a function like:

```python
def judge(self) -> float:
    reward = -0.01

    if not self.last_action_success:
        reward += -0.03

    if self._took_end_action:
        reward += 1.0 if self._success else -1.0

    return float(reward)
```

Any reward shaping can be easily added by e.g. modifying the definition of an existing class:

```python
class NavigationWithShaping(allenact_plugins.ithor_plugin.ithor_tasks.ObjectNaviThorGridTask):
    def judge(self) -> float:
        reward = super().judge()
        
        if self.previous_state is not None:
            reward += float(my_reward_shaping_function(
                self.previous_state,
                self.current_state,
            ))
        
        self.previous_state = self.current_state
        
        return reward

``` 

## Losses

We support [A2C](/api/core/algorithms/onpolicy_sync/losses/a2cacktr#a2c), [PPO](/api/core/algorithms/onpolicy_sync/losses/ppo#ppo),
and [imitation](/api/core/algorithms/onpolicy_sync/losses/imitation#imitation) losses amongst others. We can easily include
[DAgger](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf) or variations thereof by assuming the
availability of an expert providing optimal actions to agents and combining imitation and PPO losses in different ways
through multiple stages:

```python
class MyExperimentConfig(allenact.base_abstractions.experiment_config.ExperimentConfig):
    ...
    @classmethod
    def training_pipeline(cls, **kwargs):
        dagger_steps = int(3e4)
        ppo_steps = int(3e4)
        ppo_steps2 = int(1e6)
        ...
        return utils.experiment_utils.TrainingPipeline(
            named_losses={
                "imitation_loss": allenact.algorithms.onpolicy_sync.losses.imitation.Imitation(),
                "ppo_loss": allenact.algorithms.onpolicy_sync.losses.ppo.PPO(
                    **allenact.algorithms.onpolicy_sync.losses.ppo.PPOConfig,
                ),
            },
            ...
            pipeline_stages=[
                utils.experiment_utils.PipelineStage(
                    loss_names=["imitation_loss", "ppo_loss"],
                    teacher_forcing=utils.experiment_utils.LinearDecay(
                        startp=1.0, endp=0.0, steps=dagger_steps,
                    ),
                    max_stage_steps=dagger_steps,
                ),
                utils.experiment_utils.PipelineStage(
                    loss_names=["ppo_loss", "imitation_loss"],
                    max_stage_steps=ppo_steps
                ),
                utils.experiment_utils.PipelineStage(
                    loss_names=["ppo_loss"],
                    max_stage_steps=ppo_steps2,
                ),
            ],
        )
```