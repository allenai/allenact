# Tutorial: Off-policy training

In this tutorial we'll learn how to train an agent from an external dataset by imitating expert actions via
Behavior Cloning. We'll use a [BabyAI agent](/api/plugins/babyai_plugin/babyai_models#BabyAIRecurrentACModel) to solve
`GoToLocal` tasks on [MiniGrid](https://github.com/maximecb/gym-minigrid); see the
`projects/babyai_baselines/experiments/go_to_local` directory for more details.

This tutorial assumes the [installation instructions](../installation/installation-allenact.md) have already been
followed and, to some extent, this framework's [abstractions](../getting_started/abstractions.md) are known.

## The task

In a `GoToLocal` task, the agent immersed in a grid world has to navigate to a specific object in the presence of
multiple distractors, requiring the agent to understand `go to` instructions like "go to the red ball". For further
details, please consult the [original paper](https://arxiv.org/abs/1810.08272).  

## Getting the dataset

We will use a large dataset (**more than 4 GB**) including expert demonstrations for `GoToLocal` tasks. To download
the data we'll run

```bash
PYTHONPATH=. python plugins/babyai_plugin/scripts/download_babyai_expert_demos.py GoToLocal
```

from the project's root directory, which will download `BabyAI-GoToLocal-v0.pkl` and `BabyAI-GoToLocal-v0_valid.pkl` to
the `plugins/babyai_plugin/data/demos` directory.

We will also generate small versions of the datasets, which will be useful if running on CPU, by calling

```bash
PYTHONPATH=. python plugins/babyai_plugin/scripts/truncate_expert_demos.py
```
from the project's root directory, which will generate `BabyAI-GoToLocal-v0-small.pkl` under the same
`plugins/babyai_plugin/data/demos` directory.

## Data iterator

In order to train with an off-policy dataset, we need to define a data `Iterator`.
The `Data Iterator` merges the functionality of the `Dataset` and `Dataloader` in PyTorch,
in that it defines the way to both sample data from the dataset and convert them into batches to be
used for training. 
An example of a `Data Iterator` for BabyAI expert demos might look as follows:
 
```python
class ExpertTrajectoryIterator(Iterator):
    def __init__(
        self,
        data: List[Tuple[str, bytes, List[int], MiniGridEnv.Actions]],
        nrollouts: int,  # equivalent to batch size in a PyTorch RNN
        rollout_len: int,  # number of steps in each rollout
        ...
    ):
        super().__init__()
        ...
        self.data = data
        self.nrollouta = nrollouts
        self.rollout_len = rollout_len
        ...

    def get_data_for_rollout_ind(self, rollout_ind: int) -> Dict[str, np.ndarray]:
        masks: List[bool] = []
        ...
        while len(masks) != self.rollout_len:
            # collect data, including an is_first_obs boolean for rollout_ind,
            # or raise StopIteration if finished
            ...
            masks.append(not is_first_obs)
            ...

        return {
            "masks": np.array(masks, dtype=np.float32).reshape(
                (self.rollout_len, 1, 1)  # steps x agent x mask
            ),
            ...
        }

    def __next__(self) -> Dict[str, torch.Tensor]:
        all_data = defaultdict(lambda: [])

        for rollout_ind in range(self.nrollouts):
            data_for_ind = self.get_data_for_rollout_ind(rollout_ind=rollout_ind)
            for key in data_for_ind:
                all_data[key].append(data_for_ind[key])

        return {
            key: torch.from_numpy(np.stack(all_data[key], axis=1))  # new sampler dim
            for key in all_data
        }
```

A complete example can be found in
[ExpertTrajectoryIterator](/api/plugins/minigrid_plugin/minigrid_offpolicy#ExpertTrajectoryIterator).

## Loss function

Off-policy losses must implement the
[AbstractOffPolicyLoss](/api/core/algorithms/offpolicy_sync/losses/abstract_offpolicy_loss/#abstractoffpolicyloss)
interface. In this case, we minimize the cross-entropy between the actor's policy and the expert action:

```python
class MiniGridOffPolicyExpertCELoss(AbstractOffPolicyLoss[ActorCriticModel]):
    def loss(
        self,
        model: ActorCriticModel,
        batch: ObservationType,
        memory: Memory,
    ) -> Tuple[torch.FloatTensor, Dict[str, float], Memory, int]:
        rollout_len, nrollouts = batch["minigrid_ego_image"].shape[:2]

        # Initialize Memory if empty
        if len(memory) == 0:
            for key in model.recurrent_memory_specification:
                dims_template, dtype = spec[key]
                # get sampler_dim and all_dims from dims_template (and nrollouts)
                ...
                memory.check_append(
                    key=key,
                    tensor=torch.zeros(
                        *all_dims,
                        dtype=dtype,
                        device=batch["minigrid_ego_image"].device
                    ),
                    sampler_dim=sampler_dim,
                )

        # Forward data (through the actor and critic)
        ac_out, memory = model.forward(
            observations=batch,
            memory=memory,
            prev_actions=None, # unused by BabyAI ActorCriticModel
            masks=batch["masks"],
        )

        # Compute the loss from the actor's output and expert action
        expert_ce_loss = -ac_out.distributions.log_probs(batch["expert_action"]).mean()

        info = {"expert_ce": expert_ce_loss.item()}

        return expert_ce_loss, info, memory, rollout_len * nrollouts
```

A complete example can be found in
[MiniGridOffPolicyExpertCELoss](/api/plugins/minigrid_plugin/minigrid_offpolicy#MiniGridOffPolicyExpertCELoss).
Note that in this case we train the entire actor, but it would also be possible to forward data through a different
subgraph of the ActorCriticModel.

## Experiment configuration

For the experiment configuration, we'll build on top of an existing
[base BabyAI GoToLocal Experiment Config](/api/projects/babyai_baselines/experiments/go_to_local/base/#basebabyaigotolocalexperimentconfig).
The complete `ExperimentConfig` file for off-policy training is
[here](/api/projects/tutorials/babyai_go_to_local_bc_offpolicy/#bcoffpolicybabyaigotolocalexperimentconfig), but let's
focus on the most relevant aspect to enable this type of training: 
providing an [OffPolicyPipelineComponent](/api/utils/experiment_utils/#offpolicypipelinecomponent) object as input to a
`PipelineStage` when instantiating the `TrainingPipeline` in the `training_pipeline` method.

```python
class BCOffPolicyBabyAIGoToLocalExperimentConfig(BaseBabyAIGoToLocalExperimentConfig):
    ...
    DATA = babyai.utils.load_demos(
               os.path.join(
                   BABYAI_EXPERT_TRAJECTORIES_DIR,
                   "BabyAI-GoToLocal-v0{}.pkl".format(
                       "" if torch.cuda.is_available() else "-small"
                   ),
               )
           )

    @classmethod
    def tag(cls):
        return "BabyAIGoToLocalBCOffPolicy"

    @classmethod
    def training_pipeline(cls, **kwargs):
        total_train_steps = int(1e7)
        num_steps = 128
        return TrainingPipeline(
            save_interval=10000,  # Save every 10000 steps (approximately)
            metric_accumulate_interval=1,
            optimizer_builder=Builder(optim.Adam, dict(lr=2.5e-4)),
            # As we don't have any on-policy losses, we set the next
            # two values to zero to ensure we don't attempt to
            # compute gradients for on-policy rollouts:
            num_mini_batch=0,
            update_repeats=0,
            num_steps=num_steps // 4,  # rollout length for tasks sampled from env.
            # Instantiate the off-policy loss
            named_losses={
                "offpolicy_expert_ce_loss": MiniGridOffPolicyExpertCELoss(),
            },
            gamma=0.99,
            use_gae=True,
            gae_lambda=1.0,
            max_grad_norm=0.5,
            advance_scene_rollout_period=None,
            pipeline_stages=[
                # Single stage, only with off-policy training
                PipelineStage(
                    loss_names=[],                                # no on-policy losses
                    max_stage_steps=total_train_steps,            # keep sampling episodes in the stage
                    # Enable off-policy training:
                    offpolicy_component=OffPolicyPipelineComponent(
                        # Pass a method to instantiate data iterators
                        data_iterator_builder=lambda **kwargs: ExpertTrajectoryIterator(
                            data=cls.DATA,
                            nrollouts=128,                        # per trainer batch size
                            rollout_len=num_steps,                # For truncated-BPTT
                            **kwargs,
                        ),
                        loss_names=["offpolicy_expert_ce_loss"],  # off-policy losses
                        updates=16,                               # 16 batches per rollout
                    ),
                ),
            ],
        )
```
You'll have noted that it is possible to combine on-policy and off-policy training in the same stage, even though here
we apply pure off-policy training.

## Training

We recommend using a machine with a CUDA-capable GPU for this experiment. In order to start training, we just need to
invoke

```bash
python main.py -b projects/tutorials babyai_go_to_local_bc_offpolicy -m 8 -o <OUTPUT_PATH>
```

Note that with the `-m 8` option we limit to 8 the number of on-policy task sampling processes used between off-policy
updates.

If everything goes well, the training success should quickly reach values around 0.7-0.8 on GPU and converge to values
close to 1 if given sufficient time to train.

If running tensorboard, you'll notice a separate group of scalars named `offpolicy` with losses, approximate frame rate
and other tracked values in addition to the standard `train` used for on-policy training.

A view of the training progress about 5 minutes after starting on a CUDA-capable GPU should look similar to

![off-policy progress](/img/offpolicy_training_tutorial.jpg)
