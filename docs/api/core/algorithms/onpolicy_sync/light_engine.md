# core.algorithms.onpolicy_sync.light_engine [[source]](https://github.com/allenai/allenact/tree/master/core/algorithms/onpolicy_sync/light_engine.py)
Defines the reinforcement learning `OnPolicyRLEngine`.
## OnPolicyRLEngine
```python
OnPolicyRLEngine(
    self,
    experiment_name: str,
    config: core.base_abstractions.experiment_config.ExperimentConfig,
    results_queue: <bound method BaseContext.Queue of <multiprocessing.context.DefaultContext object at 0x101f5e910>>,
    checkpoints_queue: Optional[<bound method BaseContext.Queue of <multiprocessing.context.DefaultContext object at 0x101f5e910>>],
    checkpoints_dir: str,
    mode: str = 'train',
    seed: Optional[int] = None,
    deterministic_cudnn: bool = False,
    mp_ctx: Optional[multiprocessing.context.BaseContext] = None,
    worker_id: int = 0,
    num_workers: int = 1,
    device: Union[str, torch.device, int] = 'cpu',
    distributed_port: int = 0,
    max_sampler_processes_per_worker: Optional[int] = None,
    kwargs,
)
```
The reinforcement learning primary controller.

This `OnPolicyRLEngine` class handles all training, validation, and
testing as well as logging and checkpointing. You are not expected
to instantiate this class yourself, instead you should define an
experiment which will then be used to instantiate an
`OnPolicyRLEngine` and perform any desired tasks.

### worker_seeds
```python
OnPolicyRLEngine.worker_seeds(
    nprocesses: int,
    initial_seed: Optional[int],
) -> List[int]
```
Create a collection of seeds for workers without modifying the RNG
state.
