# plugins.babyai_plugin.babyai_models [[source]](https://github.com/allenai/embodied-rl/tree/master/plugins/babyai_plugin/babyai_models.py)

## BabyAIACModelWrapped
```python
BabyAIACModelWrapped(
    self,
    obs_space: Dict[str, int],
    action_space: gym.spaces.discrete.Discrete,
    image_dim = 128,
    memory_dim = 128,
    instr_dim = 128,
    use_instr = False,
    lang_model = 'gru',
    use_memory = False,
    arch = 'cnn1',
    aux_info = None,
    include_auxiliary_head: bool = False,
)
```

### forward_once
```python
BabyAIACModelWrapped.forward_once(self, obs, memory, instr_embedding=None)
```
Copied (with minor modifications) from
`babyai.model.ACModel.forward(...)`.
