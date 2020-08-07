# projects.pointnav_baselines.models.point_nav_models [[source]](https://github.com/allenai/embodied-rl/tree/master/projects/pointnav_baselines/models/point_nav_models.py)

## ResnetDualTensorGoalEncoder
```python
ResnetDualTensorGoalEncoder(
    self,
    observation_spaces: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    rgb_resnet_preprocessor_uuid: str,
    depth_resnet_preprocessor_uuid: str,
    goal_dims: int = 32,
    resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
    combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
) -> None
```

### get_object_type_encoding
```python
ResnetDualTensorGoalEncoder.get_object_type_encoding(
    self,
    observations: Dict[str, torch.FloatTensor],
) -> torch.FloatTensor
```
Get the object type encoding from input batched observations.
## ResnetTensorGoalEncoder
```python
ResnetTensorGoalEncoder(
    self,
    observation_spaces: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    resnet_preprocessor_uuid: str,
    goal_dims: int = 32,
    resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
    combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
) -> None
```

### get_object_type_encoding
```python
ResnetTensorGoalEncoder.get_object_type_encoding(
    self,
    observations: Dict[str, torch.FloatTensor],
) -> torch.FloatTensor
```
Get the object type encoding from input batched observations.
## ResnetTensorPointNavActorCritic
```python
ResnetTensorPointNavActorCritic(
    self,
    action_space: gym.spaces.discrete.Discrete,
    observation_space: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    rgb_resnet_preprocessor_uuid: Optional[str] = None,
    depth_resnet_preprocessor_uuid: Optional[str] = None,
    hidden_size: int = 512,
    goal_dims: int = 32,
    resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
    combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
)
```

### get_object_type_encoding
```python
ResnetTensorPointNavActorCritic.get_object_type_encoding(
    self,
    observations: Dict[str, torch.FloatTensor],
) -> torch.FloatTensor
```
Get the object type encoding from input batched observations.
### is_blind
True if the model is blind (e.g. neither 'depth' or 'rgb' is an
input observation type).
### num_recurrent_layers
Number of recurrent hidden layers.
### recurrent_hidden_state_size
The recurrent hidden state size of the model.
