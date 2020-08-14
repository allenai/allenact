# plugins.robothor_plugin.robothor_models [[source]](https://github.com/allenai/allenact/tree/master/plugins/robothor_plugin/robothor_models.py)

## ResnetFasterRCNNTensorsGoalEncoder
```python
ResnetFasterRCNNTensorsGoalEncoder(
    self,
    observation_spaces: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    resnet_preprocessor_uuid: str,
    detector_preprocessor_uuid: str,
    class_dims: int = 32,
    max_dets: int = 3,
    resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
    box_embedder_hidden_out_dims: Tuple[int, int] = (128, 32),
    class_embedder_hidden_out_dims: Tuple[int, int] = (128, 32),
    combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
) -> None
```

### get_object_type_encoding
```python
ResnetFasterRCNNTensorsGoalEncoder.get_object_type_encoding(
    self,
    observations: Dict[str, torch.FloatTensor],
) -> torch.FloatTensor
```
Get the object type encoding from input batched observations.
## ResnetFasterRCNNTensorsObjectNavActorCritic
```python
ResnetFasterRCNNTensorsObjectNavActorCritic(
    self,
    action_space: gym.spaces.discrete.Discrete,
    observation_space: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    resnet_preprocessor_uuid: str,
    detector_preprocessor_uuid: str,
    rnn_hidden_size = 512,
    goal_dims: int = 32,
    max_dets: int = 3,
    resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
    box_embedder_hidden_out_dims: Tuple[int, int] = (128, 32),
    class_embedder_hidden_out_dims: Tuple[int, int] = (128, 32),
    combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
)
```

### get_object_type_encoding
```python
ResnetFasterRCNNTensorsObjectNavActorCritic.get_object_type_encoding(
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
## ResnetTensorGoalEncoder
```python
ResnetTensorGoalEncoder(
    self,
    observation_spaces: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    resnet_preprocessor_uuid: str,
    class_dims: int = 32,
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
## ResnetTensorObjectNavActorCritic
```python
ResnetTensorObjectNavActorCritic(
    self,
    action_space: gym.spaces.discrete.Discrete,
    observation_space: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    resnet_preprocessor_uuid: str,
    rnn_hidden_size: int = 512,
    goal_dims: int = 32,
    resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
    combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
)
```

### get_object_type_encoding
```python
ResnetTensorObjectNavActorCritic.get_object_type_encoding(
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
## ResnetTensorObjectNavActorCriticMemory
```python
ResnetTensorObjectNavActorCriticMemory(
    self,
    action_space: gym.spaces.discrete.Discrete,
    observation_space: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    resnet_preprocessor_uuid: str,
    rnn_hidden_size: int = 512,
    goal_dims: int = 32,
    resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
    combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
)
```

### get_object_type_encoding
```python
ResnetTensorObjectNavActorCriticMemory.get_object_type_encoding(
    self,
    observations: Dict[str, torch.FloatTensor],
) -> torch.FloatTensor
```
Get the object type encoding from input batched observations.
### num_recurrent_layers
Returns -1, indicating we are using a memory specification in
recurrent_hidden_state_size.
### recurrent_hidden_state_size
The memory spec of the model: A dictionary with string keys and
tuple values, each with the dimensions of the memory, e.g. (2, 32) for
two layers of 32-dimensional recurrent hidden states; an integer
indicating the index of the sampler in a batch, e.g. 1 for RNNs; the
data type, e.g. torch.float32.
