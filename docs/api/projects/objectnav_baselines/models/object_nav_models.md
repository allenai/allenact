# projects.objectnav_baselines.models.object_nav_models [[source]](https://github.com/allenai/allenact/tree/master/projects/objectnav_baselines/models/object_nav_models.py)
Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.

## ObjectNavActorCriticTrainResNet50RNN
```python
ObjectNavActorCriticTrainResNet50RNN(
    self,
    action_space: gym.spaces.discrete.Discrete,
    observation_space: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    hidden_size = 512,
    object_type_embedding_dim = 8,
    trainable_masked_hidden_state: bool = False,
    num_rnn_layers = 1,
    rnn_type = 'GRU',
)
```

### get_object_type_encoding
```python
ObjectNavActorCriticTrainResNet50RNN.get_object_type_encoding(
    self,
    observations: Dict[str, torch.FloatTensor],
) -> torch.FloatTensor
```
Get the object type encoding from input batched observations.
## ObjectNavBaselineActorCritic
```python
ObjectNavBaselineActorCritic(
    self,
    action_space: gym.spaces.discrete.Discrete,
    observation_space: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    hidden_size = 512,
    object_type_embedding_dim = 8,
    trainable_masked_hidden_state: bool = False,
    num_rnn_layers = 1,
    rnn_type = 'GRU',
)
```
Baseline recurrent actor critic model for object-navigation.

__Attributes__

- `action_space `: The space of actions available to the agent. Currently only discrete
    actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
- `observation_space `: The observation space expected by the agent. This observation space
    should include (optionally) 'rgb' images and 'depth' images and is required to
    have a component corresponding to the goal `goal_sensor_uuid`.
- `goal_sensor_uuid `: The uuid of the sensor of the goal object. See `GoalObjectTypeThorSensor`
    as an example of such a sensor.
- `hidden_size `: The hidden size of the GRU RNN.
- `object_type_embedding_dim`: The dimensionality of the embedding corresponding to the goal
    object type.

### forward
```python
ObjectNavBaselineActorCritic.forward(
    self,
    observations: Dict[str, torch.FloatTensor],
    rnn_hidden_states: torch.FloatTensor,
    prev_actions: torch.LongTensor,
    masks: torch.FloatTensor,
) -> Tuple[core.base_abstractions.misc.ActorCriticOutput, torch.FloatTensor]
```
Processes input batched observations to produce new actor and critic
values. Processes input batched observations (along with prior hidden
states, previous actions, and masks denoting which recurrent hidden
states should be masked) and returns an `ActorCriticOutput` object
containing the model's policy (distribution over actions) and
evaluation of the current state (value).

__Parameters__

- __observations __: Batched input observations.
- __rnn_hidden_states __: Hidden states from initial timepoints.
- __prev_actions __: Tensor of previous actions taken.
- __masks __: Masks applied to hidden states. See `RNNStateEncoder`.
__Returns__

Tuple of the `ActorCriticOutput` and recurrent hidden state.

### get_object_type_encoding
```python
ObjectNavBaselineActorCritic.get_object_type_encoding(
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
## ObjectNavResNetActorCritic
```python
ObjectNavResNetActorCritic(
    self,
    action_space: gym.spaces.discrete.Discrete,
    observation_space: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    hidden_size = 512,
    object_type_embedding_dim = 8,
    trainable_masked_hidden_state: bool = False,
    num_rnn_layers = 1,
    rnn_type = 'GRU',
)
```
Baseline recurrent actor critic model for object-navigation.

__Attributes__

- `action_space `: The space of actions available to the agent. Currently only discrete
    actions are allowed (so this space will always be of type `gym.spaces.Discrete`).
- `observation_space `: The observation space expected by the agent. This observation space
    should include (optionally) 'rgb' images and 'depth' images and is required to
    have a component corresponding to the goal `goal_sensor_uuid`.
- `goal_sensor_uuid `: The uuid of the sensor of the goal object. See `GoalObjectTypeThorSensor`
    as an example of such a sensor.
- `hidden_size `: The hidden size of the GRU RNN.
- `object_type_embedding_dim`: The dimensionality of the embedding corresponding to the goal
    object type.

### forward
```python
ObjectNavResNetActorCritic.forward(
    self,
    observations: Dict[str, torch.FloatTensor],
    rnn_hidden_states: torch.FloatTensor,
    prev_actions: torch.LongTensor,
    masks: torch.FloatTensor,
) -> Tuple[core.base_abstractions.misc.ActorCriticOutput, torch.FloatTensor]
```
Processes input batched observations to produce new actor and critic
values. Processes input batched observations (along with prior hidden
states, previous actions, and masks denoting which recurrent hidden
states should be masked) and returns an `ActorCriticOutput` object
containing the model's policy (distribution over actions) and
evaluation of the current state (value).

__Parameters__

- __observations __: Batched input observations.
- __rnn_hidden_states __: Hidden states from initial timepoints.
- __prev_actions __: Tensor of previous actions taken.
- __masks __: Masks applied to hidden states. See `RNNStateEncoder`.
__Returns__

Tuple of the `ActorCriticOutput` and recurrent hidden state.

### get_object_type_encoding
```python
ObjectNavResNetActorCritic.get_object_type_encoding(
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
## ResnetDualTensorGoalEncoder
```python
ResnetDualTensorGoalEncoder(
    self,
    observation_spaces: gym.spaces.dict.Dict,
    goal_sensor_uuid: str,
    rgb_resnet_preprocessor_uuid: str,
    depth_resnet_preprocessor_uuid: str,
    class_dims: int = 32,
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
    rgb_resnet_preprocessor_uuid: Optional[str],
    depth_resnet_preprocessor_uuid: Optional[str] = None,
    hidden_size: int = 512,
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
