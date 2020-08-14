# core.models.basic_models [[source]](https://github.com/allenai/allenact/tree/master/core/models/basic_models.py)
Basic building block torch networks that can be used across a variety of
tasks.
## Flatten
```python
Flatten(self)
```
Flatten input tensor so that it is of shape (batchs x -1).
### forward
```python
Flatten.forward(self, x)
```
Flatten input tensor.

__Parameters__

- __x __: Tensor of size (batches x ...) to flatten to size (batches x -1)
__Returns__

Flattened tensor.

## RNNStateEncoder
```python
RNNStateEncoder(
    self,
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    rnn_type: str = 'GRU',
    trainable_masked_hidden_state: bool = False,
)
```
A simple RNN-based model playing a role in many baseline embodied-
navigation agents.

See `seq_forward` for more details of how this model is used.

### forward
```python
RNNStateEncoder.forward(
    self,
    x: torch.FloatTensor,
    hidden_states: torch.FloatTensor,
    masks: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]]
```
Calls `seq_forward` or `single_forward` depending on the input size.

See the above methods for more information.

### layer_init
```python
RNNStateEncoder.layer_init(self)
```
Initialize the RNN parameters in the model.
### num_recurrent_layers
The number of recurrent layers in the network.
### seq_forward
```python
RNNStateEncoder.seq_forward(
    self,
    x: torch.FloatTensor,
    hidden_states: torch.FloatTensor,
    masks: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]]
```
Forward for a sequence of length T.

__Parameters__


- __x __: (T, N, -1) Tensor that has been flattened to (T * N, -1).
- __hidden_states __: The starting hidden states.
- __masks __: A (T, N) tensor flattened to (T * N).
    The masks to be applied to hidden state at every timestep, equal to 0 whenever the previous step finalized
    the task, 1 elsewhere.

### single_forward
```python
RNNStateEncoder.single_forward(
    self,
    x: torch.FloatTensor,
    hidden_states: torch.FloatTensor,
    masks: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, Union[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]]
```
Forward for a non-sequence input.
## SimpleCNN
```python
SimpleCNN(self, observation_space: gym.spaces.dict.Dict, output_size: int)
```
A Simple 3-Conv CNN followed by a fully connected layer. Takes in
observations (of type gym.spaces.dict) and produces an embedding of the
`"rgb"` and/or `"depth"` components.

__Attributes__


- `observation_space `: The observation_space of the agent, should have 'rgb' or 'depth' as
    a component (otherwise it is a blind model).
- `output_size `: The size of the embedding vector to produce.

### is_blind
True if the observation space doesn't include `"rgb"` or
`"depth"`.
### layer_init
```python
SimpleCNN.layer_init(cnn) -> None
```
Initialize layer parameters using kaiming normal.
