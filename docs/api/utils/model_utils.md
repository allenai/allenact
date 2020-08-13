# utils.model_utils [[source]](https://github.com/allenai/allenact/tree/master/utils/model_utils.py)
Functions used to initialize and manipulate pytorch models.
## init_linear_layer
```python
init_linear_layer(
    module: torch.nn.modules.linear.Linear,
    weight_init: collections.abc.Callable,
    bias_init: collections.abc.Callable,
    gain = 1,
)
```
Initialize a torch.nn.Linear layer.

__Parameters__


- __module __: A torch linear layer.
- __weight_init __: Function used to initialize the weight parameters of the linear layer. Should take the weight data
    tensor and gain as input.
- __bias_init __: Function used to initialize the bias parameters of the linear layer. Should take the bias data
    tensor and gain as input.
- __gain __: The gain to apply.

__Returns__


The initialized linear layer.

