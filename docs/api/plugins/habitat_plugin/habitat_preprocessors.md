# plugins.habitat_plugin.habitat_preprocessors [[source]](https://github.com/allenai/allenact/tree/master/plugins/habitat_plugin/habitat_preprocessors.py)

## ResnetPreProcessorHabitat
```python
ResnetPreProcessorHabitat(
    self,
    input_uuids: List[str],
    output_uuid: str,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    output_dims: int,
    pool: bool,
    torchvision_resnet_model: Callable[..., torchvision.models.resnet.ResNet] = <function resnet18 at 0x12718c1f0>,
    parallel: bool = True,
    device: Optional[torch.device] = None,
    device_ids: Optional[List[torch.device]] = None,
    kwargs: Any,
)
```
Preprocess RGB or depth image using a ResNet model.
