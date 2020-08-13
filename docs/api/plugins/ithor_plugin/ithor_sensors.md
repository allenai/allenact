# plugins.ithor_plugin.ithor_sensors [[source]](https://github.com/allenai/allenact/tree/master/plugins/ithor_plugin/ithor_sensors.py)

## RGBSensorThor
```python
RGBSensorThor(
    self,
    use_resnet_normalization: bool = False,
    mean: Optional[numpy.ndarray] = [[[0.485 0.456 0.406]]],
    stdev: Optional[numpy.ndarray] = [[[0.229 0.224 0.225]]],
    height: Optional[int] = None,
    width: Optional[int] = None,
    uuid: str = 'rgb',
    output_shape: Optional[Tuple[int, ...]] = None,
    output_channels: int = 3,
    unnormalized_infimum: float = 0.0,
    unnormalized_supremum: float = 1.0,
    scale_first: bool = True,
    kwargs: Any,
)
```
Sensor for RGB images in AI2-THOR.

Returns from a running IThorEnvironment instance, the current RGB
frame corresponding to the agent's egocentric view.

