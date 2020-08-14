# plugins.robothor_plugin.robothor_sensors [[source]](https://github.com/allenai/allenact/tree/master/plugins/robothor_plugin/robothor_sensors.py)

## GPSCompassSensorRoboThor
```python
GPSCompassSensorRoboThor(
    self,
    uuid: str = 'target_coordinates_ind',
    kwargs: Any,
)
```

### quaternion_from_coeff
```python
GPSCompassSensorRoboThor.quaternion_from_coeff(
    coeffs: numpy.ndarray,
) -> quaternion.quaternion
```
Creates a quaternions from coeffs in [x, y, z, w] format

### quaternion_from_y_angle
```python
GPSCompassSensorRoboThor.quaternion_from_y_angle(
    angle: float,
) -> quaternion.quaternion
```
Creates a quaternion from rotation angle around y axis

### quaternion_rotate_vector
```python
GPSCompassSensorRoboThor.quaternion_rotate_vector(
    quat: quaternion.quaternion,
    v: <built-in function array>,
) -> <built-in function array>
```
Rotates a vector by a quaternion
Args:
    quat: The quaternion to rotate by
    v: The vector to rotate
Returns:
    np.array: The rotated vector

## RGBSensorRoboThor
```python
RGBSensorRoboThor(
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
Sensor for RGB images in RoboTHOR.

Returns from a running RoboThorEnvironment instance, the current RGB
frame corresponding to the agent's egocentric view.

