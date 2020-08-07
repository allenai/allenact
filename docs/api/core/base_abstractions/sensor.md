# core.base_abstractions.sensor [[source]](https://github.com/allenai/embodied-rl/tree/master/core/base_abstractions/sensor.py)

## ResNetSensor
```python
ResNetSensor(
    self,
    mean: Optional[numpy.ndarray] = None,
    stdev: Optional[numpy.ndarray] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    uuid: str = 'resnet',
    output_shape: Optional[Tuple[int, ...]] = None,
    output_channels: Optional[int] = None,
    unnormalized_infimum: float = -inf,
    unnormalized_supremum: float = inf,
    scale_first: bool = True,
    kwargs: Any,
)
```

### to
```python
ResNetSensor.to(self, device: torch.device) -> 'ResNetSensor'
```
Moves sensor to specified device.

__Parameters__


- __device __: The device for the sensor.

## Sensor
```python
Sensor(
    self,
    uuid: str,
    observation_space: gym.spaces.space.Space,
    kwargs: Any,
) -> None
```
Represents a sensor that provides data from the environment to agent.
The user of this class needs to implement the get_observation method and
the user is also required to set the below attributes:

__Attributes__


- `uuid `: universally unique id.
- `observation_space `: ``gym.Space`` object corresponding to observation of
    sensor.

### get_observation
```python
Sensor.get_observation(
    self,
    env: ~EnvType,
    task: Optional[~SubTaskType],
    args: Any,
    kwargs: Any,
) -> Any
```
Returns observations from the environment (or task).

__Parameters__


- __env __: The environment the sensor is used upon.
- __task __: (Optionally) a Task from which the sensor should get data.

__Returns__


Current observation for Sensor.

## SensorSuite
```python
SensorSuite(
    self,
    sensors: Sequence[core.base_abstractions.sensor.Sensor],
) -> None
```
Represents a set of sensors, with each sensor being identified through a
unique id.

__Attributes__


- `sensors`: list containing sensors for the environment, uuid of each
    sensor must be unique.

### get
```python
SensorSuite.get(self, uuid: str) -> core.base_abstractions.sensor.Sensor
```
Return sensor with the given `uuid`.

__Parameters__


- __uuid __: The unique id of the sensor

__Returns__


The sensor with unique id `uuid`.

### get_observations
```python
SensorSuite.get_observations(
    self,
    env: ~EnvType,
    task: Optional[~SubTaskType],
    kwargs: Any,
) -> Dict[str, Any]
```
Get all observations corresponding to the sensors in the suite.

__Parameters__


- __env __: The environment from which to get the observation.
- __task __: (Optionally) the task from which to get the observation.

__Returns__


Data from all sensors packaged inside a Dict.

## SubTaskType
Type variable.

Usage::

  T = TypeVar('T')  # Can be anything
  A = TypeVar('A', str, bytes)  # Must be str or bytes

Type variables exist primarily for the benefit of static type
checkers.  They serve as the parameters for generic types as well
as for generic function definitions.  See class Generic for more
information on generic types.  Generic functions work as follows:

  def repeat(x: T, n: int) -> List[T]:
      '''Return a list containing n references to x.'''
      return [x]*n

  def longest(x: A, y: A) -> A:
      '''Return the longest of two strings.'''
      return x if len(x) >= len(y) else y

The latter example's signature is essentially the overloading
of (str, str) -> str and (bytes, bytes) -> bytes.  Also note
that if the arguments are instances of some subclass of str,
the return type is still plain str.

At runtime, isinstance(x, T) and issubclass(C, T) will raise TypeError.

Type variables defined with covariant=True or contravariant=True
can be used to declare covariant or contravariant generic types.
See PEP 484 for more details. By default generic types are invariant
in all type variables.

Type variables can be introspected. e.g.:

  T.__name__ == 'T'
  T.__constraints__ == ()
  T.__covariant__ == False
  T.__contravariant__ = False
  A.__constraints__ == (str, bytes)

Note that only type variables defined in global scope can be pickled.

## VisionSensor
```python
VisionSensor(
    self,
    mean: Optional[numpy.ndarray] = None,
    stdev: Optional[numpy.ndarray] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    uuid: str = 'vision',
    output_shape: Optional[Tuple[int, ...]] = None,
    output_channels: Optional[int] = None,
    unnormalized_infimum: float = -inf,
    unnormalized_supremum: float = inf,
    scale_first: bool = True,
    kwargs: Any,
)
```

### height
Height that input image will be rescale to have.

__Returns__


The height as a non-negative integer or `None` if no rescaling is done.

### width
Width that input image will be rescale to have.

__Returns__


The width as a non-negative integer or `None` if no rescaling is done.

