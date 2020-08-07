# core.base_abstractions.preprocessor [[source]](https://github.com/allenai/embodied-rl/tree/master/core/base_abstractions/preprocessor.py)

## ObservationSet
```python
ObservationSet(
    self,
    source_ids: List[str],
    all_preprocessors: List[Union[core.base_abstractions.preprocessor.Preprocessor, utils.experiment_utils.Builder[core.base_abstractions.preprocessor.Preprocessor]]],
    all_sensors: List[core.base_abstractions.sensor.Sensor],
) -> None
```
Represents a list of source_ids, corresponding to sensors and
preprocessors, with each source being identified through a unique id.

__Attributes__


- `source_ids `: List containing sensor and preprocessor ids to be consumed by agents. Each source uuid must be unique.
- `graph `: Computation graph for all preprocessors.
- `observation_spaces `: Observation spaces of all output sources.
- `device `: Device where the PreprocessorGraph is executed.

### get
```python
ObservationSet.get(
    self,
    uuid: str,
) -> core.base_abstractions.preprocessor.Preprocessor
```
Return preprocessor with the given `uuid`.

__Parameters__


- __uuid __: The unique id of the preprocessor.

__Returns__


The preprocessor with unique id `uuid`.

### get_observations
```python
ObservationSet.get_observations(
    self,
    obs: Dict[str, Any],
    args: Any,
    kwargs: Any,
) -> Dict[str, Any]
```
Get all observations within a dictionary.

__Returns__


Collect observations from all sources and return them packaged inside a Dict.

## Preprocessor
```python
Preprocessor(
    self,
    input_uuids: List[str],
    output_uuid: str,
    observation_space: gym.spaces.space.Space,
    kwargs: Any,
) -> None
```
Represents a preprocessor that transforms data from a sensor or another
preprocessor to the input of agents or other preprocessors. The user of
this class needs to implement the process method and the user is also
required to set the below attributes:

__Attributes:__

    input_uuids : List of input universally unique ids.
    uuid : Universally unique id.
    observation_space : ``gym.Space`` object corresponding to processed observation spaces.

### process
```python
Preprocessor.process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any
```
Returns processed observations from sensors or other preprocessors.

__Parameters__


- __obs __: Dict with available observations and processed observations.

__Returns__


Processed observation.

## PreprocessorGraph
```python
PreprocessorGraph(
    self,
    preprocessors: List[Union[core.base_abstractions.preprocessor.Preprocessor, utils.experiment_utils.Builder[core.base_abstractions.preprocessor.Preprocessor]]],
) -> None
```
Represents a graph of preprocessors, with each preprocessor being
identified through a universally unique id.

__Attributes__


- `preprocessors `: List containing preprocessors with required input uuids, output uuid of each
    sensor must be unique.

### get
```python
PreprocessorGraph.get(
    self,
    uuid: str,
) -> core.base_abstractions.preprocessor.Preprocessor
```
Return preprocessor with the given `uuid`.

__Parameters__


- __uuid __: The unique id of the preprocessor.

__Returns__


The preprocessor with unique id `uuid`.

### get_observations
```python
PreprocessorGraph.get_observations(
    self,
    obs: Dict[str, Any],
    args: Any,
    kwargs: Any,
) -> Dict[str, Any]
```
Get processed observations.

__Returns__


Collect observations processed from all sensors and return them packaged inside a Dict.

