# plugins.robothor_plugin.robothor_environment [[source]](https://github.com/allenai/embodied-rl/tree/master/plugins/robothor_plugin/robothor_environment.py)

## RoboThorCachedEnvironment
```python
RoboThorCachedEnvironment(self, **kwargs)
```
Wrapper for the robo2thor controller providing additional functionality
and bookkeeping.

See [here](https://ai2thor.allenai.org/robothor/documentation) for comprehensive
 documentation on RoboTHOR.

__Attributes__


- `controller `: The AI2THOR controller.
- `config `: The AI2THOR controller configuration

### agent_state
```python
RoboThorCachedEnvironment.agent_state(
    self,
) -> Dict[str, Union[Dict[str, float], float]]
```
Return agent position, rotation and horizon.
### all_objects
```python
RoboThorCachedEnvironment.all_objects(self) -> List[Dict[str, Any]]
```
Return all object metadata.
### all_objects_with_properties
```python
RoboThorCachedEnvironment.all_objects_with_properties(
    self,
    properties: Dict[str, Any],
) -> List[Dict[str, Any]]
```
Find all objects with the given properties.
### current_depth
Returns depth image corresponding to the agent's egocentric view.
### current_frame
Returns rgb image corresponding to the agent's egocentric view.
### currently_reachable_points
List of {"x": x, "y": y, "z": z} locations in the scene that are
currently reachable.
### last_action
Last action, as a string, taken by the agent.
### last_action_return
Get the value returned by the last action (if applicable).

For an example of an action that returns a value, see
`"GetReachablePositions"`.

### last_action_success
In the cached environment, all actions succeed.
### last_event
Last event returned by the controller.
### reset
```python
RoboThorCachedEnvironment.reset(self, scene_name: str = None) -> None
```
Resets scene to a known initial state.
### scene_name
Current ai2thor scene.
### step
```python
RoboThorCachedEnvironment.step(
    self,
    action_dict: Dict[str, Union[str, int, float]],
) -> ai2thor.server.Event
```
Take a step in the ai2thor environment.
### stop
```python
RoboThorCachedEnvironment.stop(self)
```
Stops the ai2thor controller.
### visible_objects
```python
RoboThorCachedEnvironment.visible_objects(self) -> List[Dict[str, Any]]
```
Return all visible objects.
## RoboThorEnvironment
```python
RoboThorEnvironment(self, **kwargs)
```
Wrapper for the robo2thor controller providing additional functionality
and bookkeeping.

See [here](https://ai2thor.allenai.org/robothor/documentation) for comprehensive
 documentation on RoboTHOR.

__Attributes__


- `controller `: The AI2THOR controller.
- `config `: The AI2THOR controller configuration

### access_grid
```python
RoboThorEnvironment.access_grid(self, target: str) -> float
```
Returns the geodesic distance from the quantized location of the
agent in the current scene's grid to the target object of given
type.
### agent_state
```python
RoboThorEnvironment.agent_state(
    self,
) -> Dict[str, Union[Dict[str, float], float]]
```
Return agent position, rotation and horizon.
### all_objects
```python
RoboThorEnvironment.all_objects(self) -> List[Dict[str, Any]]
```
Return all object metadata.
### all_objects_with_properties
```python
RoboThorEnvironment.all_objects_with_properties(
    self,
    properties: Dict[str, Any],
) -> List[Dict[str, Any]]
```
Find all objects with the given properties.
### current_depth
Returns depth image corresponding to the agent's egocentric view.
### current_frame
Returns rgb image corresponding to the agent's egocentric view.
### currently_reachable_points
List of {"x": x, "y": y, "z": z} locations in the scene that are
currently reachable.
### dist_to_object
```python
RoboThorEnvironment.dist_to_object(self, object_type: str) -> float
```
Minimal geodesic distance to object of given type from agent's
current location.

It might return -1.0 for unreachable targets.

### dist_to_point
```python
RoboThorEnvironment.dist_to_point(self, xyz: Dict[str, float]) -> float
```
Minimal geodesic distance to end point from agent's current
location.

It might return -1.0 for unreachable targets.

### initialize_grid
```python
RoboThorEnvironment.initialize_grid(self) -> None
```
Initializes grid for current scene if not already initialized.
### initialize_grid_dimensions
```python
RoboThorEnvironment.initialize_grid_dimensions(
    self,
    reachable_points: Collection[Dict[str, float]],
) -> Tuple[int, int, int, int]
```
Computes bounding box for reachable points quantized with the
current gridSize.
### last_action
Last action, as a string, taken by the agent.
### last_action_return
Get the value returned by the last action (if applicable).

For an example of an action that returns a value, see
`"GetReachablePositions"`.

### last_action_success
Was the last action taken by the agent a success?
### last_event
Last event returned by the controller.
### object_reachable
```python
RoboThorEnvironment.object_reachable(self, object_type: str) -> bool
```
Determines whether a path can be computed from the discretized
current agent location to the target object of given type.
### path_corners
```python
RoboThorEnvironment.path_corners(
    self,
    target: Union[str, Dict[str, float]],
) -> Collection[Dict[str, float]]
```
Returns an array with a sequence of xyz dictionaries objects
representing the corners of the shortest path to the object of given
type or end point location.
### path_corners_to_dist
```python
RoboThorEnvironment.path_corners_to_dist(
    self,
    corners: Sequence[Dict[str, float]],
) -> float
```
Computes the distance covered by the given path described by its
corners.
### point_reachable
```python
RoboThorEnvironment.point_reachable(self, xyz: Dict[str, float]) -> bool
```
Determines whether a path can be computed from the current agent
location to the target point.
### quantized_agent_state
```python
RoboThorEnvironment.quantized_agent_state(
    self,
    xz_subsampling: int = 1,
    rot_subsampling: int = 1,
) -> Tuple[int, int, int]
```
Quantizes agent location (x, z) to a (subsampled) position in a
fixed size grid derived from the initial set of reachable points; and
rotation (around y axis) as a (subsampled) discretized angle given the
current `rotateStepDegrees`.
### random_reachable_state
```python
RoboThorEnvironment.random_reachable_state(
    self,
    seed: Optional[int] = None,
) -> Dict[str, Union[Dict[str, float], float]]
```
Returns a random reachable location in the scene.
### randomize_agent_location
```python
RoboThorEnvironment.randomize_agent_location(
    self,
    seed: int = None,
    partial_position: Optional[Dict[str, float]] = None,
) -> Dict[str, Union[Dict[str, float], float]]
```
Teleports the agent to a random reachable location in the scene.
### reset
```python
RoboThorEnvironment.reset(self, scene_name: str = None) -> None
```
Resets scene to a known initial state.
### scene_name
Current ai2thor scene.
### step
```python
RoboThorEnvironment.step(
    self,
    action_dict: Dict[str, Union[str, int, float]],
) -> ai2thor.server.Event
```
Take a step in the ai2thor environment.
### stop
```python
RoboThorEnvironment.stop(self)
```
Stops the ai2thor controller.
### visible_objects
```python
RoboThorEnvironment.visible_objects(self) -> List[Dict[str, Any]]
```
Return all visible objects.
