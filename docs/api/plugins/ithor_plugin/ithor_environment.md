# plugins.ithor_plugin.ithor_environment [[source]](https://github.com/allenai/allenact/tree/master/plugins/ithor_plugin/ithor_environment.py)
A wrapper for engaging with the THOR environment.
## IThorEnvironment
```python
IThorEnvironment(
    self,
    x_display: Optional[str] = None,
    docker_enabled: bool = False,
    local_thor_build: Optional[str] = None,
    visibility_distance: float = 1.25,
    fov: float = 90.0,
    player_screen_width: int = 300,
    player_screen_height: int = 300,
    quality: str = 'Very Low',
    restrict_to_initially_reachable_points: bool = False,
    make_agents_visible: bool = True,
    object_open_speed: float = 1.0,
    simplify_physics: bool = False,
) -> None
```
Wrapper for the ai2thor controller providing additional functionality
and bookkeeping.

See [here](https://ai2thor.allenai.org/documentation/installation) for comprehensive
 documentation on AI2-THOR.

__Attributes__


- `controller `: The ai2thor controller.

### all_objects
```python
IThorEnvironment.all_objects(self) -> List[Dict[str, Any]]
```
Return all object metadata.
### all_objects_with_properties
```python
IThorEnvironment.all_objects_with_properties(
    self,
    properties: Dict[str, Any],
) -> List[Dict[str, Any]]
```
Find all objects with the given properties.
### closest_object_of_type
```python
IThorEnvironment.closest_object_of_type(
    self,
    object_type: str,
) -> Optional[Dict[str, Any]]
```
Find the object closest to the agent that has the given type.
### closest_object_with_properties
```python
IThorEnvironment.closest_object_with_properties(
    self,
    properties: Dict[str, Any],
) -> Optional[Dict[str, Any]]
```
Find the object closest to the agent that has the given
properties.
### closest_reachable_point_to_position
```python
IThorEnvironment.closest_reachable_point_to_position(
    self,
    position: Dict[str, float],
) -> Tuple[Dict[str, float], float]
```
Of all reachable positions, find the one that is closest to the
given location.
### closest_visible_object_of_type
```python
IThorEnvironment.closest_visible_object_of_type(
    self,
    object_type: str,
) -> Optional[Dict[str, Any]]
```
Find the object closest to the agent that is visible and has the
given type.
### current_frame
Returns rgb image corresponding to the agent's egocentric view.
### currently_reachable_points
List of {"x": x, "y": y, "z": z} locations in the scene that are
currently reachable.
### get_agent_location
```python
IThorEnvironment.get_agent_location(self) -> Dict[str, Union[float, bool]]
```
Gets agent's location.
### GRAPH_ACTIONS_SET
set() -> new empty set object
set(iterable) -> new set object

Build an unordered collection of unique elements.
### initially_reachable_points
List of {"x": x, "y": y, "z": z} locations in the scene that were
reachable after initially resetting.
### initially_reachable_points_set
Set of (x,z) locations in the scene that were reachable after
initially resetting.
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
### object_in_hand
```python
IThorEnvironment.object_in_hand(self)
```
Object metadata for the object in the agent's hand.
### object_pixels_in_frame
```python
IThorEnvironment.object_pixels_in_frame(
    self,
    object_id: str,
    hide_all: bool = True,
    hide_transparent: bool = False,
) -> numpy.ndarray
```
Return an mask for a given object in the agent's current view.

__Parameters__


- __object_id __: The id of the object.
- __hide_all __: Whether or not to hide all other objects in the scene before getting the mask.
- __hide_transparent __: Whether or not partially transparent objects are considered to occlude the object.

__Returns__


A numpy array of the mask.

### object_pixels_on_grid
```python
IThorEnvironment.object_pixels_on_grid(
    self,
    object_id: str,
    grid_shape: Tuple[int, int],
    hide_all: bool = True,
    hide_transparent: bool = False,
) -> numpy.ndarray
```
Like `object_pixels_in_frame` but counts object pixels in a
partitioning of the image.
### position_dist
```python
IThorEnvironment.position_dist(
    p0: Mapping[str, Any],
    p1: Mapping[str, Any],
    ignore_y: bool = False,
) -> float
```
Distance between two points of the form {"x": x, "y":y, "z":z"}.
### random_reachable_state
```python
IThorEnvironment.random_reachable_state(self, seed: int = None) -> Dict
```
Returns a random reachable location in the scene.
### randomize_agent_location
```python
IThorEnvironment.randomize_agent_location(
    self,
    seed: int = None,
    partial_position: Optional[Dict[str, float]] = None,
) -> Dict
```
Teleports the agent to a random reachable location in the scene.
### reset
```python
IThorEnvironment.reset(
    self,
    scene_name: Optional[str],
    move_mag: float = 0.25,
    kwargs,
)
```
Resets the ai2thor in a new scene.

Resets ai2thor into a new scene and initializes the scene/agents with
prespecified settings (e.g. move magnitude).

__Parameters__


- __scene_name __: The scene to load.
- __move_mag __: The amount of distance the agent moves in a single `MoveAhead` step.
- __kwargs __: additional kwargs, passed to the controller "Initialize" action.

### rotation_dist
```python
IThorEnvironment.rotation_dist(a: Dict[str, float], b: Dict[str, float])
```
Distance between rotations.
### scene_name
Current ai2thor scene.
### start
```python
IThorEnvironment.start(
    self,
    scene_name: Optional[str],
    move_mag: float = 0.25,
    kwargs,
) -> None
```
Starts the ai2thor controller if it was previously stopped.

After starting, `reset` will be called with the scene name and move magnitude.

__Parameters__


- __scene_name __: The scene to load.
- __move_mag __: The amount of distance the agent moves in a single `MoveAhead` step.
- __kwargs __: additional kwargs, passed to reset.

### started
Has the ai2thor controller been started.
### step
```python
IThorEnvironment.step(
    self,
    action_dict: Dict[str, Union[str, int, float]],
) -> ai2thor.server.Event
```
Take a step in the ai2thor environment.
### stop
```python
IThorEnvironment.stop(self) -> None
```
Stops the ai2thor controller.
### teleport_agent_to
```python
IThorEnvironment.teleport_agent_to(
    self,
    x: float,
    y: float,
    z: float,
    rotation: float,
    horizon: float,
    standing: Optional[bool] = None,
    force_action: bool = False,
    only_initially_reachable: Optional[bool] = None,
    verbose = True,
    ignore_y_diffs = False,
) -> None
```
Helper function teleporting the agent to a given location.
### visible_objects
```python
IThorEnvironment.visible_objects(self) -> List[Dict[str, Any]]
```
Return all visible objects.
