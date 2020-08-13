# projects.pointnav_baselines.experiments.robothor.pointnav_robothor_base [[source]](https://github.com/allenai/allenact/tree/master/projects/pointnav_baselines/experiments/robothor/pointnav_robothor_base.py)

## PointNavRoboThorBaseConfig
```python
PointNavRoboThorBaseConfig(self)
```
The base config for all iTHOR PointNav experiments.
### ADVANCE_SCENE_ROLLOUT_PERIOD
int([x]) -> integer
int(x, base=10) -> integer

Convert a number or string to an integer, or return 0 if no arguments
are given.  If x is a number, return x.__int__().  For floating point
numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string,
bytes, or bytearray instance representing an integer literal in the
given base.  The literal can be preceded by '+' or '-' and be surrounded
by whitespace.  The base defaults to 10.  Valid bases are 0 and 2-36.
Base 0 means to interpret the base from the string as an integer literal.
>>> int('0b100', base=0)
4
