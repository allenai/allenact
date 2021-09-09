"""
Note: I add this file just for the format consistence with other baselines in the project, so it is just the same as
`allenact_plugins.gym_models.py` so far. However, if it is in the Gym Robotics, some modification is need.
For example, for `state_dim`:
        if input_uuid == 'gym_robotics_data':
            # consider that the observation space is Dict for robotics env
            state_dim = observation_space[self.input_uuid]['observation'].shape[0]
        else:
            assert len(observation_space[self.input_uuid].shape) == 1
            state_dim = observation_space[self.input_uuid].shape[0]
"""
