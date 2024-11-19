from typing import Dict, Any, Optional
from collections import defaultdict

import numpy as np
from ai2thor.controller import Controller


def make_angle_thresholds(quant_angles: int = 4):
    angle_spacing = 360 / quant_angles
    angle_shift = angle_spacing / 2
    angle_thres = [angle_shift + angle_spacing * i for i in range(quant_angles)]
    return np.array(
        [angle_thres[-1] - 360.0] + angle_thres
    )  # monotonic bin thresholds for numpy digitize


def quantize_loc(
    current_agent_state: Dict[str, Any],
    initial_agent_state: Dict[str, Any],
    quant_grid_size: float = 0.5,
    quant_grid_axes: str = "xz",
    quant_angles: int = 4,
    angle_thresholds: Optional[np.ndarray] = None,
):
    if quant_angles > 1:
        if angle_thresholds is None:
            angle_thresholds = make_angle_thresholds(quant_angles)

        angle_dif = (
            current_agent_state["rotation"]["y"] - initial_agent_state["rotation"]["y"]
        ) % 360

        quant_angle = (np.digitize(angle_dif, angle_thresholds) - 1) % quant_angles
    else:
        quant_angle = 0

    current_location = (
        current_agent_state["position"]
        if "position" in current_agent_state
        else current_agent_state
    )
    initial_location = (
        initial_agent_state["position"]
        if "position" in initial_agent_state
        else initial_agent_state
    )

    return tuple(
        int(round((current_location[x] - initial_location[x]) / quant_grid_size))
        for x in quant_grid_axes
    ) + (quant_angle,)


def visited_cells_metric(
    trajectory,
    return_state_visits=False,
    quant_grid_size: float = 0.5,
    quant_grid_axes: str = "xz",
    quant_angles: int = 4,
    angle_thresholds: Optional[np.ndarray] = None,
):
    visited_cells = defaultdict(list)

    if quant_angles > 1 and angle_thresholds is None:
        angle_thresholds = make_angle_thresholds(quant_angles)

    for t, state in enumerate(trajectory):
        visited_cells[
            quantize_loc(
                current_agent_state=state,
                initial_agent_state=trajectory[0],
                quant_grid_size=quant_grid_size,
                quant_grid_axes=quant_grid_axes,
                quant_angles=quant_angles,
                angle_thresholds=angle_thresholds,
            )
        ].append(t)

    if return_state_visits:
        return len(visited_cells), visited_cells
    else:
        return len(visited_cells)


def num_reachable_positions_cells(
    controller: Controller,
    quant_grid_size: float = 0.5,
    quant_grid_axes: str = "xz",
    quant_angles: int = 4,
):
    """
    Assumes the agent is at the episode's initial state.
    Note that there's a chance more states are visitable than here counted.
    For quant_angles, we just assume we can always get to each reachable position
    in each of the quant_angles relative angles to the initial orientation.
    """

    initial_agent_state = controller.last_event.metadata["agent"]

    controller.step("GetReachablePositions")
    assert controller.last_event.metadata["lastActionSuccess"]
    reachable_positions = controller.last_event.metadata["actionReturn"]

    return (
        visited_cells_metric(
            [initial_agent_state] + reachable_positions,
            return_state_visits=False,
            quant_grid_size=quant_grid_size,
            quant_grid_axes=quant_grid_axes,
            quant_angles=1,
        )
        * quant_angles
    )
