import glob
import math
import os
import platform
import traceback
import warnings
from contextlib import contextmanager
from typing import Sequence

import Xlib
import Xlib.display
import ai2thor.controller


@contextmanager
def include_object_data(controller: ai2thor.controller.Controller):
    needs_reset = len(controller.last_event.metadata["objects"]) == 0
    try:
        if needs_reset:
            controller.step("ResetObjectFilter")
            assert controller.last_event.metadata["lastActionSuccess"]
        yield None
    finally:
        if needs_reset:
            controller.step("SetObjectFilter", objectIds=[])
            assert controller.last_event.metadata["lastActionSuccess"]


def vertical_to_horizontal_fov(
    vertical_fov_in_degrees: float, height: float, width: float
):
    assert 0 < vertical_fov_in_degrees < 180
    aspect_ratio = width / height
    vertical_fov_in_rads = (math.pi / 180) * vertical_fov_in_degrees
    return (
        (180 / math.pi)
        * math.atan(math.tan(vertical_fov_in_rads * 0.5) * aspect_ratio)
        * 2
    )


def horizontal_to_vertical_fov(
    horizontal_fov_in_degrees: float, height: float, width: float
):
    return vertical_to_horizontal_fov(
        vertical_fov_in_degrees=horizontal_fov_in_degrees, height=width, width=height,
    )


def round_to_factor(num: float, base: int) -> int:
    """Rounds floating point number to the nearest integer multiple of the
    given base. E.g., for floating number 90.1 and integer base 45, the result
    is 90.

    # Attributes

    num : floating point number to be rounded.
    base: integer base
    """
    return round(num / base) * base


def get_open_x_displays(throw_error_if_empty: bool = False) -> Sequence[str]:
    assert platform.system() == "Linux", "Can only get X-displays for Linux systems."

    displays = []

    open_display_strs = [
        os.path.basename(s)[1:] for s in glob.glob("/tmp/.X11-unix/X*")
    ]

    for open_display_str in sorted(open_display_strs):
        try:
            open_display_str = str(int(open_display_str))
            display = Xlib.display.Display(f":{open_display_str}")
        except Exception:
            warnings.warn(
                f"Encountered error when attempting to open display :{open_display_str},"
                f" error message:\n{traceback.format_exc()}"
            )
            continue

        displays.extend(
            [f"{open_display_str}.{i}" for i in range(display.screen_count())]
        )

    if throw_error_if_empty and len(displays) == 0:
        raise IOError(
            "Could not find any open X-displays on which to run AI2-THOR processes. "
            " Please see the AI2-THOR installation instructions at"
            " https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
            " for information as to how to start such displays."
        )

    return displays
