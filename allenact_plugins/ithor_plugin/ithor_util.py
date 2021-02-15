import math


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
