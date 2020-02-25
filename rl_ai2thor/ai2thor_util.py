def round_to_factor(num: float, base: int) -> int:
    """Rounds floating point number to the nearest integer multiple of the given base. E.g., for floating number
     90.1 and integer base 45, the result is 90.

    # Attributes

    num : floating point number to be rounded.
    base: integer base
    """
    return round(num / base) * base
