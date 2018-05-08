from .exceptions import IntegerNotUnsignedException, IntegerLargerThan64BitsException
from .validation import ensure_int


def determine_required_bytes(value: int) -> int:
    """
    Determines the number of bytes that are required to store value
    :param value: an UNSIGNED integer
    :return: 1, 2, 4, or 8
    """
    value = ensure_int(value)
    if value < 0:
        raise IntegerNotUnsignedException
    if (value >> 8) == 0:
        return 1
    if (value >> 16) == 0:
        return 2
    if (value >> 32) == 0:
        return 4
    if (value >> 64) == 0:
        return 8
    raise IntegerLargerThan64BitsException
