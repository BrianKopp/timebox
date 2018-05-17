from timebox.utils.exceptions import IntegerNotUnsignedException, IntegerLargerThan64BitsException
from timebox.utils.validation import ensure_int


def determine_required_bytes_unsigned_integer(value: int) -> int:
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


def determine_required_bytes_signed_integer(value: int) -> int:
    """
    Determines the number of bytes that are required to store value
    :param value: a SIGNED integer
    :return: 1, 2, 4, or 8
    """
    value = ensure_int(value)
    if value < 0:
        value *= -1
        value -= 1
    if (value >> 7) == 0:
        return 1
    if (value >> 15) == 0:
        return 2
    if (value >> 31) == 0:
        return 4
    if (value >> 63) == 0:
        return 8
    raise IntegerLargerThan64BitsException


def read_unsigned_int(from_bytes: bytes) -> int:
    """
    from_bytes is the binary file contents to read from
    :param from_bytes: string representation of binary bytes read from a file
    :return: integer
    """
    return int.from_bytes(from_bytes, byteorder='little', signed=False)
