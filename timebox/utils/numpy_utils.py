import numpy as np
from .validation import ensure_int
from .exceptions import CharConversionException


def get_type_char_int(type_char_or_int):
    """
    Gets the integer ord() of the character
    :param type_char_or_int: string or integer to convert into an integer.
    :return: integer corresponding to the char (or int)
    """
    if isinstance(type_char_or_int, str):
        return ord(type_char_or_int)
    if isinstance(type_char_or_int, int):
        return type_char_or_int
    try:
        return ensure_int(type_char_or_int)
    except:
        raise CharConversionException(
            'Could not convert type: {} for value: {}'.format(
                type(type_char_or_int),
                type_char_or_int
            )
        )


def get_type_char_char(type_char_or_int):
    """
    Gets the character representation of the char or int.
    :param type_char_or_int: string or integer to convert into a char
    :return: char corresponding to parameter
    """
    if isinstance(type_char_or_int, int):
        return chr(type_char_or_int)
    if isinstance(type_char_or_int, float):
        return chr(ensure_int(type_char_or_int))
    if isinstance(type_char_or_int, str):
        if len(type_char_or_int) == 1:
            return type_char_or_int
        else:
            raise CharConversionException('Could not convert string \'{}\'into '
                                          'char because its length was > 1'.format(type_char_or_int))
    # last ditch, try to cast it as int
    try:
        return chr(int(type_char_or_int))
    except TypeError:
        raise CharConversionException(
            'Could not convert type: {} for value: {}'.format(
                type(type_char_or_int),
                type_char_or_int
            )
        )


def get_numpy_type(type_char, size: int) -> np.dtype:
    """
    Gets the numpy data type corresponding to the type char ('i', 'u', 'f')
    and size in bits of the value.
    :param type_char: string or integer corresponding to char in ('i', 'u', 'f')
    :param size:
    :return:
    """
    type_char = get_type_char_char(type_char)
    size = ensure_int(size)
    if type_char == 'i':
        if size == 8:
            return np.int8
        elif size == 16:
            return np.int16
        elif size == 32:
            return np.int32
        elif size == 64:
            return np.int64
    elif type_char == 'u':
        if size == 8:
            return np.uint8
        elif size == 16:
            return np.uint16
        elif size == 32:
            return np.uint32
        elif size == 64:
            return np.uint64
    elif type_char == 'f':
        if size == 16:
            return np.float16
        elif size == 32:
            return np.float32
        elif size == 64:
            return np.float64
    raise ValueError(
        'Could not find match for char {} and size {}'.format(
            type_char,
            size
        )
    )
