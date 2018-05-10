import numpy as np
from .validation import ensure_int
from .exceptions import *


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


def compress_float_array(arr: np.array) -> np.array:
    """
    Takes a numpy array with dtype.kind = 'f',
    and checks to see which bits are used in the float,
    then returns the compressed array as a float16, float32, or float64
    in a loss-less format
    :param arr: numpy array of floats
    :return: compressed array
    """
    if arr.dtype.kind != 'f':
        raise ArrayNotFloatException
    # filter out NaN values
    not_nan = arr[~np.isnan(arr)]
    if not_nan.size == 0:
        return arr.astype(np.float16)

    # put the array into a byte array
    bytes_dtype = np.dtype([('byte_{}'.format(i), np.uint8) for i in range(0, arr.itemsize)])
    byte_array = np.frombuffer(not_nan.tobytes(), dtype=bytes_dtype)
    byte_arr_len = byte_array['byte_0'].shape[0]
    # mantissa is X bits long, with left-right having value of
    # 0.5 * 2^(i)
    big_endian_nth_byte = dict(
        [('byte_{}'.format(i), 'byte_{}'.format(arr.itemsize - i - 1)) for i in range(0, arr.itemsize)]
    )
    if arr.itemsize == 8:
        # numpy 64-bit float has 52 bits of mantissa
        # in order to compress to 32-bit float, we need to pack mantissa into
        # 23 bits. This means we need 52-23=29 bits of zeros on the end.
        # that means that the 3 right-most bytes must be zero
        # and that the 4th right-most byte must have 5 right-most bits = 0
        # we'll check this by seeing if '0001 1111' & byte == 0. (0001 1111 is 31)
        # finally, we'll need to check the exponent compression

        # first check the 4th right-most byte. it's most likely to have the
        # significant bits we can't drop
        bitwise_reduction = np.bitwise_and(
            byte_array[big_endian_nth_byte['byte_4']],
            np.full(byte_arr_len, 31, dtype=np.uint8)
        )
        if np.count_nonzero(bitwise_reduction) > 0:
            return arr
        # next check the remaining bits
        if np.count_nonzero(byte_array[big_endian_nth_byte['byte_5']]) > 0 \
            or np.count_nonzero(byte_array[big_endian_nth_byte['byte_6']]) > 0 \
                or np.count_nonzero(byte_array[big_endian_nth_byte['byte_7']]) > 0:
            return arr
        # finally check the exponent. in 64-bit float, exponent can be +/- 1024.
        # in 32-bit float, exponent can be +/-128
        # do this the easy way and get this from numpy
        bitwise_exponent = np.frexp(not_nan)[1]
        if np.amax(bitwise_exponent) > 128 or np.amin(bitwise_exponent) < -128:
            return arr

        # else, we're good to go onto 32-bit reduction
        return compress_float_array(arr.astype(np.float32))

    if arr.itemsize == 4:
        # numpy 32-bit float has 23 bits of mantissa.
        # numpy 16-bit float has 10 bits mantissa.
        # we require the right-most 13-bits to be zero. byte_3 (4th byte)
        # must be all zero, and byte_2 (3rd byte) must have
        # the 5 right-most bits be zero. We will bitwise-and against 31
        # (0001 1111) to see if it complies
        bitwise_reduction = np.bitwise_and(
            byte_array[big_endian_nth_byte['byte_2']],
            np.full(byte_arr_len, 31, dtype=np.uint8)
        )
        if np.count_nonzero(bitwise_reduction) > 0:
            return arr
        # next check the 4th byte for zeros
        if np.count_nonzero(byte_array[big_endian_nth_byte['byte_3']]) > 0:
            return arr
        # finally, check exponent. In 32-bit float, exponent can be +/- 128
        # in 16-bit float, exponent can be +/-16
        bitwise_exponent = np.frexp(not_nan)[1]
        if np.amax(bitwise_exponent) > 16 or np.amin(bitwise_exponent) < -16:
            return arr
        # else, return compressed
        return arr.astype(np.float16)

    return arr
