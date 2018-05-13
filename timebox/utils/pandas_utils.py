import numpy as np
from .exceptions import InvalidPandasDataTypeError


def parse_pandas_dtype(dtype):
    """
    Parses the dtype and returns a tuple indicating the type and the size of the data type
    :param dtype: string or dtype object
    :return: tuple like (8, 'f')
    """
    try:
        string_dtype = str(np.dtype(dtype))
    except TypeError:
        raise InvalidPandasDataTypeError('Could not parse dtype {} into a valid data type'.format(dtype))
    if 'float' in string_dtype:
        if 'float64' == string_dtype:
            return 8, 'f'
        elif 'float32' == string_dtype:
            return 4, 'f'
        elif 'float16' == string_dtype:
            return 2, 'f'
    elif 'uint' in string_dtype:
        if 'uint8' == string_dtype:
            return 1, 'u'
        elif 'uint16' == string_dtype:
            return 2, 'u'
        elif 'uint32' == string_dtype:
            return 4, 'u'
        elif 'uint64' == string_dtype:
            return 8, 'u'
    elif 'int' in string_dtype:
        if 'int8' == string_dtype:
            return 1, 'i'
        elif 'int16' == string_dtype:
            return 2, 'i'
        elif 'int32' == string_dtype:
            return 4, 'i'
        elif 'int64' == string_dtype:
            return 8, 'i'
    raise InvalidPandasDataTypeError('Data type {} was not in acceptable list of float or int/unsigned int'
                                     ' or did not have acceptable bits'.format(string_dtype))
