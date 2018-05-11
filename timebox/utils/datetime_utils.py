import numpy as np
from collections import namedtuple
from .exceptions import *


NANO_SECONDS = 0
MICRO_SECONDS = 1
MILLI_SECONDS = 2
SECONDS = 3
MINUTES = 4
HOURS = 5
DAYS = 6


UnitInfo = namedtuple('UnitInfo', ['units', 'order', 'multiplier'])


units = {
    'ns': UnitInfo(units='ns', order=NANO_SECONDS, multiplier=1),
    'us': UnitInfo(units='us', order=MICRO_SECONDS, multiplier=1000),
    'ms': UnitInfo(units='ms', order=MILLI_SECONDS, multiplier=1000000),
    's': UnitInfo(units='s', order=SECONDS, multiplier=1000000000),
    'm': UnitInfo(units='m', order=MINUTES, multiplier=60000000000),
    'h': UnitInfo(units='h', order=HOURS, multiplier=3600000000000),
    'D': UnitInfo(units='D', order=DAYS, multiplier=86400000000000)
}


units_by_order = {
    NANO_SECONDS: 'ns',
    MICRO_SECONDS: 'us',
    MILLI_SECONDS: 'ms',
    SECONDS: 's',
    MINUTES: 'm',
    HOURS: 'h',
    DAYS: 'D'
}


def get_unit_data(units_to_get) -> UnitInfo:
    """
    Gets units from either a string of units, or a integer representing which unit
    :param units_to_get: string like 's' or 'ms' or int
    :return: True or raise error
    """
    if isinstance(units_to_get, int):
        try:
            return get_unit_data(units_by_order[units_to_get])
        except (KeyError, TypeError):
            raise DateUnitsError('Could not get units represented by int code'
                                 ' {}, units not found'.format(units_to_get))
    try:
        return units[units_to_get]
    except (KeyError, TypeError):
        raise DateUnitsError('Could not get units {}, units not found'.format(units_to_get))


def get_more_granular_units(from_units: str) -> str:
    """
    Looks up the more granular units and returns that string. E.g. from units 's' returns
    'ms' for milliseconds
    :param from_units: string units in ['us', 'ms', 's', 'm', 'h', 'D']
    :return: string in ['ns', 'us', 'ms', 's', 'm', 'h']
    """
    from_units_data = get_unit_data(from_units)
    try:
        return units_by_order[from_units_data.order - 1]
    except KeyError:
        raise DateUnitsGranularityError('Could not get more granular units from {}, it is the most granular unit'.format(from_units))


def get_less_granular_units(from_units: str) -> str:
    """
    Looks up the less granular units and returns that string. E.g. from units 's' returns
    'm' for minutes
    :param from_units: string units in ['ns', 'us', 'ms', 's', 'm', 'h']
    :return: string in ['us', 'ms', 's', 'm', 'h', 'D']
    """
    from_units_data = get_unit_data(from_units)
    try:
        return units_by_order[from_units_data.order + 1]
    except KeyError:
        raise DateUnitsGranularityError('Could not get less granular units from '
                                        '{}, it is the least granular unit'.format(from_units))


def get_conversion_multiplier(from_units: str, to_units: str) -> float:
    """
    Gets the conversion multiplier from 'from_units' to 'to_units', such that
    caller can multiply a number having 'from_units' by return result to get 'to_units'.
    For example get_conversion_multiplier('m', 's') = 60. Multiply 1 [m] by 60 = 60 [s]
    :param from_units: string in ['ns', 'us', 'ms', 's', 'm', 'h', 'D']
    :param to_units: string in ['ns', 'us', 'ms', 's', 'm', 'h', 'D']
    :return: float, multiplier
    """
    from_units_data = get_unit_data(from_units)
    to_units_data = get_unit_data(to_units)
    return from_units_data.multiplier / to_units_data.multiplier


def get_units_from_dtype(dtype) -> str:
    """
    Reads the numpy dtype and returns the string of the units
    :param dtype: dtype of datetime64 or timedelta64
    :return: units string of value
    """
    data_type = str(dtype)
    if 'timedelta64' not in data_type and 'datetime64' not in data_type:
        raise DateUnitsError('Could not parse data type {} to extract date/time units'.format(data_type))
    if '[' not in data_type or ']' not in data_type:
        raise DateUnitsError('Could not parse data type {} to extract date/time units'.format(data_type))
    date_units = data_type[data_type.find('[') + 1:data_type.find(']')]
    if date_units not in units:
        raise DateUnitsError('Date units parsed from dtype could not be found in units list'.format(date_units))
    return date_units


def compress_time_delta_array(arr: np.array) -> (np.array, str):
    """
    Tries to compress the timedelta64 array by units
    :param arr: numpy array
    :return: tuple, (numpy array of int64s, units string)
    """
    mod_value = np.full(arr.shape, 0, dtype=np.uint64)
    mod_result = np.full(arr.shape, 0, dtype=np.float64)
    result_array = arr.astype(np.int64)
    curr_units = get_units_from_dtype(arr.dtype)
    while True:
        try:
            try_units = get_less_granular_units(curr_units)
        except DateUnitsGranularityError:  # we couldn't get less granular
            break
        divisor = get_conversion_multiplier(try_units, curr_units)
        mod_value.fill(divisor)
        np.mod(result_array, mod_value, mod_result)
        # check if we are not successful
        if np.count_nonzero(mod_result) > 0:
            break
        # else, we are successful, update the result_array and curr_units
        curr_units = try_units
        result_array = (result_array / divisor).astype(np.int64)
    return result_array, curr_units
