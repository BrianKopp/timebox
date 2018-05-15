from enum import Enum


class TimeBoxOptionPositions(Enum):
    TAG_NAME_BIT_POSITION = 0
    DATE_DIFFERENTIALS_STORED_POSITION = 1


class TimeBoxTagOptionPositions(Enum):
    USE_COMPRESSION = 0
    USE_HASH_TABLE = 1
    FLOATING_POINT_ROUNDED = 2


def get_date_utils_constant_from_stored_units_int(value: int) -> int:
    """
    Gets the constant in datetime_utils.py that maps to the value stored
    :param value: integer value from file
    :return: integer value from datetime_utils
    """
    return value  # for now this is 1-1 mapping


def get_int_for_date_units_from_date_utils_constant(date_utils_constant: int) -> int:
    """
    Converts the datetime utils constant to the value to be stored
    :param date_utils_constant: int like MICRO_SECONDS
    :return: int to be stored
    """
    return date_utils_constant  # for now this is 1-1 mapping
