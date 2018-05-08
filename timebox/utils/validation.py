from .exceptions import NotIntegerException


def ensure_int(value):
    if isinstance(value, int):
        return value
    try:
        if value % 1 == 0:
            return int(value)
        raise NotIntegerException(
            'Value {} provided is not an integer value'.format(value)
        )
    except (ValueError, TypeError):
        raise NotIntegerException(
            'Could not cast value {} to integer'.format(value)
        )
