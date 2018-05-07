class CharConversionException(Exception):
    pass


class NotIntegerException(ValueError):
    pass


class IntegerNotUnsignedException(ValueError):
    pass


class IntegerLargerThan64BitsException(ValueError):
    pass
