class CharConversionException(Exception):
    pass


class NotIntegerException(ValueError):
    pass


class IntegerNotUnsignedException(ValueError):
    pass


class IntegerLargerThan64BitsException(ValueError):
    pass


class ArrayNotFloatException(TypeError):
    pass


class DateUnitsError(TypeError):
    pass


class DateUnitsGranularityError(TypeError):
    pass


class CompressionModeInvalidError(ValueError):
    pass


class CompressionError(ValueError):
    pass


class InvalidPandasIndexError(ValueError):
    pass


class InvalidPandasDataTypeError(TypeError):
    pass
