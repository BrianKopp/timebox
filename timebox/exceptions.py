class TagIdentifierByteRepresentationError(TypeError):
    pass


class DataDoesNotMatchTagDefinitionError(TypeError):
    pass


class DataShapeError(ValueError):
    pass


class CouldNotAcquireFileLockError(OSError):
    pass


class DateDataError(ValueError):
    pass


class CompressionModeInvalidError(ValueError):
    pass


class CompressionError(ValueError):
    pass
