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


class CouldNotCalculateNumBytesError(ValueError):
    pass
