class TagIdentifierByteRepresentationError(TypeError):
    pass


class DataDoesNotMatchTagDefinitionError(TypeError):
    pass


class CouldNotAcquireFileLockError(OSError):
    pass