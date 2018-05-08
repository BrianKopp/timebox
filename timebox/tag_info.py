from .utils.numpy_utils import get_numpy_type, get_type_char_int, get_type_char_char
from typing import Union


class TagInfo:
    identifier = None
    bytes_per_value = 0
    type_char = 0
    dtype = None

    def __init__(self, identifier, bytes_per_value: int, type_char: Union[int, str]):
        self.identifier = identifier
        self.bytes_per_value = bytes_per_value
        self.type_char = get_type_char_char(type_char)
        self.dtype = get_numpy_type(
            self.type_char,
            self.bytes_per_value * 8
        )
        return
