import numpy as np
from collections import namedtuple
from typing import Union
from timebox.utils.numpy_utils import get_numpy_type, get_type_char_char, get_type_char_int
from timebox.exceptions import TagIdentifierByteRepresentationError
from timebox.utils.exceptions import NotIntegerException
from timebox.utils.validation import ensure_int


NumBytesByteCodeTuple = namedtuple('TagToBytesResult', ['num_bytes', 'byte_code'])


class TimeBoxTag:
    def __init__(self, identifier, bytes_per_value: int, type_char: Union[int, str]):
        self.identifier = identifier
        self.bytes_per_value = bytes_per_value
        self.type_char = get_type_char_char(type_char)
        self.dtype = get_numpy_type(
            self.type_char,
            self.bytes_per_value * 8
        )
        return

    def to_bytes(self, num_bytes_for_tag_identifier: int, tag_identifier_is_string: bool) -> NumBytesByteCodeTuple:
        """
        Sends the tag definition to binary form.
        :param num_bytes_for_tag_identifier: number of bytes used in the unsigned int or unicode tag identifier
        :param tag_identifier_is_string: if True, tag identifier will be treated as 4-byte unicode. if False, int
        :return: namedtuple TagToBytesResult like ('num_bytes', 'byte_code')
        """
        info = np.array(
            [(self.identifier, self.bytes_per_value, get_type_char_int(self.type_char))],
            dtype=TimeBoxTag.tag_info_dtype(num_bytes_for_tag_identifier, tag_identifier_is_string)
        )
        return NumBytesByteCodeTuple(num_bytes=info.nbytes, byte_code=info.tobytes())

    @classmethod
    def tag_info_dtype(cls, num_bytes_for_tag_identifier: int, tag_identifier_is_string: bool) -> np.dtype:
        """
        gets a dtype object that will be used to extract tag name info
        :param num_bytes_for_tag_identifier: number of bytes used in the unsigned int or unicode tag identifier
        :param tag_identifier_is_string: if True, tag identifier will be treated as 4-byte unicode. if False, int
        :return: dtype object
        """
        if tag_identifier_is_string:
            if num_bytes_for_tag_identifier <= 0:
                raise TagIdentifierByteRepresentationError('Number of bytes for tag identifier cannot be zero')
            try:
                id_type = np.dtype([('tag_identifier', '<U{}'.format(ensure_int(num_bytes_for_tag_identifier / 4)))])
            except NotIntegerException:
                raise TagIdentifierByteRepresentationError(
                    'Number of bytes for tag identifier must be multiple of 4'
                )
        else:
            id_type = np.dtype([('tag_identifier', get_numpy_type('u', num_bytes_for_tag_identifier * 8))])
        return np.dtype([id_type.descr[0], ('bytes_per_point', np.uint8), ('type_char', np.uint8)])

    @classmethod
    def tag_definitions_from_bytes(cls, from_bytes: str, num_bytes_for_identifier: int,
                                   tag_names_are_strings: bool) -> dict:
        """
        Reads the tag definitions from bytes
        :param from_bytes: string representation of binary bytes read from a file
        :param num_bytes_for_identifier: integer specifying number of bytes for the tag identifier
        :param tag_names_are_strings: whether or not the tag identifiers are strings
        :return: dictionary like {identifier : TimeBoxTag}
        """
        tags = np.frombuffer(
            from_bytes,
            dtype=TimeBoxTag.tag_info_dtype(num_bytes_for_identifier,tag_names_are_strings)
        )
        return dict([(t['tag_identifier'], TimeBoxTag(t['tag_identifier'], t['bytes_per_point'], t['type_char'])) for t in tags])

    @classmethod
    def tag_list_to_bytes(cls, tag_list: list, num_bytes_for_tag_identifier: int,
                          tag_identifier_is_string: bool) -> NumBytesByteCodeTuple:
        """
        Executes to_bytes() on each element in tag_list, then combines the result into a NumBytesByteCodeTuple
        :param tag_list: list of TimeBoxTag items
        :param num_bytes_for_tag_identifier: number of bytes used in the unsigned int or unicode tag identifier
        :param tag_identifier_is_string: if True, tag identifier will be treated as 4-byte unicode. if False, int
        :return: NumBytesByteCodeTuple object, summed/joined across the tags
        """
        tags_to_bytes_result = [t.to_bytes(num_bytes_for_tag_identifier, tag_identifier_is_string) for t in tag_list]
        num_bytes = sum([r[0] for r in tags_to_bytes_result])
        byte_code = b''.join([r[1] for r in tags_to_bytes_result])
        return NumBytesByteCodeTuple(num_bytes=num_bytes, byte_code=byte_code)
