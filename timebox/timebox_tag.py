import numpy as np
from collections import namedtuple
from typing import Union
from timebox.utils.numpy_utils import get_numpy_type, get_type_char_char, get_type_char_int
from timebox.exceptions import TagIdentifierByteRepresentationError
from timebox.utils.exceptions import NotIntegerException
from timebox.utils.validation import ensure_int
from timebox.constants import TimeBoxTagOptionPositions


NumBytesByteCodeTuple = namedtuple('TagToBytesResult', ['num_bytes', 'byte_code'])
NUM_BYTES_PER_DEFINITION_WITHOUT_IDENTIFIER = 40


class TimeBoxTag:
    def __init__(self, identifier, bytes_per_value: int, type_char: Union[int, str], options=None, untyped_bytes=None):
        self.identifier = identifier
        self.bytes_per_value = bytes_per_value
        self.type_char = get_type_char_char(type_char)
        self.dtype = get_numpy_type(
            self.type_char,
            self.bytes_per_value * 8
        )
        self.data = None
        self.compressed_type_char = None
        self.compressed_bytes_per_value = None
        self.compression_mode = None
        self.use_compression = False
        self.use_hash_table = False
        self.num_bytes_extra_information = 0
        if options is not None:
            self._decode_options(options)
        if untyped_bytes is not None:
            self._decode_def_bytes(untyped_bytes)
        return

    def to_bytes(self, num_bytes_for_tag_identifier: int, tag_identifier_is_string: bool) -> NumBytesByteCodeTuple:
        """
        Sends the tag definition to binary form.
        :param num_bytes_for_tag_identifier: number of bytes used in the unsigned int or unicode tag identifier
        :param tag_identifier_is_string: if True, tag identifier will be treated as 4-byte unicode. if False, int
        :return: namedtuple TagToBytesResult like ('num_bytes', 'byte_code')
        """
        options = np.uint16(self._encode_options())
        def_bytes = self._encode_def_bytes()
        info = np.array(
            [(
                self.identifier,
                options,
                self.bytes_per_value,
                get_type_char_int(self.type_char),
                self.num_bytes_extra_information
            )],
            dtype=TimeBoxTag.tag_info_dtype(
                num_bytes_for_tag_identifier,
                tag_identifier_is_string,
                exclude_trailing_bytes=True
            )
        )
        ret_bytes = info.tobytes() + def_bytes
        num_bytes = info.nbytes + 32
        return NumBytesByteCodeTuple(num_bytes=num_bytes, byte_code=ret_bytes)

    def _encode_options(self) -> int:
        """
        Encodes 16 bit options onto an integer
        :return: integer, no more than 16 bits (unsigned)
        """
        options = 0
        options |= 1 if self.use_hash_table else 0
        options <<= 1
        options |= 1 if self.use_compression else 0
        return options

    def _decode_options(self, from_int: int):
        """
        Decodes 16 bits of options from integer
        :param from_int: unsigned 16-bit integer to decode from
        :return: void, populates class internals
        """
        # starting with the right-most bits and working left
        self.use_compression = True if (from_int >> TimeBoxTagOptionPositions.USE_COMPRESSION.value) & 1 else False
        self.use_hash_table = True if (from_int >> TimeBoxTagOptionPositions.USE_HASH_TABLE.value) & 1 else False
        return

    def _encode_def_bytes(self) -> bytes:
        """
        Gets the 32-bytes of integer values to pass into the binary
        :return: byte-code of the 32-bytes
        """
        ret_bytes = [b'\x00' for _ in range(0, 32)]
        counter = 0
        if self.use_compression:
            ret_bytes[counter] = get_type_char_int(self.compression_mode).to_bytes(1, 'little')
            counter += 1
            ret_bytes[counter] = self.compressed_bytes_per_value.to_bytes(1, 'little')
            counter += 1
            ret_bytes[counter] = get_type_char_int(self.compressed_type_char).to_bytes(1, 'little')
            counter += 1
        return b''.join(ret_bytes)

    def _decode_def_bytes(self, from_bytes: bytes):
        """
        Decodes the 32-bytes of binary data to populate class internals
        :param from_bytes: 32-bytes of data to decode
        :return: None, populates class internals
        """
        counter = 0
        if self.use_compression:
            compression_info = np.frombuffer(from_bytes[counter:(counter+3)], dtype=np.uint8, count=3)
            counter += 3
            self.compression_mode = get_type_char_char(compression_info[0])
            self.compressed_bytes_per_value = compression_info[1]
            self.compressed_type_char = get_type_char_char(compression_info[2])
        return

    @classmethod
    def tag_info_dtype(cls, num_bytes_for_tag_identifier: int, tag_identifier_is_string: bool,
                       exclude_trailing_bytes: bool = False) -> np.dtype:
        """
        gets a dtype object that will be used to extract tag name info
        :param num_bytes_for_tag_identifier: number of bytes used in the unsigned int or unicode tag identifier
        :param tag_identifier_is_string: if True, tag identifier will be treated as 4-byte unicode. if False, int
        :param exclude_trailing_bytes: Whether on not to exclude the trailing 32-bytes of definition
        :return: dtype object
        """
        dtype_list = []
        if tag_identifier_is_string:
            if num_bytes_for_tag_identifier <= 0:
                raise TagIdentifierByteRepresentationError('Number of bytes for tag identifier cannot be zero')
            try:
                dtype_list.append((
                    'tag_identifier',
                    '<U{}'.format(ensure_int(num_bytes_for_tag_identifier / 4))
                ))
            except NotIntegerException:
                raise TagIdentifierByteRepresentationError(
                    'Number of bytes for tag identifier must be multiple of 4'
                )
        else:
            dtype_list.append((
                'tag_identifier',
                get_numpy_type('u', 8 * num_bytes_for_tag_identifier)
            ))
        # 2 bytes of options
        dtype_list.append(('options', np.uint16))
        # bytes per point
        dtype_list.append(('bytes_per_point', np.uint8))
        # type char
        dtype_list.append(('type_char', np.uint8))
        # 4-bytes indicating length of extra information
        dtype_list.append(('bytes_extra_information', np.uint32))
        if not exclude_trailing_bytes:
            dtype_list.extend([('def_byte_{}'.format(i + 1), np.uint8) for i in range(0, 32)])
        return np.dtype(dtype_list)

    @classmethod
    def tag_definitions_from_bytes(cls, from_bytes: bytes, num_bytes_for_identifier: int,
                                   tag_names_are_strings: bool) -> dict:
        """
        Reads the tag definitions from bytes
        :param from_bytes: binary bytes read from a file
        :param num_bytes_for_identifier: integer specifying number of bytes for the tag identifier
        :param tag_names_are_strings: whether or not the tag identifiers are strings
        :return: dictionary like {identifier : TimeBoxTag}
        """
        bytes_per_tag_def = num_bytes_for_identifier + NUM_BYTES_PER_DEFINITION_WITHOUT_IDENTIFIER
        num_typed_bytes = bytes_per_tag_def - 32
        split_bytes = [from_bytes[i:i+bytes_per_tag_def] for i in range(0, len(from_bytes), bytes_per_tag_def)]
        typed_bytes = [s[0:num_typed_bytes] for s in split_bytes]
        untyped_bytes = [s[num_typed_bytes:] for s in split_bytes]
        raw_tags = np.frombuffer(
            b''.join(typed_bytes),
            dtype=TimeBoxTag.tag_info_dtype(
                num_bytes_for_identifier,
                tag_names_are_strings,
                exclude_trailing_bytes=True
            )
        )
        tags = []
        for i in range(0, raw_tags.shape[0]):
            tag = TimeBoxTag(
                raw_tags[i]['tag_identifier'],
                raw_tags[i]['bytes_per_point'],
                raw_tags[i]['type_char'],
                options=raw_tags[i]['options']
            )
            tag._decode_def_bytes(untyped_bytes[i])
            tags.append(tag)
        tags = [
            TimeBoxTag(t['tag_identifier'], t['bytes_per_point'], t['type_char'], options=t['options'])
            for t in raw_tags
        ]
        for i in range(0, len(tags)):
            tags[i]._decode_def_bytes(untyped_bytes[i])

        return dict([(t.identifier, t) for t in tags])

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
