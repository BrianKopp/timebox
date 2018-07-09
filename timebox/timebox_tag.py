import numpy as np
import logging
from collections import namedtuple
from typing import Union
from timebox.utils.numpy_utils import get_numpy_type, get_type_char_char,\
    get_type_char_int, compress_array, decompress_array
from timebox.exceptions import TagIdentifierByteRepresentationError
from timebox.utils.exceptions import NotIntegerException
from timebox.utils.validation import ensure_int
from timebox.constants import TimeBoxTagOptionPositions
from math import pow


NumBytesByteCodeTuple = namedtuple('TagToBytesResult', ['num_bytes', 'byte_code'])
NUM_BYTES_PER_DEFINITION_WITHOUT_IDENTIFIER = 40


class TimeBoxTag:
    def __init__(self, identifier, bytes_per_value: int, type_char: Union[int, str],
                 options=None, untyped_bytes=None):
        """
        Initializes a TimeBoxTag object
        :param identifier: unique id for tag
        :param bytes_per_value: number of bytes needed to store the data in an uncompressed form
        :param type_char: 'i', 'u', or 'f' describing type of data
        :param options: Integer code for options
        :param untyped_bytes: byte-string containing the untyped 32-bytes of data used by the tag for different options
        """
        self.identifier = identifier
        self.bytes_per_value = bytes_per_value
        self.type_char = get_type_char_char(type_char)
        self.dtype = get_numpy_type(
            self.type_char,
            self.bytes_per_value * 8
        )
        self.num_bytes_extra_information = 0
        self.data = None
        self._encoded_data = None
        self.num_points = None

        # options
        self.use_compression = False
        self.use_hash_table = False
        self.floating_point_rounded = False
        if options is not None:
            self._decode_options(options)

        # options data from untyped bytes
        # compression data
        self._compressed_type_char = None
        self._compressed_bytes_per_value = None
        self._compression_mode = None
        self._compression_reference_value = None
        self._compression_reference_value_dtype = self.dtype

        # rounding data
        self.num_decimals_to_store = None

        if untyped_bytes is not None:
            self._decode_def_bytes(untyped_bytes)
        return

    def info_to_bytes(self, num_bytes_for_tag_identifier: int, tag_identifier_is_string: bool) -> NumBytesByteCodeTuple:
        """
        Sends the tag definition to binary form.
        :param num_bytes_for_tag_identifier: number of bytes used in the unsigned int or unicode tag identifier
        :param tag_identifier_is_string: if True, tag identifier will be treated as 4-byte unicode. if False, int
        :return: namedtuple TagToBytesResult like ('num_bytes', 'byte_code')
        """
        options = np.uint16(self._encode_options())
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
        logging.debug('Sending tag "{}" info to bytes.'.format(self.identifier))
        logging.debug('Bytes per value: {}'.format(self.bytes_per_value))
        logging.debug('Type char: {}'.format(self.type_char))
        logging.debug('Num bytes extra info: {}'.format(self.num_bytes_extra_information))

        self._encoded_data = None
        self.encode_data()

        def_bytes = self._encode_def_bytes()
        ret_bytes = info.tobytes() + def_bytes
        num_bytes = info.nbytes + 32
        return NumBytesByteCodeTuple(num_bytes=num_bytes, byte_code=ret_bytes)

    def data_to_file(self, file_handle) -> int:
        """
        Sends the binary data to the file handle at the current seek position.
        :param file_handle: file handle in mode 'wb' at the current seek position
        :return: int number of bytes written
        """
        self.encode_data()
        self._encoded_data.tofile(file_handle)
        return self._encoded_data.nbytes

    def fill_data_from_file(self, file_handle, num_points: int) -> int:
        """
        reads in tag data from file handle
        :param file_handle: file handle in 'rb' mode at correct seek position
        :param num_points: number of points to extract from the file
        :return: int, num bytes read from file
        """
        self.num_points = num_points
        read_num_points = num_points
        read_dtype = self.dtype
        if self.use_compression:
            read_dtype = get_numpy_type(self._compressed_type_char, self._compressed_bytes_per_value * 8)
            if self._compression_mode == 'e':
                read_num_points -= 1
        self._encoded_data = np.fromfile(
            file_handle,
            read_dtype,
            count=read_num_points
        )
        self._decode_data()
        return self._encoded_data.nbytes

    def _encode_options(self) -> int:
        """
        Encodes 16 bit options onto an integer
        :return: integer, no more than 16 bits (unsigned)
        """
        options = 0
        options |= 1 if self.floating_point_rounded else 0
        options <<= 1
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
        compression_result = (from_int >> TimeBoxTagOptionPositions.USE_COMPRESSION.value) & 1
        self.use_compression = True if compression_result else False
        hash_result = (from_int >> TimeBoxTagOptionPositions.USE_HASH_TABLE.value) & 1
        self.use_hash_table = True if hash_result else False
        rounding_result = (from_int >> TimeBoxTagOptionPositions.FLOATING_POINT_ROUNDED.value) & 1
        self.floating_point_rounded = True if rounding_result else False
        return

    def _encode_def_bytes(self) -> bytes:
        """
        Gets the 32-bytes of integer values to pass into the binary
        :return: byte-code of the 32-bytes
        """
        ret_bytes = [b'\x00' for _ in range(0, 32)]
        counter = 0
        if self.use_compression:
            ret_bytes[counter] = get_type_char_int(self._compression_mode).to_bytes(1, 'little')
            counter += 1
            ret_bytes[counter] = self._compressed_bytes_per_value.to_bytes(1, 'little')
            counter += 1
            ret_bytes[counter] = get_type_char_int(self._compressed_type_char).to_bytes(1, 'little')
            counter += 1
            ret_bytes[counter] = self._compression_reference_value_dtype.itemsize.to_bytes(1, 'little')
            counter += 1
            ret_bytes[counter] = get_type_char_int(self._compression_reference_value_dtype.kind).to_bytes(1, 'little')
            counter += 1
            reference_value_bytes = np.array(
                [self._compression_reference_value],
                dtype=self._compression_reference_value_dtype
            ).tobytes()
            for i in range(0, len(reference_value_bytes)):
                ret_bytes[counter] = reference_value_bytes[i].to_bytes(1, 'little')
                counter += 1
        if self.floating_point_rounded:
            ret_bytes[counter] = self.num_decimals_to_store.to_bytes(1, 'little')
            counter += 1
        logging.debug('Encoded definition:')
        logging.debug('\tCompression mode: {}'.format(self._compression_mode))
        logging.debug('\tCompression bytes: {}'.format(self._compressed_bytes_per_value))
        logging.debug('\tCompression type char: {}'.format(self._compressed_type_char))
        logging.debug('\tCompression ref val: {}'.format(self._compression_reference_value))
        logging.debug('\tCompression ref val dtype: {}'.format(self._compression_reference_value_dtype))
        return b''.join(ret_bytes)

    def _decode_def_bytes(self, from_bytes: bytes):
        """
        Decodes the 32-bytes of binary data to populate class internals
        :param from_bytes: 32-bytes of data to decode
        :return: None, populates class internals
        """
        counter = 0
        if self.use_compression:
            compression_info = np.frombuffer(from_bytes[counter:(counter+5)], dtype=np.uint8, count=5)
            counter += 5
            self._compression_mode = get_type_char_char(compression_info[0])
            self._compressed_bytes_per_value = compression_info[1]
            self._compressed_type_char = get_type_char_char(compression_info[2])
            self._compression_reference_value_dtype = get_numpy_type(
                get_type_char_char(compression_info[4]),
                compression_info[3] * 8
            )
            ref_value_bytes = self.bytes_per_value
            self._compression_reference_value = np.frombuffer(
                from_bytes[counter:counter+ref_value_bytes],
                dtype=self._compression_reference_value_dtype,
                count=1
            )[0]
            counter += ref_value_bytes
        if self.floating_point_rounded:
            self.num_decimals_to_store = from_bytes[counter]
            counter += 1
        logging.debug('Decoded definition for tag: {}'.format(self.identifier))
        logging.debug('\tCompression mode: {}'.format(self._compression_mode))
        logging.debug('\tCompression bytes: {}'.format(self._compressed_bytes_per_value))
        logging.debug('\tCompression type char: {}'.format(self._compressed_type_char))
        logging.debug('\tCompression ref val: {}'.format(self._compression_reference_value))
        logging.debug('\tCompression ref val dtype: {}'.format(self._compression_reference_value_dtype))
        return

    def encode_data(self):
        """
        Performs compression and alteration on data to produce data set that will be written in binary to file
        :return: None
        """
        if self._encoded_data is not None:
            return

        self._encoded_data = self.data
        if self.floating_point_rounded:
            self._encoded_data *= pow(10, self.num_decimals_to_store)
            self._encoded_data = np.around(self._encoded_data).astype(np.int64)
        if self.use_compression:
            self._compression_reference_value_dtype = self._encoded_data.dtype
            mode = 'm' if self._compression_mode is None else self._compression_mode
            compression_result = compress_array(self._encoded_data, mode)
            self._compression_mode = mode
            self._compressed_type_char = compression_result.numpy_array.dtype.kind
            self._compressed_bytes_per_value = compression_result.numpy_array.itemsize
            self._encoded_data = compression_result.numpy_array
            self._compression_reference_value = compression_result.reference_value
        return

    def _decode_data(self):
        """
        Decodes the data from a file buffer
        :return:
        """
        self.data = self._encoded_data
        if self.use_compression:
            self.data = decompress_array(
                self.data,
                self._compression_mode,
                self._compression_reference_value
            ).astype(self.dtype)
        if self.floating_point_rounded:
            self.data /= pow(10, self.num_decimals_to_store)
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
        logging.debug('converting tags to bytes: {}'.format([t.identifier for t in tag_list]))
        tags_to_bytes_result = [
            t.info_to_bytes(num_bytes_for_tag_identifier, tag_identifier_is_string)
            for t in tag_list
            ]
        num_bytes = sum([r[0] for r in tags_to_bytes_result])
        byte_code = b''.join([r[1] for r in tags_to_bytes_result])
        return NumBytesByteCodeTuple(num_bytes=num_bytes, byte_code=byte_code)
