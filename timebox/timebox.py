import numpy as np
from .utils.numpy_utils import get_numpy_type, get_type_char_int
from .utils.validation import ensure_int
from .utils.exceptions import NotIntegerException
from .utils.binary import determine_required_bytes
from .constants import *
from .tag_info import TagInfo
from .exceptions import TagIdentifierByteRepresentationError


class TimeBox:
    _file_path = None
    _timebox_version = 0
    _tag_names_are_strings = False
    _date_differentials_stored = True
    _num_points = 0
    _tag_definitions = {}  # like { int|string tag_identifier }
    _start_date = None
    _seconds_between_points = 0
    _bytes_per_date_differential = 0
    _date_differential_units = 0

    def __init__(self, file_path: str):
        self._file_path = file_path
        return

    @classmethod
    def _read_unsigned_int(cls, from_bytes: bytes) -> int:
        """
        from_bytes is the binary file contents to read from
        :param from_bytes: string representation of 1 binary byte read from a file
        :return: integer
        """
        return int.from_bytes(from_bytes, byteorder='big', signed=False)

    @classmethod
    def _get_tag_info_dtype(cls, num_bytes_for_tag_identifier: int, tag_identifier_is_string: bool) -> np.dtype:
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

    def _unpack_tag_definitions(self, from_bytes: str):
        """
        Reads the tag definitions from bytes
        :param from_bytes: string representation of binary bytes read from a file
        :return: void, populates class internals
        """
        tags = np.frombuffer(
            from_bytes,
            dtype=TimeBox._get_tag_info_dtype(
                self._num_bytes_for_tag_identifier,
                self._tag_names_are_strings
            )
        )
        self._tag_definitions = dict(
            [(
                t['tag_identifier'],
                TagInfo(t['tag_identifier'], t['bytes_per_point'], t['type_char'])
            ) for t in tags]
        )
        return

    def _unpack_options(self, from_int: int):
        """
        Reads the options from the 1-byte options bit
        :param from_int: int holding options
        :return: void, populates class internals
        """
        # starting with the right-most bits and working left
        self._tag_names_are_strings = True if (from_int >> TAG_NAME_BIT_POSITION) & 1 else False
        self._date_differentials_stored = True if (from_int >> DATE_DIFFERENTIALS_STORED_POSITION) & 1 else False
        return

    def _read_file_info(self, file_handle) -> int:
        """
        Reads the file info from a file_handle. Populates file internals
        :param file_handle: file handle object in 'rb' mode that is seeked to the correct position (0)
        :return: int, seek bytes increased since file_handle was received
        """
        self._timebox_version = TimeBox._read_unsigned_int(file_handle.read(1))
        self._unpack_options(int(TimeBox._read_unsigned_int(file_handle.read(2))))
        num_tags = TimeBox._read_unsigned_int(file_handle.read(1))
        self._num_points = TimeBox._read_unsigned_int(file_handle.read(4))
        self._num_bytes_for_tag_identifier = TimeBox._read_unsigned_int(file_handle.read(1))
        bytes_seek = 1 + 2 + 1 + 4 + 1

        # first 2 bytes are info on the tag
        bytes_for_tag_def = num_tags * (1 + 1 + self._num_bytes_for_tag_identifier)
        self._unpack_tag_definitions(file_handle.read(bytes_for_tag_def))
        bytes_seek += bytes_for_tag_def

        self._start_date = np.fromfile(file_handle, dtype='datetime64[s]', count=1)[0]
        bytes_seek += 8

        if self._date_differentials_stored:
            self._seconds_between_points = 0
            self._bytes_per_date_differential = TimeBox._read_unsigned_int(file_handle.read(1))
            self._date_differential_units = TimeBox._read_unsigned_int(file_handle.read(1))
            bytes_seek += 2
        else:
            self._seconds_between_points = TimeBox._read_unsigned_int(file_handle.read(4))
            self._bytes_per_date_differential = 0
            self._date_differential_units = 0
            bytes_seek += 4
        return bytes_seek

    def _encode_options(self) -> int:
        """
        Stores the bit-options in a 16-bit integer
        :return: int, no more than 16 bits
        """
        options = 1 if self._tag_names_are_strings else 0
        options <<= 1
        options |= 1 if self._date_differential_units else 0
        return options

    def _update_required_bytes_for_tag_identifier(self):
        """
        Looks at the tag list and determines what the max bytes required is
        :return: void, updates class internals
        """
        if self._tag_names_are_strings:
            max_length = max([len(k) for k in self._tag_definitions])
            self._num_bytes_for_tag_identifier = max_length * 4
        else:
            self._num_bytes_for_tag_identifier = determine_required_bytes(max([k for k in self._tag_definitions]))
        return

    def _tag_definitions_to_bytes(self) -> (int, bytes):
        """
        returns the tag definitions in binary form
        :return: tuple of number of bytes and then the actual bytes
        """
        a = np.array(
            [
                (
                    t,
                    self._tag_definitions[t].bytes_per_value,
                    get_type_char_int(self._tag_definitions[t].type_char)
                )
                for t in self._tag_definitions
            ],
            dtype=TimeBox._get_tag_info_dtype(self._num_bytes_for_tag_identifier, self._tag_names_are_strings)
        )
        return a.nbytes, a.tobytes()

    def _write_file_info(self, file_handle) -> int:
        """
        Writes out the file info to the file handle
        :param file_handle: file handle object in 'wb' mode. pre-seeked to correct position (0)
        :return: int, seek bytes advanced in this method
        """
        np.array([np.uint8(self._timebox_version)], dtype=np.uint8).tofile(file_handle)
        np.array([np.uint16(self._encode_options())], dtype=np.uint16).tofile(file_handle)
        np.array([np.uint8(self._num_tags)], dtype=np.uint8).tofile(file_handle)
        np.array([np.uint32(self._num_points)], dtype=np.uint32).tofile(file_handle)

        self._update_required_bytes_for_tag_identifier()
        np.array([np.uint8(self._num_bytes_for_tag_identifier)], dtype=np.uint8).tofile(file_handle)
        bytes_seek = 1 + 2 + 1 + 4 + 1

        tag_definition_bytes = self._tag_definitions_to_bytes()
        file_handle.write(tag_definition_bytes[1])
        bytes_seek += tag_definition_bytes[0]

        np.array([np.datetime64(self._start_date, dtype='datetime64[s]')]).tofile(file_handle)
        bytes_seek += 8

        if self._date_differentials_stored:
            np.array([np.uint8(self._bytes_per_date_differential)], dtype=np.uint8).tofile(file_handle)
            np.array([np.uint8(self._date_differential_units)], dtype=np.uint8).tofile(file_handle)
            bytes_seek += 2
        else:
            np.array([np.uint32(self._seconds_between_points)], dtype=np.uint32).tofile(file_handle)
            bytes_seek += 4

        return bytes_seek
