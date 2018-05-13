import numpy as np
import pandas as pd
import os
import time
from fcntl import flock, LOCK_EX, LOCK_SH, LOCK_UN, LOCK_NB
from .utils.datetime_utils import compress_time_delta_array, get_unit_data
from .utils.numpy_utils import *
from .utils.validation import ensure_int
from .utils.exceptions import NotIntegerException
from .utils.binary import determine_required_bytes_unsigned_integer, read_unsigned_int
from .utils.pandas_utils import parse_pandas_dtype
from .constants import *
from .tag_info import TagInfo
from .exceptions import *


MAX_WRITE_BLOCK_WAIT_SECONDS = 60
MAX_READ_BLOCK_WAIT_SECONDS = 30


class TimeBox:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._timebox_version = 1
        self._tag_names_are_strings = False
        self._date_differentials_stored = True
        self._num_points = 0
        self._tag_definitions = {}  # like { int|string tag_identifier }
        self._start_date = None
        self._seconds_between_points = 0
        self._bytes_per_date_differential = 0
        self._date_differential_units = 0
        self._data = {}  # like { tag_identifier: numpy array }
        self._date_differentials = None  # numpy array
        self._dates = None  # numpy array of datetime64[s]
        self._MAX_WRITE_BLOCK_WAIT_SECONDS = MAX_WRITE_BLOCK_WAIT_SECONDS
        self._MAX_READ_BLOCK_WAIT_SECONDS = MAX_READ_BLOCK_WAIT_SECONDS
        return

    @classmethod
    def save_pandas(cls, df: pd.DataFrame, file_path: str):
        """
        Expects that the passing df has an index that is type Timestamp
        or string which can be converted to Timestamp. All dtypes in pandas
        data frame must be in the float/int/u-int family
        :param df: pandas DataFrame
        :param file_path: file path to save pandas DataFrame
        :return: TimeBox object
        """
        # make sure the pandas data frame is sorted on date
        df = df.sort_index()

        tb = TimeBox(file_path)
        tb._tag_names_are_strings = True

        # ensure index is there and can be converted to numpy array of datetime64s
        tb._dates = df.index.values.astype(np.datetime64)
        tb._start_date = np.amin(tb._dates.astype(np.dtype('datetime64[s]')))
        tb._date_differentials_stored = True
        tb._num_points = tb._dates.size

        # get column names and info
        for c in df.columns:
            type_info = parse_pandas_dtype(df[c].dtype)
            tb._tag_definitions[c] = TagInfo(c, type_info[0], type_info[1])
            tb._data[c] = df[c].values

        try:
            tb.write()
        except DateUnitsError:
            raise InvalidPandasIndexError('There was an error reading the date-time index on data frame')
        return tb

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

    def to_pandas(self) -> pd.DataFrame:
        """
        Populates a pandas data frame and returns it.
        :return: Pandas DataFrame
        """
        # check if haven't read
        if len(self._data) == 0:
            self.read()
        data = [(t, self._data[t]) for t in self._data]
        data.append(('DateTimes', self._dates))
        df = pd.DataFrame.from_items(data)
        return df.set_index('DateTimes')

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

    def _encode_options(self) -> int:
        """
        Stores the bit-options in a 16-bit integer
        :return: int, no more than 16 bits
        """
        # note, this needs to be in the opposite order as _unpack_options
        options = 0
        options |= 1 if self._date_differentials_stored else 0
        options <<= 1
        options |= 1 if self._tag_names_are_strings else 0
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
            self._num_bytes_for_tag_identifier = determine_required_bytes_unsigned_integer(
                max([k for k in self._tag_definitions])
            )
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

    def _read_file_info(self, file_handle) -> int:
        """
        Reads the file info from a file_handle. Populates file internals
        :param file_handle: file handle object in 'rb' mode that is seeked to the correct position (0)
        :return: int, seek bytes increased since file_handle was received
        """
        self._timebox_version = read_unsigned_int(file_handle.read(1))
        self._unpack_options(int(read_unsigned_int(file_handle.read(2))))
        num_tags = read_unsigned_int(file_handle.read(1))
        self._num_points = read_unsigned_int(file_handle.read(4))
        self._num_bytes_for_tag_identifier = read_unsigned_int(file_handle.read(1))
        bytes_seek = 1 + 2 + 1 + 4 + 1

        # first 2 bytes are info on the tag
        bytes_for_tag_def = num_tags * (1 + 1 + self._num_bytes_for_tag_identifier)
        self._unpack_tag_definitions(file_handle.read(bytes_for_tag_def))
        bytes_seek += bytes_for_tag_def

        self._start_date = np.fromfile(file_handle, dtype='datetime64[s]', count=1)[0]
        bytes_seek += 8

        if self._date_differentials_stored:
            self._seconds_between_points = 0
            self._bytes_per_date_differential = read_unsigned_int(file_handle.read(1))
            stored_value_for_date_diff_units = read_unsigned_int(file_handle.read(2))
            self._date_differential_units = get_date_utils_constant_from_stored_units_int(
                stored_value_for_date_diff_units
            )
            bytes_seek += 3
        else:
            self._seconds_between_points = read_unsigned_int(file_handle.read(4))
            self._bytes_per_date_differential = 0
            self._date_differential_units = 0
            bytes_seek += 4
        return bytes_seek

    def _write_file_info(self, file_handle) -> int:
        """
        Writes out the file info to the file handle
        :param file_handle: file handle object in 'wb' mode. pre-seeked to correct position (0)
        :return: int, seek bytes advanced in this method
        """
        np.array([np.uint8(self._timebox_version)], dtype=np.uint8).tofile(file_handle)
        np.array([np.uint16(self._encode_options())], dtype=np.uint16).tofile(file_handle)
        np.array([np.uint8(len(self._tag_definitions))], dtype=np.uint8).tofile(file_handle)
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
            int_to_store_date_diff_units = get_int_for_date_units_from_date_utils_constant(
                self._date_differential_units
            )
            np.array([np.uint16(int_to_store_date_diff_units)], dtype=np.uint16).tofile(file_handle)
            bytes_seek += 3
        else:
            np.array([np.uint32(self._seconds_between_points)], dtype=np.uint32).tofile(file_handle)
            bytes_seek += 4

        return bytes_seek

    def _validate_data_for_write(self):
        """
        This method checks the data to ensure that the tag data is within date ranges, etc.
        :return: void
        """
        if len(self._tag_definitions) != len(self._data):
            raise DataDoesNotMatchTagDefinitionError('Tag definition length is different than tag data length')
        if len([t for t in self._tag_definitions if t not in self._data]) > 0:
            raise DataDoesNotMatchTagDefinitionError('Tag identifier in tag definitions not found in data')
        for t in self._tag_definitions:
            if self._data[t].dtype != self._tag_definitions[t].dtype:
                raise DataDoesNotMatchTagDefinitionError('Data for tag {} does not have correct '
                                                         'dtype {}'.format(t, self._tag_definitions[t].dtype))
            if self._data[t].size != self._num_points:
                raise DataShapeError('Data for tag {} does not have the correct shape'.format(t))

        if self._date_differentials_stored:
            if self._date_differentials.dtype != get_numpy_type('u', 8 * self._bytes_per_date_differential):
                raise DateDataError('Date differential dtype does not match bytes per date differential.')
            if self._date_differentials.size != (self._num_points - 1):
                raise DateDataError('Date differential array does not have the correct shape')
        else:  # date differentials aren't stored
            if self._seconds_between_points <= 0:
                raise DateDataError('Seconds between points must be positive')
        return

    def _write_tag_data(self, file_handle) -> int:
        """
        writes out the tag data, first writing the booleans (TODO) then writing the actual data
        :param file_handle: file handle object in 'wb' mode, pre-seeked to correct position
        :return: int, seek bytes advanced in this method
        """
        self._validate_data_for_write()
        seek_bytes = 0

        # then write out file data
        for t in self._data:
            seek_bytes += self._data[t].nbytes
            self._data[t].tofile(file_handle)
        return seek_bytes

    def _read_tag_data(self, file_handle) -> int:
        """
        reads in tag data from the file handle
        :param file_handle: file handle in 'rb' mode, pre-seeked to the correct starting position
        :return: int, seek bytes advanced in this method
        """
        seek_bytes = 0
        self._data = {}
        for t in self._tag_definitions:
            seek_bytes += self._num_points * self._tag_definitions[t].bytes_per_value
            self._data[t] = np.fromfile(
                file_handle,
                dtype=get_numpy_type(self._tag_definitions[t].type_char, 8 * self._tag_definitions[t].bytes_per_value),
                count=self._num_points
            )
        return seek_bytes

    def _write_date_deltas(self, file_handle) -> int:
        """
        writes out the date differentials
        :param file_handle: file handle object in 'wb' mode, pre-seeked to the correct position
        :return: int, seek bytes advanced in this method
        """
        self._date_differentials.tofile(file_handle)
        return self._date_differentials.nbytes

    def _read_date_deltas(self, file_handle) -> int:
        """
        reads the date differentials
        :param file_handle: file handle object in 'rb' mode, pre-seeked to the correct position
        :return: int, seek bytes advanced in this method
        """
        self._date_differentials = np.fromfile(
            file_handle,
            dtype=get_numpy_type('u', 8 * self._bytes_per_date_differential),
            count=self._num_points-1
        )

        # populate dates array
        unit_data = get_unit_data(self._date_differential_units)
        data_type = np.dtype('timedelta64[{}]'.format(unit_data.units))
        cumulative_time_deltas = np.cumsum(self._date_differentials.astype(data_type))
        dates = cumulative_time_deltas + self._start_date
        self._dates = np.insert(dates, 0, self._start_date)
        return self._date_differentials.nbytes

    def _calculate_date_differentials(self):
        """
        Calculates the date differentials array from the _dates array
        :return: void
        """
        self._start_date = np.amin(self._dates)
        differences = np.ediff1d(self._dates)
        # ensure that the dates are sorted
        if np.amin(differences).astype(np.int64) < 0:
            raise DateDataError('Dates were not in order')
        self._date_differentials = differences
        return

    def _compress_date_differentials(self):
        """
        Tries to compress date differentials from their original units to something else,
        then the actual array is compressed
        :return: void
        """
        result = compress_time_delta_array(self._date_differentials)
        unit_data = get_unit_data(result[1])
        self._date_differential_units = unit_data.order
        max_diff = np.amax(result[0])
        bytes_needed = determine_required_bytes_unsigned_integer(max_diff)
        self._date_differentials = result[0].astype(get_numpy_type('u', 8 * bytes_needed))
        self._bytes_per_date_differential = bytes_needed
        return

    def _blocking_file_name(self) -> str:
        """
        returns a file blocking name
        :return: file name of blocking file
        """
        return '{}.lock'.format(self.file_path)

    def _get_fcntl_lock(self, mode: str = 'r'):
        """
        gets a lock of type 'w' (writing) or 'r' (reading). throws error if can't get lock in time
        this is a blocking function, but doesn't block for more than the specified
        _MAX_READ/WRITE_BLOCK_WAIT_SECONDS.
        :param mode: single char, 'w' or 'r'
        :return: file handle if succeeded, raise exception if failed
        """
        if mode not in ['r', 'w']:
            raise ValueError('Could not get fcntl lock because mode specified was invalid: {}'.format(mode))
        block_file_name = self._blocking_file_name()
        count = 0
        sleep_seconds = 0.1
        file_locked = False
        handle = open(self.file_path, 'rb' if mode is 'r' else 'wb')
        if mode == 'r':
            # check and see if a blocking file exists, meaning we're waiting for a write job to clear
            # then try to get the lock
            while not file_locked and count < (self._MAX_READ_BLOCK_WAIT_SECONDS / sleep_seconds):
                if not os.path.exists(block_file_name):
                    try:
                        flock(handle, LOCK_SH | LOCK_NB)
                        file_locked = True
                    except IOError:
                        pass
                count += 1
                time.sleep(sleep_seconds)
        if mode == 'w':
            block_file_is_mine = False
            while not file_locked and count < (self._MAX_WRITE_BLOCK_WAIT_SECONDS / sleep_seconds):
                if not os.path.exists(block_file_name):
                    # put a blocking file
                    open(block_file_name, 'w').close()
                    block_file_is_mine = True
                if block_file_is_mine and os.path.exists(block_file_name):
                    try:
                        flock(handle, LOCK_EX | LOCK_NB)
                        file_locked = True
                    except IOError:
                        pass
                count += 1
                time.sleep(sleep_seconds)
            if not file_locked and block_file_is_mine and os.path.exists(block_file_name):
                os.remove(block_file_name)
        if not file_locked:
            handle.close()
            raise CouldNotAcquireFileLockError
        return handle

    def read(self):
        """
        This function reads the entire file contents into memory.
        Later it can be improved to only read certain tags/dates
        :return: dictionary of results like {tag_identifier: numpy.array}
        """
        with self._get_fcntl_lock('r') as handle:
            try:
                # read in the data
                self._read_file_info(handle)

                if self._date_differentials_stored:
                    self._read_date_deltas(handle)

                self._read_tag_data(handle)
            finally:
                # release shared lock
                flock(handle, LOCK_UN)
        return

    def write(self):
        """
        writes the file out to file_name.
        requires an exclusive LOCK_EX fcntl lock.
        blocks until it can get a lock
        :return: void
        """
        # put a file in the same directory to block new shared requests
        # this prevents a popular file from blocking forever
        # note, this is a blocking function as it waits for other write events to finish
        with self._get_fcntl_lock('w') as handle:
            try:
                # prepare datetime data
                if self._date_differentials_stored:
                    self._calculate_date_differentials()
                    self._compress_date_differentials()

                self._write_file_info(handle)

                if self._date_differentials_stored:
                    self._write_date_deltas(handle)

                self._write_tag_data(handle)
            finally:
                flock(handle, LOCK_UN)  # release lock
                block_file_name = self._blocking_file_name()
                if os.path.exists(block_file_name):
                    os.remove(block_file_name)
        return
