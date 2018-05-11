from ..timebox import TimeBox
from ..tag_info import TagInfo
from ..utils.datetime_utils import SECONDS
import unittest
import numpy as np
import os


def example_tag_definitions():
    return {
        0: TagInfo(0, 1, 'u'),
        1: TagInfo(1, 2, 'i'),
        2: TagInfo(2, 4, 'f'),
        255: TagInfo(255, 8, 'i'),
        256: TagInfo(256, 8, 'f')
    }


class TestTimeBoxFileInfo(unittest.TestCase):
    def test_init(self):
        tb = TimeBox('test_file_path.txt')
        self.assertEqual('test_file_path.txt', tb._file_path)
        return

    def test_update_required_bytes(self):
        tb = TimeBox('')
        tb._tag_names_are_strings = False
        tb._tag_definitions[0] = TagInfo(0, 1, 'u')
        tb._update_required_bytes_for_tag_identifier()
        self.assertEqual(1, tb._num_bytes_for_tag_identifier)
        tb._tag_definitions[256] = TagInfo(256, 1, 'u')
        tb._update_required_bytes_for_tag_identifier()
        self.assertEqual(2, tb._num_bytes_for_tag_identifier)

        tb._tag_names_are_strings = True
        tb._tag_definitions = {
            'a': TagInfo('a', 1, 'u'),
            'ab': TagInfo('ab', 1, 'u'),
            'abc': TagInfo('abc', 1, 'u')
        }
        tb._update_required_bytes_for_tag_identifier()
        self.assertEqual(12, tb._num_bytes_for_tag_identifier)
        return

    def test_tag_definitions_to_from_bytes_integer(self):
        first = TimeBox('')
        first._tag_names_are_strings = False
        first._tag_definitions = example_tag_definitions()
        first._update_required_bytes_for_tag_identifier()
        first_bytes = first._tag_definitions_to_bytes()
        self.assertEqual(5 * (2 + 1 + 1), first_bytes[0])
        self.assertEqual(813, np.frombuffer(first_bytes[1], dtype=np.uint8).sum())

        second = TimeBox('')
        second._num_bytes_for_tag_identifier = 2
        second._tag_names_are_strings = False
        second._unpack_tag_definitions(first_bytes[1])
        second_bytes = second._tag_definitions_to_bytes()
        self.assertEqual(first_bytes[0], second_bytes[0])
        self.assertEqual(first_bytes[1], second_bytes[1])

        second._num_bytes_for_tag_identifier = 4
        bad_bytes = second._tag_definitions_to_bytes()
        self.assertNotEqual(second_bytes[0], bad_bytes[0])

        second._tag_names_are_strings = True
        bad_bytes = second._tag_definitions_to_bytes()
        self.assertNotEqual(second_bytes[0], bad_bytes[0])
        return

    def test_unpack_options(self):
        # we are looking at individual bits.
        # so far only 2 options, so we'll test integer
        # values 0, 1, 2, and 3
        # 0000 0000
        # 0000 0001
        # 0000 0010
        # 0000 0011
        tb = TimeBox('')

        tb._unpack_options(0)
        self.assertFalse(tb._tag_names_are_strings)
        self.assertFalse(tb._date_differentials_stored)

        tb._unpack_options(1)
        self.assertTrue(tb._tag_names_are_strings)
        self.assertFalse(tb._date_differentials_stored)

        tb._unpack_options(2)
        self.assertFalse(tb._tag_names_are_strings)
        self.assertTrue(tb._date_differentials_stored)

        tb._unpack_options(3)
        self.assertTrue(tb._tag_names_are_strings)
        self.assertTrue(tb._date_differentials_stored)
        return

    def test_encode_options(self):
        # we are looking at individual bits.
        # so far only 2 options, so we'll test integer
        # values 0, 1, 2, and 3
        # 0000 0000
        # 0000 0001
        # 0000 0010
        # 0000 0011
        tb = TimeBox('')

        tb._tag_names_are_strings = False
        tb._date_differentials_stored = False
        self.assertEqual(0, tb._encode_options())

        tb._tag_names_are_strings = True
        tb._date_differentials_stored = False
        self.assertEqual(1, tb._encode_options())

        tb._tag_names_are_strings = False
        tb._date_differentials_stored = True
        self.assertEqual(2, tb._encode_options())

        tb._tag_names_are_strings = True
        tb._date_differentials_stored = True
        self.assertEqual(3, tb._encode_options())
        return

    def test_read_write_file_info_uniform_dates(self):
        tb = TimeBox('')
        tb._timebox_version = 1
        tb._tag_names_are_strings = False
        tb._date_differentials_stored = False
        tb._num_points = 10
        tb._tag_definitions = example_tag_definitions()
        tb._start_date = np.datetime64('2018-01-01', 's')
        tb._seconds_between_points = 3600

        file_name = 'test.npb'
        with open(file_name, 'wb') as f:
            self.assertEqual(41, tb._write_file_info(f))

        tb_read = TimeBox('')
        with open(file_name, 'rb') as f:
            self.assertEqual(41, tb_read._read_file_info(f))

        self.assertEqual(tb._timebox_version, tb_read._timebox_version)
        self.assertEqual(tb._tag_names_are_strings, tb_read._tag_names_are_strings)
        self.assertEqual(tb._date_differentials_stored, tb_read._date_differentials_stored)
        self.assertEqual(tb._num_points, tb_read._num_points)
        self.assertEqual(tb._start_date, tb_read._start_date)
        self.assertEqual(tb._seconds_between_points, tb_read._seconds_between_points)
        for t in tb._tag_definitions:
            self.assertTrue(t in tb_read._tag_definitions)
            self.assertEqual(tb._tag_definitions[t].identifier, tb_read._tag_definitions[t].identifier)
            self.assertEqual(tb._tag_definitions[t].type_char, tb_read._tag_definitions[t].type_char)
            self.assertEqual(tb._tag_definitions[t].dtype, tb_read._tag_definitions[t].dtype)
            self.assertEqual(tb._tag_definitions[t].bytes_per_value, tb_read._tag_definitions[t].bytes_per_value)

        os.remove(file_name)
        return

    def test_read_write_file_info_date_deltas(self):
        tb = TimeBox('')
        tb._timebox_version = 1
        tb._tag_names_are_strings = False
        tb._date_differentials_stored = True
        tb._num_points = 10
        tb._tag_definitions = example_tag_definitions()
        tb._start_date = np.datetime64('2018-01-01', 's')
        tb._bytes_per_date_differential = 4
        tb._date_differential_units = SECONDS

        file_name = 'test_delta.npb'
        with open(file_name, 'wb') as f:
            self.assertEqual(40, tb._write_file_info(f))

        tb_read = TimeBox('')
        with open(file_name, 'rb') as f:
            self.assertEqual(40, tb_read._read_file_info(f))

        self.assertEqual(tb._timebox_version, tb_read._timebox_version)
        self.assertEqual(tb._tag_names_are_strings, tb_read._tag_names_are_strings)
        self.assertEqual(tb._date_differentials_stored, tb_read._date_differentials_stored)
        self.assertEqual(tb._num_points, tb_read._num_points)
        self.assertEqual(tb._start_date, tb_read._start_date)
        self.assertEqual(tb._bytes_per_date_differential, tb_read._bytes_per_date_differential)
        self.assertEqual(tb._date_differential_units, tb_read._date_differential_units)
        for t in tb._tag_definitions:
            self.assertTrue(t in tb_read._tag_definitions)
            self.assertEqual(tb._tag_definitions[t].identifier, tb_read._tag_definitions[t].identifier)
            self.assertEqual(tb._tag_definitions[t].type_char, tb_read._tag_definitions[t].type_char)
            self.assertEqual(tb._tag_definitions[t].dtype, tb_read._tag_definitions[t].dtype)
            self.assertEqual(tb._tag_definitions[t].bytes_per_value, tb_read._tag_definitions[t].bytes_per_value)

        os.remove(file_name)
        return

if __name__ == '__main__':
    unittest.main()
