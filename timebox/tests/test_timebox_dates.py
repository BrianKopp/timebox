from ..exceptions import *
from ..timebox import TimeBox
from ..tag_info import TagInfo
import unittest
import numpy as np
import os


def example_time_box(file_name: str):
    tb = TimeBox(file_name)
    tb._timebox_version = 1
    tb._tag_names_are_strings = False
    tb._date_differentials_stored = True
    tb._num_points = 4
    tb._tag_definitions = {
        0: TagInfo(0, 1, 'u'),
        1: TagInfo(1, 2, 'i'),
        2: TagInfo(2, 4, 'f')
    }
    tb._start_date = np.datetime64('2018-01-01', 's')
    tb._data = {
        0: np.array([1, 2, 3, 4], dtype=np.uint8),
        1: np.array([-4, -2, 0, 2000], dtype=np.int16),
        2: np.array([5.2, 0.8, 3.1415, 8], dtype=np.float32)
    }
    tb._date_differentials = np.array([1, 1, 1], dtype=np.uint8)
    tb._bytes_per_date_differential = 1
    return tb


class TestTimeBoxDateData(unittest.TestCase):
    def test_date_validation_errors(self):
        file_name = 'test_date_data.npb'
        tb = example_time_box(file_name)
        tb._validate_data_for_write()  # pass

        tb._date_differentials_stored = False
        with self.assertRaises(DateDataError):
            tb._validate_data_for_write()

        tb._date_differentials_stored = True
        tb._date_differentials = tb._date_differentials.astype(np.int8)
        with self.assertRaises(DateDataError):
            tb._validate_data_for_write()

        tb._date_differentials = tb._date_differentials.astype(np.uint32)
        with self.assertRaises(DateDataError):
            tb._validate_data_for_write()

        tb._date_differentials = np.array([1, 1, 1, 1], dtype=np.uint8)
        with self.assertRaises(DateDataError):
            tb._validate_data_for_write()
        return

    def test_date_differential_io(self):
        file_name = 'test_date_data.npb'
        tb = example_time_box(file_name)
        with open(file_name, 'wb') as f:
            self.assertEqual(3, tb._write_date_deltas(f))

        with open(file_name, 'rb') as f:
            self.assertEqual(3, tb._read_date_deltas(f))

        self.assertEqual(np.uint8, tb._date_differentials.dtype)
        self.assertEqual(3, tb._date_differentials.size)
        os.remove(file_name)
        return

if __name__ == '__main__':
    unittest.main()
