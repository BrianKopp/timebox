from timebox.exceptions import *
from timebox.timebox import TimeBox
from timebox.timebox_tag import TimeBoxTag
from ..utils.datetime_utils import DAYS, HOURS
import unittest
import numpy as np
import os


def example_time_box(file_name: str):
    tb = TimeBox(file_name)
    tb._timebox_version = 1
    tb._tag_names_are_strings = False
    tb._date_differentials_stored = True
    tb._num_points = 4
    tb._tags = {
        0: TimeBoxTag(0, 1, 'u'),
        1: TimeBoxTag(1, 2, 'i'),
        2: TimeBoxTag(2, 4, 'f')
    }
    tb._start_date = np.datetime64('2018-01-01', 's')

    tb._tags[0].data = np.array([1, 2, 3, 4], dtype=np.uint8)
    tb._tags[1].data = np.array([-4, -2, 0, 2000], dtype=np.int16)
    tb._tags[2].data = np.array([5.2, 0.8, 3.1415, 8], dtype=np.float32)

    tb._date_differentials = np.array([1, 1, 1], dtype=np.uint8)
    tb._date_differential_units = DAYS
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

        tb = example_time_box('')
        tb._date_differentials = None
        tb._dates = np.array(
            [
                np.datetime64('2018-01-05', 's'),
                np.datetime64('2018-01-04', 's'),
                np.datetime64('2018-01-03', 's'),
                np.datetime64('2018-01-02', 's')
            ],
            dtype=np.datetime64
        )
        with self.assertRaises(DateDataError):
            tb._calculate_date_differentials()
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

    def test_calculate_date_differentials(self):
        tb = example_time_box('')
        tb._date_differentials = None
        tb._dates = np.array(
            [
                np.datetime64('2018-01-01', 's'),
                np.datetime64('2018-01-02', 's'),
                np.datetime64('2018-01-03', 's'),
                np.datetime64('2018-01-05', 's')
            ]
        )
        tb._calculate_date_differentials()
        self.assertEqual(3, tb._date_differentials.size)
        self.assertEqual('timedelta64[s]', str(tb._date_differentials.dtype))
        self.assertEqual(86400, tb._date_differentials[0].astype(np.int64))
        self.assertEqual(86400, tb._date_differentials[1].astype(np.int64))
        self.assertEqual(2*86400, tb._date_differentials[2].astype(np.int64))
        return

    def test_compress_date_differentials(self):
        tb = example_time_box('')
        tb._date_differentials = None
        tb._dates = np.array(
            [
                np.datetime64('2018-01-01', 's'),
                np.datetime64('2018-01-02', 's'),
                np.datetime64('2018-01-03', 's'),
                np.datetime64('2018-01-05', 's')
            ]
        )
        tb._calculate_date_differentials()
        tb._compress_date_differentials()
        self.assertEqual(3, tb._date_differentials.size)
        self.assertEqual(np.uint8, tb._date_differentials.dtype)
        self.assertEqual(DAYS, tb._date_differential_units)
        self.assertEqual(1, tb._bytes_per_date_differential)
        self.assertEqual(1, tb._date_differentials[0])
        self.assertEqual(1, tb._date_differentials[1])
        self.assertEqual(2, tb._date_differentials[2])
        return

    def test_time_box_date_io(self):
        file_name = 'date_io.npb'
        tb = example_time_box(file_name)
        tb._date_differentials = None
        tb._dates = np.array(
            [
                np.datetime64('2018-01-01T00:00', 's'),
                np.datetime64('2018-01-02T12:00', 's'),
                np.datetime64('2018-01-03T05:00', 's'),
                np.datetime64('2018-01-05T00:00', 's')
            ]
        )
        tb.write()
        self.assertEqual(3, tb._date_differentials.size)
        self.assertEqual(np.uint8, tb._date_differentials.dtype)
        self.assertEqual(HOURS, tb._date_differential_units)
        self.assertEqual(24+12, tb._date_differentials[0])
        self.assertEqual(12+5, tb._date_differentials[1])
        self.assertEqual(19+24, tb._date_differentials[2])

        tb_new = TimeBox(file_name)
        tb_new.read()
        self.assertEqual(1, tb_new._timebox_version)
        self.assertFalse(tb_new._tag_names_are_strings)
        self.assertTrue(tb_new._date_differentials_stored)
        self.assertEqual(1, tb_new._bytes_per_date_differential)
        self.assertEqual(HOURS, tb_new._date_differential_units)
        self.assertEqual(np.uint8, tb_new._date_differentials.dtype)

        self.assertEqual(3, tb_new._date_differentials.size)
        self.assertEqual(24+12, tb_new._date_differentials[0])
        self.assertEqual(12+5, tb_new._date_differentials[1])
        self.assertEqual(19+24, tb_new._date_differentials[2])

        os.remove(file_name)
        return

    def test_time_box_date_io_error(self):
        file_name = 'date_io.npb'
        tb = example_time_box(file_name)
        tb._date_differentials = None
        tb._dates = np.array(
            [
                np.datetime64('2018-01-01T00:00', 's'),
                np.datetime64('2018-01-02T12:00', 's'),
                np.datetime64('2018-01-01T05:00', 's'),
                np.datetime64('2018-01-05T00:00', 's')
            ]
        )
        with self.assertRaises(DateDataError):
            tb.write()
        self.assertFalse(os.path.exists(file_name))
        return

if __name__ == '__main__':
    unittest.main()
