from ..timebox import TimeBox
from ..tag_info import TagInfo
from ..exceptions import *
import unittest
import numpy as np
import os


def example_time_box(tb_file_name: str):
    tb = TimeBox(tb_file_name)
    tb._timebox_version = 1
    tb._tag_names_are_strings = False
    tb._date_differentials_stored = False
    tb._num_points = 4
    tb._tag_definitions = {
        0: TagInfo(0, 1, 'u'),
        1: TagInfo(1, 2, 'i'),
        2: TagInfo(2, 4, 'f')
    }
    tb._start_date = np.datetime64('2018-01-01', 's')
    tb._seconds_between_points = 3600
    tb._data = {
        0: np.array([1, 2, 3, 4], dtype=np.uint32),
        1: np.array([-4, -2, 0, 2000], dtype=np.int16),
        2: np.array([5.2, 0.8, 3.1415, 8], dtype=np.float64),
        3: np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                     4096, 8192, 16384, 32768, 65536], dtype=np.float64)
    }
    return tb


file_name = 'test_tags_compressed.npb'


class TestTimeBoxDataCompression(unittest.TestCase):
    def test_compress_array_errors(self):
        tb = example_time_box(file_name)
        with self.assertRaises(CompressionModeInvalidError):
            TimeBox._compress_array(tb._data[0], 'bad_mode')
        with self.assertRaises(CompressionError):
            TimeBox._compress_array(tb._data[2], 'e')  # negative derivative
        return

    def test_compress_data_zero(self):
        tb = example_time_box(file_name)
        compression_result = TimeBox._compress_array(tb._data[0], 'e')
        c_arr = compression_result[0]
        self.assertEqual(1, compression_result[1])
        self.assertEqual(1, c_arr.itemsize)
        self.assertEqual(3, c_arr.size)
        self.assertEqual(1, c_arr[0])
        self.assertEqual(1, c_arr[1])
        self.assertEqual(1, c_arr[2])

        compression_result = TimeBox._compress_array(tb._data[0], 'm')
        c_arr = compression_result[0]
        self.assertEqual(1, compression_result[1])
        self.assertEqual(1, c_arr.itemsize)
        self.assertEqual(4, c_arr.size)
        self.assertEqual(0, c_arr[0])
        self.assertEqual(1, c_arr[1])
        self.assertEqual(2, c_arr[2])
        self.assertEqual(3, c_arr[3])
        return

    def test_compress_data_one(self):
        tb = example_time_box(file_name)
        compression_result = TimeBox._compress_array(tb._data[1], 'e')
        c_arr = compression_result[0]
        self.assertEqual(-4, compression_result[1])
        self.assertEqual(2, c_arr.itemsize)
        self.assertEqual(3, c_arr.size)
        self.assertEqual(2, c_arr[0])
        self.assertEqual(2, c_arr[1])
        self.assertEqual(2000, c_arr[2])

        compression_result = TimeBox._compress_array(tb._data[1], 'm')
        c_arr = compression_result[0]
        self.assertEqual(-4, compression_result[1])
        self.assertEqual(2, c_arr.itemsize)
        self.assertEqual(4, c_arr.size)
        self.assertEqual(0, c_arr[0])
        self.assertEqual(2, c_arr[1])
        self.assertEqual(4, c_arr[2])
        self.assertEqual(2004, c_arr[3])
        return

    def test_compress_data_two(self):
        tb = example_time_box(file_name)
        compression_result = TimeBox._compress_array(tb._data[2], 'm')
        c_arr = compression_result[0]
        self.assertEqual(0.8, compression_result[1])
        self.assertEqual(8, c_arr.itemsize)
        self.assertEqual(4, c_arr.size)
        self.assertEqual(4.4, c_arr[0])
        self.assertEqual(0, c_arr[1])
        self.assertEqual(2.3415, c_arr[2])
        self.assertEqual(7.2, c_arr[3])
        return

    def test_compress_data_three(self):
        tb = example_time_box(file_name)
        compression_result = TimeBox._compress_array(tb._data[3], 'e')
        c_arr = compression_result[0]
        self.assertEqual(2, compression_result[1])
        self.assertEqual(2, c_arr.itemsize)
        self.assertEqual(15, c_arr.size)
        self.assertEqual(2, c_arr[0])
        self.assertEqual(4, c_arr[1])
        self.assertEqual(8, c_arr[2])
        self.assertEqual(16, c_arr[3])
        self.assertEqual(32, c_arr[4])
        self.assertEqual(64, c_arr[5])
        self.assertEqual(128, c_arr[6])
        self.assertEqual(256, c_arr[7])
        self.assertEqual(512, c_arr[8])
        self.assertEqual(1024, c_arr[9])
        self.assertEqual(2048, c_arr[10])
        self.assertEqual(4096, c_arr[11])
        self.assertEqual(8192, c_arr[12])
        self.assertEqual(16384, c_arr[13])
        self.assertEqual(32768, c_arr[14])

        compression_result = TimeBox._compress_array(tb._data[3], 'm')
        c_arr = compression_result[0]
        self.assertEqual(2, compression_result[1])
        self.assertEqual(4, c_arr.itemsize)
        self.assertEqual(16, c_arr.size)
        self.assertEqual(0, c_arr[0])
        self.assertEqual(2, c_arr[1])
        self.assertEqual(6, c_arr[2])
        self.assertEqual(65536 - 2, c_arr[15])
        return

    def test_compress_tiny_arrays(self):
        self.assertEqual(1, TimeBox._compress_array(np.array([1], dtype=np.uint8), 'm').itemsize)
        self.assertEqual(1, TimeBox._compress_array(np.array([1], dtype=np.int8), 'm').itemsize)
        self.assertEqual(2, TimeBox._compress_array(np.array([1], dtype=np.float16), 'm').itemsize)

        self.assertEqual(2, TimeBox._compress_array(np.array([1], dtype=np.uint16), 'e').itemsize)
        return

if __name__ == '__main__':
    unittest.main()
