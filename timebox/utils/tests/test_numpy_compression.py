from timebox.utils.exceptions import *
from timebox.utils.numpy_utils import compress_array
import unittest
import numpy as np


class TestNumpyCompression(unittest.TestCase):
    def test_compress_array_errors(self):
        data = np.array([1, -2, 3, 4], dtype=np.int32)
        with self.assertRaises(CompressionModeInvalidError):
            compress_array(data, 'bad_mode')
        return

    def test_compress_data_zero(self):
        data = np.array([1, 2, 3, 4], dtype=np.uint32)
        compression_result = compress_array(data, 'e')
        c_arr = compression_result.numpy_array
        self.assertEqual(1, compression_result.reference_value)
        self.assertEqual(1, c_arr.itemsize)
        self.assertEqual(3, c_arr.size)
        self.assertEqual(1, c_arr[0])
        self.assertEqual(1, c_arr[1])
        self.assertEqual(1, c_arr[2])

        compression_result = compress_array(data, 'm')
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
        data = np.array([-4, -2, 0, 2000], dtype=np.int16)
        compression_result = compress_array(data, 'e')
        c_arr = compression_result.numpy_array
        self.assertEqual(-4, compression_result.reference_value)
        self.assertEqual(2, c_arr.itemsize)
        self.assertEqual(3, c_arr.size)
        self.assertEqual(2, c_arr[0])
        self.assertEqual(2, c_arr[1])
        self.assertEqual(2000, c_arr[2])

        compression_result = compress_array(data, 'm')
        c_arr = compression_result.numpy_array
        self.assertEqual(-4, compression_result.reference_value)
        self.assertEqual(2, c_arr.itemsize)
        self.assertEqual(4, c_arr.size)
        self.assertEqual(0, c_arr[0])
        self.assertEqual(2, c_arr[1])
        self.assertEqual(4, c_arr[2])
        self.assertEqual(2004, c_arr[3])
        return

    def test_compress_data_two(self):
        data = np.array([5.2, 0.8, 3.1415, 8], dtype=np.float64)
        compression_result = compress_array(data, 'm')
        c_arr = compression_result.numpy_array
        self.assertEqual(0.8, compression_result.reference_value)
        self.assertEqual(8, c_arr.itemsize)
        self.assertEqual(4, c_arr.size)
        self.assertEqual(4.4, c_arr[0])
        self.assertEqual(0, c_arr[1])
        self.assertEqual(2.3415, c_arr[2])
        self.assertEqual(7.2, c_arr[3])
        return

    def test_compress_data_three(self):
        data = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                         4096, 8192, 16384, 32768, 65536], dtype=np.float64)
        compression_result = compress_array(data, 'e')
        c_arr = compression_result.numpy_array
        self.assertEqual(2, compression_result.reference_value)
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

        compression_result = compress_array(data, 'm')
        c_arr = compression_result.numpy_array
        self.assertEqual(2, compression_result.reference_value)
        self.assertEqual(4, c_arr.itemsize)
        self.assertEqual(16, c_arr.size)
        self.assertEqual(0, c_arr[0])
        self.assertEqual(2, c_arr[1])
        self.assertEqual(6, c_arr[2])
        self.assertEqual(65536 - 2, c_arr[15])
        return

    def test_compress_data_four(self):
        data = np.array([10, -2, 0, -2000], dtype=np.int16)
        compression_result = compress_array(data, 'e')
        c_arr = compression_result.numpy_array
        self.assertEqual(10, compression_result.reference_value)
        self.assertEqual(2, c_arr.itemsize)
        self.assertEqual(3, c_arr.size)
        self.assertEqual(-12, c_arr[0])
        self.assertEqual(2, c_arr[1])
        self.assertEqual(-2000, c_arr[2])

        data = np.array([10, -2, 0, 2000], dtype=np.int16)
        compression_result = compress_array(data, 'm')
        c_arr = compression_result.numpy_array
        self.assertEqual(-2, compression_result.reference_value)
        self.assertEqual(2, c_arr.itemsize)
        self.assertEqual(4, c_arr.size)
        self.assertEqual(12, c_arr[0])
        self.assertEqual(0, c_arr[1])
        self.assertEqual(2, c_arr[2])
        self.assertEqual(2002, c_arr[3])
        return

    def test_compress_tiny_arrays(self):
        self.assertEqual(1, compress_array(np.array([1], dtype=np.uint8), 'm').itemsize)
        self.assertEqual(1, compress_array(np.array([1], dtype=np.int8), 'm').itemsize)
        self.assertEqual(2, compress_array(np.array([1], dtype=np.float16), 'm').itemsize)
        self.assertEqual(2, compress_array(np.array([1], dtype=np.uint16), 'e').itemsize)
        return

    def test_unable_to_compress_type(self):
        with self.assertRaises(CompressionError):
            compress_array(np.array(['x'], dtype='U1'), 'e')
        return

if __name__ == '__main__':
    unittest.main()
