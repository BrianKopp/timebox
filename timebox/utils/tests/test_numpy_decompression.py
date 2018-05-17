from timebox.utils.exceptions import *
from timebox.utils.numpy_utils import compress_array, decompress_array
import unittest
import numpy as np


class TestNumpyDecompression(unittest.TestCase):
    def test_decompress_array_errors(self):
        data = np.array([1, -2, 3, 4], dtype=np.int32)
        with self.assertRaises(CompressionModeInvalidError):
            decompress_array(data, 'bad_mode', None)
        return

    def test_unable_to_decompress_type(self):
        with self.assertRaises(CompressionError):
            decompress_array(np.array(['x'], dtype='U1'), 'e', None)
        return

    def test_decompress_data_zero(self):
        data = np.array([1, 2, 3, 4], dtype=np.uint32)
        compression_result = compress_array(data, 'e')
        dec_array = decompress_array(compression_result.numpy_array, 'e', compression_result.reference_value)
        self.assertEqual(4, dec_array.size)
        self.assertEqual(1, dec_array[0])
        self.assertEqual(2, dec_array[1])
        self.assertEqual(3, dec_array[2])
        self.assertEqual(4, dec_array[3])

        compression_result = compress_array(data, 'm')
        dec_array = decompress_array(compression_result.numpy_array, 'm', compression_result.reference_value)
        self.assertEqual(4, dec_array.size)
        self.assertEqual(1, dec_array[0])
        self.assertEqual(2, dec_array[1])
        self.assertEqual(3, dec_array[2])
        self.assertEqual(4, dec_array[3])
        return

    def test_decompress_data_one(self):
        data = np.array([-4, -2, 0, 2000], dtype=np.int16)
        compression_result = compress_array(data, 'e')
        dec_array = decompress_array(compression_result.numpy_array, 'e', compression_result.reference_value)
        self.assertEqual(4, dec_array.size)
        self.assertEqual(-4, dec_array[0])
        self.assertEqual(-2, dec_array[1])
        self.assertEqual(0, dec_array[2])
        self.assertEqual(2000, dec_array[3])

        compression_result = compress_array(data, 'm')
        dec_array = decompress_array(compression_result.numpy_array, 'm', compression_result.reference_value)
        self.assertEqual(4, dec_array.size)
        self.assertEqual(-4, dec_array[0])
        self.assertEqual(-2, dec_array[1])
        self.assertEqual(0, dec_array[2])
        self.assertEqual(2000, dec_array[3])
        return

    def test_compress_data_two(self):
        data = np.array([5.2, 0.8, 3.1415, 8], dtype=np.float64)
        compression_result = compress_array(data, 'm')
        dec_array = decompress_array(compression_result.numpy_array, 'm', compression_result.reference_value)
        self.assertEqual(4, dec_array.size)
        self.assertEqual(5.2, dec_array[0])
        self.assertEqual(0.8, dec_array[1])
        self.assertAlmostEqual(3.1415, dec_array[2])
        self.assertEqual(8, dec_array[3])
        return

    def test_compress_data_three(self):
        data = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
                         4096, 8192, 16384, 32768, 65536], dtype=np.uint64)
        compression_result = compress_array(data, 'e')
        dec_array = decompress_array(compression_result.numpy_array, 'e', compression_result.reference_value)
        self.assertEqual(16, dec_array.size)
        self.assertEqual(2, dec_array[0])
        self.assertEqual(4, dec_array[1])
        self.assertEqual(8, dec_array[2])
        self.assertEqual(16, dec_array[3])
        self.assertEqual(32, dec_array[4])
        self.assertEqual(64, dec_array[5])
        self.assertEqual(128, dec_array[6])
        self.assertEqual(256, dec_array[7])
        self.assertEqual(512, dec_array[8])
        self.assertEqual(1024, dec_array[9])
        self.assertEqual(2048, dec_array[10])
        self.assertEqual(4096, dec_array[11])
        self.assertEqual(8192, dec_array[12])
        self.assertEqual(16384, dec_array[13])
        self.assertEqual(32768, dec_array[14])
        self.assertEqual(65536, dec_array[15])

        compression_result = compress_array(data, 'm')
        dec_array = decompress_array(compression_result.numpy_array, 'm', compression_result.reference_value)
        self.assertEqual(16, dec_array.size)
        self.assertEqual(2, dec_array[0])
        self.assertEqual(4, dec_array[1])
        self.assertEqual(8, dec_array[2])
        self.assertEqual(16, dec_array[3])
        self.assertEqual(32, dec_array[4])
        self.assertEqual(64, dec_array[5])
        self.assertEqual(128, dec_array[6])
        self.assertEqual(256, dec_array[7])
        self.assertEqual(512, dec_array[8])
        self.assertEqual(1024, dec_array[9])
        self.assertEqual(2048, dec_array[10])
        self.assertEqual(4096, dec_array[11])
        self.assertEqual(8192, dec_array[12])
        self.assertEqual(16384, dec_array[13])
        self.assertEqual(32768, dec_array[14])
        self.assertEqual(65536, dec_array[15])
        return

    def test_compress_tiny_arrays(self):
        self.assertEqual(1, compress_array(np.array([1], dtype=np.uint8), 'm').itemsize)
        self.assertEqual(1, compress_array(np.array([1], dtype=np.int8), 'm').itemsize)
        self.assertEqual(2, compress_array(np.array([1], dtype=np.float16), 'm').itemsize)
        self.assertEqual(2, compress_array(np.array([1], dtype=np.uint16), 'e').itemsize)
        return

if __name__ == '__main__':
    unittest.main()
