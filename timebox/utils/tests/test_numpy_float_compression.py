import unittest
import numpy as np
from timebox.utils.numpy_utils import compress_float_array
from timebox.utils.exceptions import *


class TestNumpyFloatCompression(unittest.TestCase):
    def test_non_float_type(self):
        with self.assertRaises(ArrayNotFloatException):
            compress_float_array(np.array([1], dtype=np.uint8))
        return

    def test_float_16(self):
        compressed = compress_float_array(np.array([1], dtype=np.float16))
        self.assertEqual(2, compressed.itemsize)
        return

    def test_float_nan(self):
        nan_array = np.array([np.nan], dtype=np.float32)
        compressed = compress_float_array(nan_array)
        self.assertEqual(2, compressed.itemsize)

        nan_array = np.array([np.nan], dtype=np.float64)
        compressed = compress_float_array(nan_array)
        self.assertEqual(2, compressed.itemsize)
        return

    def test_zero_array(self):
        z_array = np.zeros(5, dtype=np.float32)
        compressed = compress_float_array(z_array)
        self.assertEqual(2, compressed.itemsize)

        z_array = np.array([np.nan], dtype=np.float64)
        compressed = compress_float_array(z_array)
        self.assertEqual(2, compressed.itemsize)
        return

    def test_cannot_compress(self):
        b_array = np.frombuffer(b'\x0c\x0c\x0c\x0c\x0c\x0c\x0c\x0c', dtype=np.float64)
        compressed = compress_float_array(b_array)
        self.assertEqual(8, compressed.itemsize)

        b_array = np.frombuffer(b'\x0c\x0c\x0c\x0c', dtype=np.float32)
        compressed = compress_float_array(b_array)
        self.assertEqual(4, compressed.itemsize)

        b_array = np.array([np.finfo('d').max, np.finfo('d').min], dtype=np.float64)
        compressed = compress_float_array(b_array)
        self.assertEqual(8, compressed.itemsize)

        b_array = np.array([np.finfo(np.float32).max, np.finfo(np.float32).min], dtype=np.float32)
        compressed = compress_float_array(b_array)
        self.assertEqual(4, compressed.itemsize)
        return

    def test_can_compress(self):
        b_array = np.array([np.finfo(np.float32).max, np.finfo(np.float32).min], dtype=np.float64)
        compressed = compress_float_array(b_array)
        self.assertEqual(4, compressed.itemsize)

        b_array = np.array([np.finfo(np.float16).max, np.finfo(np.float16).min], dtype=np.float64)
        compressed = compress_float_array(b_array)
        self.assertEqual(2, compressed.itemsize)
        return

    def test_significant_bits(self):
        # this binary number gives me a 1024 in the exponent,
        # a 1 in all the mantissa bits out to the 29th least significant mantissa bit
        arr = np.frombuffer(b'\x00\x00\x00\xf0\xff\xff\x0f@', dtype=np.float64)
        arr = compress_float_array(arr)
        self.assertEqual(8, arr.itemsize)

        # now move the bit one over
        arr = np.frombuffer(b'\x00\x00\x00\xe0\xff\xff\x0f@', dtype=np.float64)
        arr = compress_float_array(arr)
        self.assertEqual(4, arr.itemsize)
        self.assertEqual(b'\xff\xff\x7f@', arr.tobytes())

        # now test 32-bit down to 16-bit.
        arr = np.frombuffer(b'\x00p\x7f@', dtype=np.float32)
        arr = compress_float_array(arr)
        self.assertEqual(4, arr.itemsize)

        # now move bit one over
        arr = np.frombuffer(b'\x00`\x7f@', dtype=np.float32)
        arr = compress_float_array(arr)
        self.assertEqual(2, arr.itemsize)
        return

    def test_least_significant_bits_prevent_compression(self):
        arr = np.frombuffer(b'\x00\xff\x00\x00\xff\xff\x0f@', dtype=np.float64)
        arr = compress_float_array(arr)
        self.assertEqual(8, arr.itemsize)

        arr = np.frombuffer(b'\x01\x00\x7f@', dtype=np.float32)
        arr = compress_float_array(arr)
        self.assertEqual(4, arr.itemsize)
        return

    def test_exponent_prevent_compression(self):
        arr = np.frombuffer(b'\x00\x00\x00\x00\xff\xff\x8fL', dtype=np.float64)
        arr = compress_float_array(arr)
        self.assertEqual(8, arr.itemsize)

        arr = np.frombuffer(b'\x00\x00\x7f\x7f', dtype=np.float32)
        arr = compress_float_array(arr)
        self.assertEqual(4, arr.itemsize)
        return

    def test_float_array_compresses(self):
        arr = np.array([2, 4, 8, 16, 32, 64, 128], dtype=np.float64)
        arr = compress_float_array(arr)
        self.assertEqual(2, arr.itemsize)

if __name__ == '__main__':
    unittest.main()
