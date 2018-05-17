import numpy as np
import unittest
from timebox.timebox_tag import TimeBoxTag


class TestTimeBoxTagCompression(unittest.TestCase):
    def test_timebox_tag_compression(self):
        t = TimeBoxTag(0, 8, 'u')
        t.use_compression = True
        t.data = np.array([1000000, 1000001, 1000002, 1000005], np.uint64)
        t.encode_data()
        self.assertEqual('m', t._compression_mode)
        self.assertEqual('u', t._compressed_type_char)
        self.assertEqual(1, t._compressed_bytes_per_value)
        self.assertEqual(1000000, t._compression_reference_value)
        self.assertEqual(np.uint64, t._compression_reference_value_dtype)
        self.assertEqual(0, t._encoded_data[0])
        self.assertEqual(1, t._encoded_data[1])
        self.assertEqual(2, t._encoded_data[2])
        self.assertEqual(5, t._encoded_data[3])
        return

    def test_timebox_tag_decompression(self):
        t = TimeBoxTag(0, 8, 'u')
        t.use_compression = True
        t._encoded_data = np.array([0, 1, 2, 5], np.uint8)
        t._compression_mode = 'm'
        t._compressed_bytes_per_value = 1
        t._compressed_type_char = 'u'
        t._compression_reference_value = 1000000

        t._decode_data()
        self.assertEqual(4, t.data.size)
        self.assertEqual(8, t.data.itemsize)
        self.assertEqual(1000000, t.data[0])
        self.assertEqual(1000001, t.data[1])
        self.assertEqual(1000002, t.data[2])
        self.assertEqual(1000005, t.data[3])
        return

    def test_timebox_floating_point_rounding(self):
        t = TimeBoxTag(0, 8, 'f')
        t.use_compression = True
        t.floating_point_rounded = True
        t.num_decimals_to_store = 2
        t.data = np.array([0.5, -0.5, 10.2345, 0], np.float64)
        t.encode_data()
        self.assertEqual('m', t._compression_mode)
        self.assertEqual('u', t._compressed_type_char)
        self.assertEqual(2, t._compressed_bytes_per_value)
        self.assertEqual(-50, t._compression_reference_value)
        self.assertEqual(np.int64, t._compression_reference_value_dtype)
        self.assertEqual(100, t._encoded_data[0])
        self.assertEqual(0, t._encoded_data[1])
        self.assertEqual(1023 + 50, t._encoded_data[2])
        self.assertEqual(50, t._encoded_data[3])

        t._decode_data()
        self.assertEqual(np.float64, t.data.dtype)
        self.assertEqual(0.5, t.data[0])
        self.assertEqual(-0.5, t.data[1])
        self.assertEqual(10.23, t.data[2])
        self.assertEqual(0, t.data[3])

        return

if __name__ == '__main__':
    unittest.main()
