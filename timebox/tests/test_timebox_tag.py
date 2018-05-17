import numpy as np
import unittest
from timebox.timebox_tag import TimeBoxTag
from timebox.exceptions import TagIdentifierByteRepresentationError


class TestTimeBoxTag(unittest.TestCase):
    def test_tag_info_init(self):
        tag_info = TimeBoxTag('my_id', 4, 'f')
        self.assertEqual('my_id', tag_info.identifier)
        self.assertEqual(4, tag_info.bytes_per_value)
        self.assertEqual('f', tag_info.type_char)
        self.assertEqual(np.float32, tag_info.dtype)

        tag_info = TimeBoxTag('my_id', 4, ord('f'))
        self.assertEqual(np.float32, tag_info.dtype)

        self.assertEqual(None, tag_info.data)
        self.assertEqual(None, tag_info._encoded_data)
        self.assertEqual(None, tag_info.num_points)

        self.assertFalse(tag_info.use_compression)
        self.assertFalse(tag_info.use_hash_table)
        self.assertFalse(tag_info.floating_point_rounded)

        self.assertEqual(None, tag_info._compressed_type_char)
        self.assertEqual(None, tag_info._compressed_bytes_per_value)
        self.assertEqual(None, tag_info._compression_mode)
        self.assertEqual(None, tag_info._compression_reference_value)
        self.assertEqual(tag_info.dtype, tag_info._compression_reference_value_dtype)

        self.assertEqual(None, tag_info.num_decimals_to_store)
        self.assertEqual(0, tag_info.num_bytes_extra_information)

        tag_info = TimeBoxTag('my_id', 4, 'f', options=1)
        self.assertTrue(tag_info.use_compression)
        tag_info = TimeBoxTag('my_id', 4, 'f', options=3)
        self.assertTrue(tag_info.use_hash_table)
        tag_info = TimeBoxTag('my_id', 4, 'f', options=2)
        self.assertTrue(tag_info.use_hash_table)
        self.assertFalse(tag_info.use_compression)

        tag_info = TimeBoxTag('my_id', 4, 'f', options=0, untyped_bytes=b''.join([b'\x00' for _ in range(0, 32)]))
        return

    def test_get_tag_info_dtype(self):
        actual = TimeBoxTag.tag_info_dtype(4, True)
        self.assertEqual('tag_identifier', actual.descr[0][0])
        self.assertEqual('<U1', actual.descr[0][1])

        self.assertEqual('options', actual.descr[1][0])
        self.assertEqual('<u2', actual.descr[1][1])

        self.assertEqual('bytes_per_point', actual.descr[2][0])
        self.assertEqual('|u1', actual.descr[2][1])

        self.assertEqual('type_char', actual.descr[3][0])
        self.assertEqual('|u1', actual.descr[3][1])

        self.assertEqual('bytes_extra_information', actual.descr[4][0])
        self.assertEqual('<u4', actual.descr[4][1])

        for i in range(0, 32):
            self.assertEqual('def_byte_{}'.format(i + 1), actual.descr[5 + i][0])
            self.assertEqual('|u1', actual.descr[5 + i][1])

        actual = TimeBoxTag.tag_info_dtype(16, True)
        self.assertEqual('<U4', actual.descr[0][1])

        actual = TimeBoxTag.tag_info_dtype(32, True)
        self.assertEqual('<U8', actual.descr[0][1])

        actual = TimeBoxTag.tag_info_dtype(128, True)
        self.assertEqual('<U32', actual.descr[0][1])

        actual = TimeBoxTag.tag_info_dtype(1, False)
        self.assertEqual('|u1', actual.descr[0][1])

        actual = TimeBoxTag.tag_info_dtype(2, False)
        self.assertEqual('<u2', actual.descr[0][1])

        actual = TimeBoxTag.tag_info_dtype(4, False)
        self.assertEqual('<u4', actual.descr[0][1])

        actual = TimeBoxTag.tag_info_dtype(8, False)
        self.assertEqual('<u8', actual.descr[0][1])

        # test errors
        with self.assertRaises(TagIdentifierByteRepresentationError):
            TimeBoxTag.tag_info_dtype(2, True)
        with self.assertRaises(TagIdentifierByteRepresentationError):
            TimeBoxTag.tag_info_dtype(0, True)
        with self.assertRaises(TagIdentifierByteRepresentationError):
            TimeBoxTag.tag_info_dtype(-1, True)
        with self.assertRaises(ValueError):
            TimeBoxTag.tag_info_dtype(0.5, False)
        return

    def test_encode_decode_def_bytes_compression(self):
        t = TimeBoxTag(0, 8, 'u', options=0)
        encoded_bytes = t._encode_def_bytes()
        self.assertEqual(encoded_bytes, b''.join([b'\x00' for _ in range(0, 32)]))
        t.use_compression = True
        t._compression_mode = 'e'
        t._compressed_bytes_per_value = 2
        t._compressed_type_char = 'u'
        t._compression_reference_value = 5
        t._compression_reference_value_dtype = np.dtype(np.uint64)
        encoded_bytes = t._encode_def_bytes()
        self.assertEqual(101, encoded_bytes[0])
        self.assertEqual(2, encoded_bytes[1])
        self.assertEqual(117, encoded_bytes[2])
        self.assertEqual(8, encoded_bytes[3])
        self.assertEqual(117, encoded_bytes[4])
        self.assertEqual(5, encoded_bytes[5])

        t = TimeBoxTag(0, 8, 'u', options=1)
        t._decode_def_bytes(encoded_bytes)
        self.assertTrue(t.use_compression)
        self.assertEqual('e', t._compression_mode)
        self.assertEqual(2, t._compressed_bytes_per_value)
        self.assertEqual('u', t._compressed_type_char)
        self.assertEqual(5, t._compression_reference_value)
        return

    def test_encode_decode_def_bytes_floating_point_rounding(self):
        t = TimeBoxTag(0, 8, 'f', options=0)
        encoded_bytes = t._encode_def_bytes()
        self.assertEqual(encoded_bytes, b''.join([b'\x00' for _ in range(0, 32)]))
        t.floating_point_rounded = True
        t.num_decimals_to_store = 2
        encoded_bytes = t._encode_def_bytes()
        self.assertEqual(2, encoded_bytes[0])

        t = TimeBoxTag(0, 8, 'f', options=4)
        t._decode_def_bytes(encoded_bytes)
        self.assertTrue(t.floating_point_rounded)
        self.assertEqual(2, t.num_decimals_to_store)
        return

    def test_tag_to_bytes(self):
        t = TimeBoxTag(1, 8, 'u', options=0)
        t_byte_result = t.info_to_bytes(1, False)
        self.assertEqual(41, t_byte_result[0])
        t_bytes = t_byte_result.byte_code
        self.assertEqual(1, t_bytes[0])  # identifier
        self.assertEqual(0, t_bytes[1])  # options, byte 1
        self.assertEqual(0, t_bytes[2])  # options, byte 2
        self.assertEqual(8, t_bytes[3])  # bytes per value
        self.assertEqual(117, t_bytes[4])  # type char
        self.assertEqual(b'\x00\x00\x00\x00', t_bytes[5:9])  # num bytes extra info
        return

    def test_tag_options(self):
        t = TimeBoxTag(1, 8, 'u', options=0)

        t.use_compression = False
        t.use_hash_table = False
        t.floating_point_rounded = False
        self.assertEqual(0, t._encode_options())

        t.use_compression = True
        t.use_hash_table = False
        t.floating_point_rounded = False
        self.assertEqual(1, t._encode_options())

        t.use_compression = False
        t.use_hash_table = True
        t.floating_point_rounded = False
        self.assertEqual(2, t._encode_options())

        t.use_compression = True
        t.use_hash_table = True
        t.floating_point_rounded = False
        self.assertEqual(3, t._encode_options())

        t.use_compression = False
        t.use_hash_table = False
        t.floating_point_rounded = True
        self.assertEqual(4, t._encode_options())

        t.use_compression = True
        t.use_hash_table = False
        t.floating_point_rounded = True
        self.assertEqual(5, t._encode_options())

        t.use_compression = False
        t.use_hash_table = True
        t.floating_point_rounded = True
        self.assertEqual(6, t._encode_options())

        t.use_compression = True
        t.use_hash_table = True
        t.floating_point_rounded = True
        self.assertEqual(7, t._encode_options())

        t.use_compression = True
        t.use_hash_table = True
        t.floating_point_rounded = False
        self.assertEqual(3, t._encode_options())

        t._decode_options(0)
        self.assertFalse(t.use_compression)
        self.assertFalse(t.use_hash_table)
        self.assertFalse(t.floating_point_rounded)

        t._decode_options(1)
        self.assertTrue(t.use_compression)
        self.assertFalse(t.use_hash_table)
        self.assertFalse(t.floating_point_rounded)

        t._decode_options(2)
        self.assertFalse(t.use_compression)
        self.assertTrue(t.use_hash_table)
        self.assertFalse(t.floating_point_rounded)

        t._decode_options(3)
        self.assertTrue(t.use_compression)
        self.assertTrue(t.use_hash_table)
        self.assertFalse(t.floating_point_rounded)

        t._decode_options(4)
        self.assertFalse(t.use_compression)
        self.assertFalse(t.use_hash_table)
        self.assertTrue(t.floating_point_rounded)

        t._decode_options(5)
        self.assertTrue(t.use_compression)
        self.assertFalse(t.use_hash_table)
        self.assertTrue(t.floating_point_rounded)

        t._decode_options(6)
        self.assertFalse(t.use_compression)
        self.assertTrue(t.use_hash_table)
        self.assertTrue(t.floating_point_rounded)

        t._decode_options(7)
        self.assertTrue(t.use_compression)
        self.assertTrue(t.use_hash_table)
        self.assertTrue(t.floating_point_rounded)
        return

if __name__ == '__main__':
    unittest.main()
