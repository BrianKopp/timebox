from ..timebox import *
from ..exceptions import TagIdentifierByteRepresentationError
import unittest


class TestTimeBoxStatics(unittest.TestCase):
    def test_read_unsigned_int(self):
        self.assertEqual(0, TimeBox._read_unsigned_int(b'\x00'))
        self.assertEqual(0, TimeBox._read_unsigned_int(b'\x00\x00'))
        self.assertEqual(0, TimeBox._read_unsigned_int(b'\x00\x00\x00'))
        self.assertEqual(0, TimeBox._read_unsigned_int(b'\x00\x00\x00\x00'))
        self.assertEqual(1, TimeBox._read_unsigned_int(b'\x01\x00'))
        self.assertEqual(255, TimeBox._read_unsigned_int(bytes([255])))
        self.assertEqual(256, TimeBox._read_unsigned_int(b'\x00\x01'))
        return

    def test_get_tag_info_dtype(self):
        actual = TimeBox._get_tag_info_dtype(4, True)
        self.assertEqual('tag_identifier', actual.descr[0][0])
        self.assertEqual('bytes_per_point', actual.descr[1][0])
        self.assertEqual('type_char', actual.descr[2][0])
        self.assertEqual('<U1', actual.descr[0][1])
        self.assertEqual('|u1', actual.descr[1][1])
        self.assertEqual('|u1', actual.descr[2][1])

        actual = TimeBox._get_tag_info_dtype(16, True)
        self.assertEqual('<U4', actual.descr[0][1])

        actual = TimeBox._get_tag_info_dtype(32, True)
        self.assertEqual('<U8', actual.descr[0][1])

        actual = TimeBox._get_tag_info_dtype(128, True)
        self.assertEqual('<U32', actual.descr[0][1])

        actual = TimeBox._get_tag_info_dtype(1, False)
        self.assertEqual('|u1', actual.descr[0][1])

        actual = TimeBox._get_tag_info_dtype(2, False)
        self.assertEqual('<u2', actual.descr[0][1])

        actual = TimeBox._get_tag_info_dtype(4, False)
        self.assertEqual('<u4', actual.descr[0][1])

        actual = TimeBox._get_tag_info_dtype(8, False)
        self.assertEqual('<u8', actual.descr[0][1])

        # test errors
        with self.assertRaises(TagIdentifierByteRepresentationError):
            TimeBox._get_tag_info_dtype(2, True)
        with self.assertRaises(TagIdentifierByteRepresentationError):
            TimeBox._get_tag_info_dtype(0, True)
        with self.assertRaises(TagIdentifierByteRepresentationError):
            TimeBox._get_tag_info_dtype(-1, True)
        with self.assertRaises(ValueError):
            TimeBox._get_tag_info_dtype(0.5, False)
        return

if __name__ == '__main__':
    unittest.main()
