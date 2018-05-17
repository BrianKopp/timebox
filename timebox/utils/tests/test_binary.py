from timebox.utils.binary import determine_required_bytes_unsigned_integer, read_unsigned_int, \
    determine_required_bytes_signed_integer
from timebox.utils.exceptions import (
    IntegerLargerThan64BitsException,
    IntegerNotUnsignedException,
    NotIntegerException
)
import unittest


class TestBinaryUtils(unittest.TestCase):
    def test_determine_byte_requirements(self):
        with self.assertRaises(IntegerNotUnsignedException):
            determine_required_bytes_unsigned_integer(-1)
        with self.assertRaises(NotIntegerException):
            determine_required_bytes_unsigned_integer(None)
        with self.assertRaises(NotIntegerException):
            determine_required_bytes_unsigned_integer([])

        self.assertEqual(1, determine_required_bytes_unsigned_integer(0))
        self.assertEqual(1, determine_required_bytes_unsigned_integer(1))
        self.assertEqual(1, determine_required_bytes_unsigned_integer(2))
        self.assertEqual(1, determine_required_bytes_unsigned_integer(3))
        self.assertEqual(1, determine_required_bytes_unsigned_integer(255))
        self.assertEqual(2, determine_required_bytes_unsigned_integer(256))
        self.assertEqual(2, determine_required_bytes_unsigned_integer(65535))
        self.assertEqual(4, determine_required_bytes_unsigned_integer(65536))
        self.assertEqual(4, determine_required_bytes_unsigned_integer(4294967295))
        self.assertEqual(8, determine_required_bytes_unsigned_integer(4294967296))
        self.assertEqual(8, determine_required_bytes_unsigned_integer(18446744073709551615))
        with self.assertRaises(IntegerLargerThan64BitsException):
            determine_required_bytes_unsigned_integer(18446744073709551616)
        return

    def test_signed_int_bytes(self):
        with self.assertRaises(NotIntegerException):
            determine_required_bytes_signed_integer(None)
        with self.assertRaises(NotIntegerException):
            determine_required_bytes_signed_integer([])

        self.assertEqual(1, determine_required_bytes_signed_integer(0))
        self.assertEqual(1, determine_required_bytes_signed_integer(1))
        self.assertEqual(1, determine_required_bytes_signed_integer(2))
        self.assertEqual(1, determine_required_bytes_signed_integer(3))
        self.assertEqual(1, determine_required_bytes_signed_integer(-1))
        self.assertEqual(1, determine_required_bytes_signed_integer(-2))
        self.assertEqual(1, determine_required_bytes_signed_integer(-3))
        self.assertEqual(1, determine_required_bytes_signed_integer(127))
        self.assertEqual(1, determine_required_bytes_signed_integer(-128))

        self.assertEqual(2, determine_required_bytes_signed_integer(128))
        self.assertEqual(2, determine_required_bytes_signed_integer(-129))
        self.assertEqual(2, determine_required_bytes_signed_integer(32767))
        self.assertEqual(2, determine_required_bytes_signed_integer(-32768))

        self.assertEqual(4, determine_required_bytes_signed_integer(32768))
        self.assertEqual(4, determine_required_bytes_signed_integer(-32769))
        self.assertEqual(4, determine_required_bytes_signed_integer(2147483647))
        self.assertEqual(4, determine_required_bytes_signed_integer(-2147483648))

        self.assertEqual(8, determine_required_bytes_signed_integer(2147483648))
        self.assertEqual(8, determine_required_bytes_signed_integer(-2147483649))

        with self.assertRaises(IntegerLargerThan64BitsException):
            determine_required_bytes_signed_integer(9223372036854775808)
        return

    def test_read_unsigned_int(self):
        self.assertEqual(0, read_unsigned_int(b'\x00'))
        self.assertEqual(0, read_unsigned_int(b'\x00\x00'))
        self.assertEqual(0, read_unsigned_int(b'\x00\x00\x00'))
        self.assertEqual(0, read_unsigned_int(b'\x00\x00\x00\x00'))
        self.assertEqual(1, read_unsigned_int(b'\x01\x00'))
        self.assertEqual(255, read_unsigned_int(bytes([255])))
        self.assertEqual(256, read_unsigned_int(b'\x00\x01'))
        return


if __name__ == '__main__':
    unittest.main()
