from ..binary import determine_required_bytes_unsigned_integer
from ..exceptions import (
    IntegerLargerThan64BitsException,
    IntegerNotUnsignedException,
    NotIntegerException
)
import unittest


class TestBinaryUtils(unittest.TestCase):
    def test_read_unsigned_int(self):
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

if __name__ == '__main__':
    unittest.main()
