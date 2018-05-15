import unittest
import numpy as np
from timebox.utils.pandas_utils import parse_pandas_dtype
from timebox.utils.exceptions import InvalidPandasDataTypeError


class TestPandasUtils(unittest.TestCase):
    def test_parse_pandas_dtype(self):
        size, type_char = parse_pandas_dtype(np.float64)
        self.assertEqual(8, size)
        self.assertEqual('f', type_char)

        size, type_char = parse_pandas_dtype(np.float32)
        self.assertEqual(4, size)
        self.assertEqual('f', type_char)

        size, type_char = parse_pandas_dtype(np.float16)
        self.assertEqual(2, size)
        self.assertEqual('f', type_char)

        size, type_char = parse_pandas_dtype(np.uint64)
        self.assertEqual(8, size)
        self.assertEqual('u', type_char)

        size, type_char = parse_pandas_dtype(np.uint32)
        self.assertEqual(4, size)
        self.assertEqual('u', type_char)

        size, type_char = parse_pandas_dtype(np.uint16)
        self.assertEqual(2, size)
        self.assertEqual('u', type_char)

        size, type_char = parse_pandas_dtype(np.uint8)
        self.assertEqual(1, size)
        self.assertEqual('u', type_char)

        size, type_char = parse_pandas_dtype(np.int64)
        self.assertEqual(8, size)
        self.assertEqual('i', type_char)

        size, type_char = parse_pandas_dtype(np.int32)
        self.assertEqual(4, size)
        self.assertEqual('i', type_char)

        size, type_char = parse_pandas_dtype(np.int16)
        self.assertEqual(2, size)
        self.assertEqual('i', type_char)

        size, type_char = parse_pandas_dtype(np.int8)
        self.assertEqual(1, size)
        self.assertEqual('i', type_char)
        return

    def test_parse_pandas_dtype_errors(self):
        with self.assertRaises(InvalidPandasDataTypeError):
            parse_pandas_dtype('really not a dtype')
        with self.assertRaises(InvalidPandasDataTypeError):
            parse_pandas_dtype('datetime64[s]')

if __name__ == '__main__':
    unittest.main()
