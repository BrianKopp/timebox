import unittest
from timebox.utils.numpy_utils import *


class TestNumpyUtils(unittest.TestCase):
    def test_get_type_char_int(self):
        self.assertEqual(ord('i'), get_type_char_int('i'))
        self.assertNotEqual(ord('i'), get_type_char_int('u'))
        self.assertNotEqual(ord('i'), get_type_char_int('f'))
        self.assertEqual(ord('u'), get_type_char_int('u'))
        self.assertNotEqual(ord('u'), get_type_char_int('i'))
        self.assertNotEqual(ord('u'), get_type_char_int('f'))
        self.assertEqual(ord('f'), get_type_char_int('f'))
        self.assertNotEqual(ord('f'), get_type_char_int('i'))
        self.assertNotEqual(ord('f'), get_type_char_int('u'))

        self.assertEqual(ord('i'), get_type_char_int(ord('i')))
        self.assertEqual(ord('u'), get_type_char_int(ord('u')))
        self.assertEqual(ord('f'), get_type_char_int(ord('f')))

        self.assertEqual(ord('i'), get_type_char_int(float(ord('i'))))

        with self.assertRaises(CharConversionException):
            get_type_char_int([])
        with self.assertRaises(CharConversionException):
            get_type_char_int({})
        with self.assertRaises(CharConversionException):
            get_type_char_int(0.5)
        with self.assertRaises(CharConversionException):
            get_type_char_int(None)
        with self.assertRaises(TypeError):
            get_type_char_int('error')
        return

    def test_get_type_char_char(self):
        self.assertEqual('i', get_type_char_char(ord('i')))
        self.assertNotEqual('i', get_type_char_char(ord('u')))
        self.assertNotEqual('i', get_type_char_char(ord('f')))
        self.assertEqual('u', get_type_char_char(ord('u')))
        self.assertNotEqual('u', get_type_char_char(ord('i')))
        self.assertNotEqual('u', get_type_char_char(ord('f')))
        self.assertEqual('f', get_type_char_char(ord('f')))
        self.assertNotEqual('f', get_type_char_char(ord('i')))
        self.assertNotEqual('f', get_type_char_char(ord('u')))

        self.assertEqual('i', get_type_char_char('i'))
        self.assertEqual('u', get_type_char_char('u'))
        self.assertEqual('f', get_type_char_char('f'))

        self.assertEqual('i', get_type_char_char(float(ord('i'))))

        with self.assertRaises(CharConversionException):
            get_type_char_char([])
        with self.assertRaises(CharConversionException):
            get_type_char_char({})
        with self.assertRaises(ValueError):
            get_type_char_char(0.5)
        with self.assertRaises(CharConversionException):
            get_type_char_char(None)
        with self.assertRaises(CharConversionException):
            get_type_char_char('error')
        return

    def test_get_numpy_type(self):
        self.assertEqual(np.int8, get_numpy_type('i', 8))
        self.assertEqual(np.int16, get_numpy_type('i', 16))
        self.assertEqual(np.int32, get_numpy_type('i', 32))
        self.assertEqual(np.int64, get_numpy_type('i', 64))
        self.assertEqual(np.uint8, get_numpy_type('u', 8))
        self.assertEqual(np.uint16, get_numpy_type('u', 16))
        self.assertEqual(np.uint32, get_numpy_type('u', 32))
        self.assertEqual(np.uint64, get_numpy_type('u', 64))
        self.assertEqual(np.float16, get_numpy_type('f', 16))
        self.assertEqual(np.float32, get_numpy_type('f', 32))
        self.assertEqual(np.float64, get_numpy_type('f', 64))
        with self.assertRaises(ValueError):
            get_numpy_type('i', 10)
        with self.assertRaises(ValueError):
            get_numpy_type('u', 15)
        with self.assertRaises(ValueError):
            get_numpy_type('f', 30)
        with self.assertRaises(ValueError):
            get_numpy_type('a', 8)
        return

if __name__ == '__main__':
    unittest.main()
