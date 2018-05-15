from timebox.utils.validation import ensure_int
from timebox.utils.exceptions import NotIntegerException
import unittest


class TestValidation(unittest.TestCase):
    def test_ensure_int(self):
        self.assertEqual(5, ensure_int(5))
        self.assertEqual(5, ensure_int(float(5.0)))
        with self.assertRaises(NotIntegerException):
            ensure_int(0.5)
        with self.assertRaises(NotIntegerException):
            ensure_int(None)
        with self.assertRaises(NotIntegerException):
            ensure_int([])
        with self.assertRaises(NotIntegerException):
            ensure_int('error')
        return

if __name__ == '__main__':
    unittest.main()
