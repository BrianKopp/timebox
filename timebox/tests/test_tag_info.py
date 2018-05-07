from ..tag_info import TagInfo
import numpy as np
import unittest


class TestTagInfo(unittest.TestCase):
    def test_tag_info_init(self):
        tag_info = TagInfo('my_id', 4, 'f')
        self.assertEqual('my_id', tag_info.identifier)
        self.assertEqual(4, tag_info.bytes_per_value)
        self.assertEqual('f', tag_info.type_char)
        self.assertEqual(np.float32, tag_info.dtype)
        tag_info = TagInfo('my_id', 4, ord('f'))
        self.assertEqual(np.float32, tag_info.dtype)
        return

if __name__ == '__main__':
    unittest.main()
