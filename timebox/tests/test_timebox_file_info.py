from ..timebox import TimeBox
from ..tag_info import TagInfo
import unittest
import numpy as np


class TestTimeBoxFileInfo(unittest.TestCase):
    def test_init(self):
        tb = TimeBox('test_file_path.txt')
        self.assertEqual('test_file_path.txt', tb._file_path)
        return

    def test_update_required_bytes(self):
        tb = TimeBox('')
        tb._tag_names_are_strings = False
        tb._tag_definitions[0] = TagInfo(0, 1, 'u')
        tb._update_required_bytes_for_tag_identifier()
        self.assertEqual(1, tb._num_bytes_for_tag_identifier)
        tb._tag_definitions[256] = TagInfo(256, 1, 'u')
        tb._update_required_bytes_for_tag_identifier()
        self.assertEqual(2, tb._num_bytes_for_tag_identifier)

        tb._tag_names_are_strings = True
        tb._tag_definitions = {
            'a': TagInfo('a', 1, 'u'),
            'ab': TagInfo('ab', 1, 'u'),
            'abc': TagInfo('abc', 1, 'u')
        }
        tb._update_required_bytes_for_tag_identifier()
        self.assertEqual(12, tb._num_bytes_for_tag_identifier)
        return

    def test_tag_definitions_to_from_bytes_integer(self):
        first = TimeBox('')
        first._tag_names_are_strings = False
        first._tag_definitions = {
            0: TagInfo(0, 1, 'u'),
            1: TagInfo(1, 2, 'i'),
            2: TagInfo(2, 4, 'f'),
            255: TagInfo(255, 8, 'i'),
            256: TagInfo(256, 8, 'f')
        }
        first._update_required_bytes_for_tag_identifier()
        first_bytes = first._tag_definitions_to_bytes()
        self.assertEqual(5 * (2 + 1 + 1), first_bytes[0])
        self.assertEqual(813, np.frombuffer(first_bytes[1], dtype=np.uint8).sum())

        second = TimeBox('')
        second._num_bytes_for_tag_identifier = 2
        second._tag_names_are_strings = False
        second._unpack_tag_definitions(first_bytes[1])
        second_bytes = second._tag_definitions_to_bytes()
        self.assertEqual(first_bytes[0], second_bytes[0])
        self.assertEqual(first_bytes[1], second_bytes[1])

        second._num_bytes_for_tag_identifier = 4
        bad_bytes = second._tag_definitions_to_bytes()
        self.assertNotEqual(second_bytes[0], bad_bytes[0])

        second._tag_names_are_strings = True
        bad_bytes = second._tag_definitions_to_bytes()
        self.assertNotEqual(second_bytes[0], bad_bytes[0])
        return

if __name__ == '__main__':
    unittest.main()
