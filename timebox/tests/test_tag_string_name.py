from ..timebox import TimeBox
from ..exceptions import *
from ..tag_info import TagInfo
import unittest
import numpy as np
import os
import fcntl


def example_time_box(file_name: str):
    tb = TimeBox(file_name)
    tb._timebox_version = 1
    tb._tag_names_are_strings = True
    tb._date_differentials_stored = False
    tb._num_points = 4
    tb._tag_definitions = {
        'tag_0': TagInfo('tag_0', 1, 'u'),
        'tag_1': TagInfo('tag_1', 2, 'i'),
        'tag_2_long_name': TagInfo('tag_2_long_name', 4, 'f')
    }
    tb._start_date = np.datetime64('2018-01-01', 's')
    tb._seconds_between_points = 3600
    tb._data = {
        'tag_0': np.array([1, 2, 3, 4], dtype=np.uint8),
        'tag_1': np.array([-4, -2, 0, 2000], dtype=np.int16),
        'tag_2_long_name': np.array([5.2, 0.8, 3.1415, 8], dtype=np.float32)
    }
    return tb


class TestTimeBoxTagStringName(unittest.TestCase):
    def test_read_write_data_with_tag_name_as_string(self):
        file_name = 'test_io.npb'
        tb = example_time_box(file_name)
        tb._update_required_bytes_for_tag_identifier()
        self.assertEqual(15*4, tb._num_bytes_for_tag_identifier)
        tb.write()

        tb_read = TimeBox(file_name)
        tb_read.read()
        self.assertTrue(tb_read._tag_names_are_strings)
        self.assertEqual(3, len(tb_read._tag_definitions))
        self.assertEqual(3, len(tb_read._data))
        self.assertTrue('tag_0' in tb_read._tag_definitions)
        self.assertTrue('tag_1' in tb_read._tag_definitions)
        self.assertTrue('tag_2_long_name' in tb_read._tag_definitions)
        self.assertTrue('tag_0' in tb_read._data)
        self.assertTrue('tag_1' in tb_read._data)
        self.assertTrue('tag_2_long_name' in tb_read._data)

        os.remove(file_name)
        return


if __name__ == '__main__':
    unittest.main()
