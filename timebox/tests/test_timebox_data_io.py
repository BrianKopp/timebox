from ..exceptions import DataDoesNotMatchTagDefinitionError
from ..timebox import TimeBox
from ..tag_info import TagInfo
import unittest
import numpy as np
import os


def example_time_box(file_name: str):
    tb = TimeBox(file_name)
    tb._timebox_version = 1
    tb._tag_names_are_strings = False
    tb._date_differentials_stored = False
    tb._num_points = 4
    tb._tag_definitions = {
        0: TagInfo(0, 1, 'u'),
        1: TagInfo(1, 2, 'i'),
        2: TagInfo(2, 4, 'f')
    }
    tb._start_date = np.datetime64('2018-01-01', 's')
    tb._seconds_between_points = 3600
    tb._data = {
        0: np.array([1, 2, 3, 4], dtype=np.uint8),
        1: np.array([-4, -2, 0, 2000], dtype=np.int16),
        2: np.array([5.2, 0.8, 3.1415, 8], dtype=np.float32)
    }
    return tb


class TestTimeBoxTagReadWrite(unittest.TestCase):
    def test_read_write_tag_data(self):
        file_name = 'test_tags_io.npb'
        tb = example_time_box(file_name)
        with open(file_name, 'wb') as f:
            self.assertEqual(28, tb._write_tag_data(f))

        tb_read = example_time_box(file_name)
        with open(file_name, 'rb') as f:
            self.assertEqual(28, tb_read._read_tag_data(f))

        for t in tb._data:
            for i in range(0, tb._num_points):
                self.assertEqual(tb._data[t][i], tb_read._data[t][i])

        os.remove(file_name)
        return

    def test_validation_errors(self):
        tb = example_time_box('')
        tb._data = {}
        with self.assertRaises(DataDoesNotMatchTagDefinitionError):
            tb._validate_data_for_write()

        tb = example_time_box('')
        tb._data.pop(0, None)
        tb._data[1234567] = []
        with self.assertRaises(DataDoesNotMatchTagDefinitionError):
            tb._validate_data_for_write()

        tb = example_time_box('')
        tb._tag_definitions[0].dtype = None
        with self.assertRaises(DataDoesNotMatchTagDefinitionError):
            tb._validate_data_for_write()

        return

if __name__ == '__main__':
    unittest.main()