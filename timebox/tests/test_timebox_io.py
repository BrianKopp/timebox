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


class TestTimeBoxReadWrite(unittest.TestCase):
    def tearDown(self):
        try:
            os.remove('test_io.npb.lock')
        except OSError:
            pass
        return

    def test_read_write_data(self):
        file_name = 'test_io.npb'
        tb = example_time_box(file_name)
        tb._MAX_WRITE_BLOCK_WAIT_SECONDS = 0.1
        tb.write()
        self.assertTrue(os.path.isfile(file_name))
        self.assertFalse(os.path.isdir(tb._blocking_file_name()))

        # ensure I can get an exclusive lock now
        with open(file_name, 'rb') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            fcntl.flock(f, fcntl.LOCK_UN)

        # ensure read file is same content as written file
        tb_read = example_time_box(file_name)
        tb_read.read()
        self.assertFalse(os.path.isfile(tb_read._blocking_file_name()))

        self.assertEqual(tb._timebox_version, tb_read._timebox_version)
        self.assertEqual(tb._tag_names_are_strings, tb_read._tag_names_are_strings)
        self.assertEqual(tb._date_differentials_stored, tb_read._date_differentials_stored)
        self.assertEqual(tb._num_points, tb_read._num_points)
        self.assertEqual(tb._start_date, tb_read._start_date)
        self.assertEqual(tb._bytes_per_date_differential, tb_read._bytes_per_date_differential)
        self.assertEqual(tb._date_differential_units, tb_read._date_differential_units)
        for t in tb._tag_definitions:
            self.assertTrue(t in tb_read._tag_definitions)
            self.assertEqual(tb._tag_definitions[t].identifier, tb_read._tag_definitions[t].identifier)
            self.assertEqual(tb._tag_definitions[t].type_char, tb_read._tag_definitions[t].type_char)
            self.assertEqual(tb._tag_definitions[t].dtype, tb_read._tag_definitions[t].dtype)
            self.assertEqual(tb._tag_definitions[t].bytes_per_value, tb_read._tag_definitions[t].bytes_per_value)

        for t in tb._data:
            for i in range(0, tb._num_points):
                self.assertEqual(tb._data[t][i], tb_read._data[t][i])

        os.remove(file_name)
        return

    def test_prevent_write_lock(self):
        file_name = 'test_io.npb'
        tb = example_time_box(file_name)
        block_file = tb._blocking_file_name()

        # put the blocking file and then watch how it cannot write
        open(block_file, 'w').close()
        tb._MAX_WRITE_BLOCK_WAIT_SECONDS = 0.1
        tb._MAX_READ_BLOCK_WAIT_SECONDS = 0.1
        with self.assertRaises(CouldNotAcquireFileLockError):
            tb.write()
        self.assertTrue(os.path.exists(block_file))  # this is mine!
        os.remove(block_file)

        tb.write()

        # test needs to wait for shared to unblock
        with open(file_name, 'rb') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            with self.assertRaises(CouldNotAcquireFileLockError):
                tb.write()
            fcntl.flock(f, fcntl.LOCK_UN)
        self.assertFalse(os.path.exists(block_file))

        os.remove(file_name)
        return

    def test_prevent_read_lock(self):
        file_name = 'test_io.npb'
        tb = example_time_box(file_name)
        tb._MAX_WRITE_BLOCK_WAIT_SECONDS = 0.1
        tb.write()
        block_file = tb._blocking_file_name()
        self.assertFalse(os.path.exists(block_file))

        # put the blocking file and then watch how it cannot read
        open(block_file, 'w').close()
        tb._MAX_READ_BLOCK_WAIT_SECONDS = 0.1
        tb._MAX_WRITE_BLOCK_WAIT_SECONDS = 0.1
        with self.assertRaises(CouldNotAcquireFileLockError):
            tb.read()

        os.remove(block_file)
        self.assertFalse(os.path.exists(block_file))

        # now put an exclusive lock, but no block file
        with open(file_name, 'rb') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            with self.assertRaises(CouldNotAcquireFileLockError):
                tb.read()
            fcntl.flock(f, fcntl.LOCK_UN)

        self.assertFalse(os.path.exists(block_file))
        os.remove(file_name)
        return

    def test_can_read_with_other_readers(self):
        file_name = 'test_io.npb'
        tb = example_time_box(file_name)
        tb._MAX_READ_BLOCK_WAIT_SECONDS = 0.1
        tb._MAX_WRITE_BLOCK_WAIT_SECONDS = 0.1
        tb.write()
        block_file = tb._blocking_file_name()
        self.assertFalse(os.path.exists(block_file))
        with open(file_name, 'rb') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            tb.read()  # no issues
            fcntl.flock(f, fcntl.LOCK_UN)

        self.assertFalse(os.path.exists(block_file))
        os.remove(file_name)
        return

    def test_bad_arguments(self):
        tb = example_time_box('')
        with self.assertRaises(ValueError):
            tb._get_fcntl_lock('x')

if __name__ == '__main__':
    unittest.main()
