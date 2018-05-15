import numpy as np
import unittest
from timebox.utils.datetime_utils import *
from timebox.utils.exceptions import DateUnitsError, DateUnitsGranularityError


class TestDateTimeUtils(unittest.TestCase):
    def test_units_dict(self):
        self.assertEqual(7, len(units))
        self.assertEqual(7, len(units_by_order))
        return

    def test_get_unit_data_errors(self):
        with self.assertRaises(DateUnitsError):
            get_unit_data(-1)
        with self.assertRaises(DateUnitsError):
            get_unit_data(7)
        with self.assertRaises(DateUnitsError):
            get_unit_data(5.5)
        with self.assertRaises(DateUnitsError):
            get_unit_data([])
        with self.assertRaises(DateUnitsError):
            get_unit_data(None)
        with self.assertRaises(DateUnitsError):
            get_unit_data('not a unit')
        return

    def test_get_unit_data_successes(self):
        self.assertEqual('s', get_unit_data(SECONDS).units)
        self.assertEqual('s', get_unit_data('s').units)
        return

    def test_get_more_less_granular_units(self):
        self.assertEqual('us', get_less_granular_units('ns'))
        self.assertEqual('h', get_less_granular_units('m'))
        self.assertEqual('us', get_more_granular_units('ms'))
        self.assertEqual('m', get_more_granular_units('h'))
        with self.assertRaises(DateUnitsGranularityError):
            get_less_granular_units('D')
        with self.assertRaises(DateUnitsGranularityError):
            get_more_granular_units('ns')
        with self.assertRaises(DateUnitsError):
            get_less_granular_units('not a unit')
        with self.assertRaises(DateUnitsError):
            get_more_granular_units('not a unit')
        return

    def test_get_conversion_multiplier(self):
        self.assertEqual(1, get_conversion_multiplier('s', 's'))
        self.assertEqual(60, get_conversion_multiplier('m', 's'))
        self.assertEqual(0.001, get_conversion_multiplier('ms', 's'))
        with self.assertRaises(DateUnitsError):
            get_conversion_multiplier('error', 'not units')
        return

    def test_get_units_from_dtype(self):
        d = np.datetime64('2018-01-01', 's')
        self.assertEqual('s', get_units_from_dtype(d.dtype))
        d = np.datetime64('2018-01-01', 'ms')
        self.assertEqual('ms', get_units_from_dtype(d.dtype))
        d = np.float64(5)
        with self.assertRaises(DateUnitsError):
            get_units_from_dtype(d.dtype)
        with self.assertRaises(DateUnitsError):
            get_units_from_dtype('datetime64')
        with self.assertRaises(DateUnitsError):
            get_units_from_dtype('datetime64[')
        with self.assertRaises(DateUnitsError):
            get_units_from_dtype('datetime64]')
        with self.assertRaises(DateUnitsError):
            get_units_from_dtype('datetime64[error_not_units]')
        return

    def test_compress_time_delta_array(self):
        seconds_array = np.array(
            [
                np.datetime64('2018-01-01', 'D'),
                np.datetime64('2018-01-01T01:00', 'h'),
                np.datetime64('2018-01-01T01:30', 'm'),
                np.datetime64('2018-01-01T01:30:30', 's')
            ],
            dtype=np.datetime64
        )
        diff_array = np.ediff1d(seconds_array)
        self.assertEqual('s', get_units_from_dtype(diff_array.dtype))
        comp_array_result = compress_time_delta_array(diff_array)
        self.assertEqual('s', comp_array_result[1])
        self.assertEqual(3600, comp_array_result[0][0])

        compressible_array = np.array(
            [
                np.datetime64('2018-01-01', 's'),
                np.datetime64('2018-01-02', 'h'),
                np.datetime64('2018-01-03', 'm'),
                np.datetime64('2018-01-04', 's')
            ],
            dtype=np.datetime64
        )
        self.assertEqual('s', get_units_from_dtype(compressible_array.dtype))
        diff_array = np.ediff1d(compressible_array)
        self.assertEqual('s', get_units_from_dtype(diff_array.dtype))
        comp_array_result = compress_time_delta_array(diff_array)
        self.assertEqual('D', comp_array_result[1])
        self.assertEqual(1, comp_array_result[0][0])
        return

if __name__ == '__main__':
    unittest.main()
