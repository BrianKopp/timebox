from ..timebox import TimeBox
from ..utils.exceptions import InvalidPandasIndexError
from ..utils.datetime_utils import get_unit_data
import pandas as pd
import numpy as np
import unittest
import os


class TestTimeBoxPandas(unittest.TestCase):
    def test_save_pandas(self):
        file_name = 'save_pandas.npb'
        df = pd.read_csv('timebox/tests/ETH-USD_combined.csv', index_col=0)
        tb = TimeBox.save_pandas(df, file_name)
        self.assertTrue(os.path.exists(file_name))

        tb_read = TimeBox(file_name)
        df2 = tb_read.to_pandas()

        df_columns = list(df)
        df_columns.sort()
        df2_columns = list(df2)
        df2_columns.sort()

        self.assertListEqual(df_columns, df2_columns)
        os.remove(file_name)
        return

    def test_pandas_errors(self):
        df = pd.DataFrame.from_dict(
            {
                'value_1': np.array([0, 1, 2], dtype=np.uint8)
            },
            orient='columns'
        )
        with self.assertRaises(InvalidPandasIndexError):
            TimeBox.save_pandas(df, 'not_going_to_save.npb')
        return

if __name__ == '__main__':
    unittest.main()
