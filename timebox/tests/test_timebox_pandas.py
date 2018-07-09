from timebox.timebox import TimeBox
from timebox.utils.exceptions import InvalidPandasIndexError
import pandas as pd
import numpy as np
import unittest
import os
import logging


class TestTimeBoxPandas(unittest.TestCase):
    def test_save_pandas(self):
        file_name = 'save_pandas.npb'
        df = pd.read_csv('timebox/tests/data/ETH-USD_combined_utc.csv', index_col=0)
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

    def test_io_pandas(self):
        file_name = 'save_pandas.npb'
        df = pd.read_csv('timebox/tests/data/test1.csv').set_index('date')
        logging.debug('Starting test_io_pandas with df\n{}'.format(df))
        tb = TimeBox.save_pandas(df, file_name)
        tb_read = TimeBox(file_name)
        df2 = tb_read.to_pandas()
        self.assertListEqual(list(df.columns.sort_values()), list(df2.columns.sort_values()))

        df = df.sort_index()
        # ensure index is same
        for i in range(0, len(df.index)):
            self.assertEqual(pd.to_datetime(df.index[i]), pd.to_datetime(df2.index[i]))

        # ensure each value is the same
        columns = df.columns
        for c in columns:
            logging.debug('Testing column: {}'.format(c))
            logging.debug('Original frame:{}'.format(df[c]))
            logging.debug('TB frame:{}'.format(df2[c]))
            self.assertEqual(df[c].sum(), df2[c].sum())

        os.remove(file_name)
        return

if __name__ == '__main__':
    unittest.main()
