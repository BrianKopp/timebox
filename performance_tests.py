import numpy as np
import pandas as pd
import os
from timebox.timebox import TimeBox
from time import time


reference_file = 'timebox/tests/ETH-USD_combined.csv'

start = time()
df = pd.read_csv(reference_file, index_col=0)
time_to_read_csv = time() - start
print('Reading csv from pandas: {}'.format(time_to_read_csv))

copy_of_reference_file = 'timebox/tests/ETH-USD_combined_copy.csv'
start = time()
df.to_csv(copy_of_reference_file)
time_to_write_csv = time() - start
print('Writing csv from pandas: {}'.format(time_to_write_csv))

os.remove(copy_of_reference_file)

timebox_file_name = 'timebox/tests/test_timebox_io.npb'
start = time()
TimeBox.save_pandas(df, timebox_file_name)
time_to_process_df_and_save_timebox = time() - start
print('Processing pandas and saving timebox: {}'.format(time_to_process_df_and_save_timebox))

tb = TimeBox(timebox_file_name)
start = time()
tb.read()
time_to_read_timebox = time() - start
print('Read-time on timebox: {}'.format(time_to_read_timebox))

start = time()
df2 = tb.to_pandas()
time_to_convert_to_pandas = time() - start
print('Convert to pandas time: {}'.format(time_to_convert_to_pandas))

new_tb = TimeBox(timebox_file_name)
start = time()
typed_df = new_tb.to_pandas()
time_to_read_and_convert_to_pandas = time() - start
print('Read and convert to pandas: {}'.format(time_to_read_and_convert_to_pandas))

start = time()
TimeBox.save_pandas(typed_df, timebox_file_name)
time_to_write_typed_data_frame = time() - start
print('Time to write already typed data frame: {}'.format(time_to_write_typed_data_frame))

os.remove(timebox_file_name)

# compare to pickle
pickle_name = 'timebox/tests/test_pickle.pk'
start = time()
df.to_pickle(pickle_name)
time_to_write_pickle = time() - start
print('Pickle write time: {}'.format(time_to_write_pickle))

start = time()
pd.read_pickle(pickle_name)
time_to_read_pickle = time() - start
print('Pickle read time: {}'.format(time_to_read_pickle))

os.remove(pickle_name)
