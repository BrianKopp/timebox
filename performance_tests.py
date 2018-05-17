import numpy as np
import pandas as pd
import os
from timebox.timebox import TimeBox
from time import time


def write_result(description, write_time, read_time, file_size):
    print('{:>40}|{:>8}|{:>8}|{}'.format(
        description,
        round(write_time, 3),
        round(read_time, 3),
        file_size
    ))
    return


print('{:>40}|{:>8}|{:>8}|{}'.format('Description', 'Write', 'Read', 'FileSize'))

reference_file = 'timebox/tests/data/ETH-USD_combined_utc.csv'

start = time()
df = pd.read_csv(reference_file, index_col=0)
time_to_read_csv = time() - start

copy_of_reference_file = 'timebox/tests/data/ETH-USD_combined_copy.csv'
start = time()
df.to_csv(copy_of_reference_file)
time_to_write_csv = time() - start

write_result('pandas csv', time_to_write_csv, time_to_read_csv, os.path.getsize(reference_file))
os.remove(copy_of_reference_file)

timebox_file_name = 'timebox/tests/data/test_timebox_io.npb'
start = time()
TimeBox.save_pandas(df, timebox_file_name)
time_to_process_df_and_save_timebox = time() - start

new_tb = TimeBox(timebox_file_name)
start = time()
typed_df = new_tb.to_pandas()
time_to_read_and_convert_to_pandas = time() - start

write_result(
    'file <-> timebox <-> pandas',
    time_to_process_df_and_save_timebox,
    time_to_read_and_convert_to_pandas,
    os.path.getsize(timebox_file_name)
)

tb = TimeBox(timebox_file_name)
start = time()
tb.read()
time_to_read_timebox = time() - start

start = time()
TimeBox.save_pandas(typed_df, timebox_file_name)
time_to_write_typed_data_frame = time() - start

write_result(
    'file <-> timebox',
    time_to_write_typed_data_frame,
    time_to_read_timebox,
    os.path.getsize(timebox_file_name)
)

#start = time()
#tb = TimeBox(timebox_file_name)
#df2 = tb.to_pandas()
#time_to_convert_to_pandas = time() - start

#write_result('timebox->pandas', 0, time_to_convert_to_pandas, os.path.getsize(timebox_file_name))

TimeBox.save_pandas(df, timebox_file_name)

tb_float_compress = TimeBox(timebox_file_name)
tb_float_compress.read()

for t in tb_float_compress._tags:
    tb_float_compress._tags[t].use_compression = True
    tb_float_compress._tags[t]._compression_mode = 'e'
    tb_float_compress._tags[t].floating_point_rounded = True
    tb_float_compress._tags[t].num_decimals_to_store = 2
tb_float_compress._tags['volume'].num_decimals_to_store = 6
start = time()
tb_float_compress.write()
time_to_write_compressed_and_rounded = time() - start

tb_float_rounded_read = TimeBox(timebox_file_name)
start = time()
tb_float_rounded_read.read()
time_to_read_compressed_and_rounded = time() - start

write_result(
    'rounded and compressed',
    time_to_write_compressed_and_rounded,
    time_to_read_compressed_and_rounded,
    os.path.getsize(timebox_file_name)
)


os.remove(timebox_file_name)

# compare to pickle
pickle_name = 'timebox/tests/data/test_pickle.pk'
start = time()
df.to_pickle(pickle_name)
time_to_write_pickle = time() - start

start = time()
pd.read_pickle(pickle_name)
time_to_read_pickle = time() - start

write_result('pickle', time_to_write_pickle, time_to_read_pickle, os.path.getsize(pickle_name))

os.remove(pickle_name)
