import shelve

import pandas as pd

import bcolz

import numpy as np
cimport numpy as np

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)


COLUMN_TYPES = {
    'open': np.float64,
    'high': np.float64,
    'low': np.float64,
    'close': np.float64,
    'volume': np.uint32,
}


cdef class DailyEquityLoader:

    cdef object daily_bar_table, daily_bar_index, trading_days


    def __cinit__(self, daily_bar_path, daily_index_path, trading_days):
        self.daily_bar_table = bcolz.open(daily_bar_path, mode='r')
        self.daily_bar_index = shelve.open(daily_index_path)
        self.trading_days = trading_days

    def load_adjusted_array(self, columns, dates, np.uint32_t[:] assets):
        """
        Load each column with self.make_column.
        """
        nrows = dates.shape[0]
        ncols = len(assets)

        data_arrays = {}
        for col in columns:
            dtype = COLUMN_TYPES[col]
            if dtype == np.float64:
                col_data = np.ndarray(
                    shape=(nrows, ncols),
                    dtype=dtype)
                col_data[:] = np.nan
                data_arrays[col] = col_data
            elif dtype == np.uint32:
                col_data = np.zeros(
                    shape=(nrows, ncols),
                    dtype=dtype)
                data_arrays[col] = col_data

        cdef dict start_pos = self.daily_bar_index['start_pos']
        cdef dict start_day_offset = self.daily_bar_index[
            'start_day_offset']

        cdef np.intp_t date_offset = self.trading_days.searchsorted(dates[0])
        cdef np.intp_t date_len = dates.shape[0]

        cdef np.intp_t start, end
        cdef np.intp_t i

        cdef np.ndarray[dtype=np.uint8_t, ndim=1] mask = np.zeros(
            self.daily_bar_table.len, dtype=np.uint8)

        for asset in assets:
            start = start_pos[asset] - \
                    start_day_offset[asset] + \
                    date_offset
            # what if negative?
            # or handle case goes over
            # may need end_day_offset
            end = start + date_len
            for i in range(start, end):
                mask[i] = True

        print type(mask)
        for col in columns:
            data_col = self.daily_bar_table[col]
            asset_data = data_col[mask.view(dtype=np.bool)]

            if col != 'volume':
                # Use int for check for better precision.
                where_nan = asset_data[asset_data == 0]
                asset_data = asset_data * 0.001
                asset_data[where_nan] = np.nan

#            data_arrays[col][:, 0] = asset_data
            del data_col

        return[
            adjusted_array(
                data_arrays[col],
                NOMASK,
                {})
        for col in columns]
