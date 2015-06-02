import shelve

import pandas as pd

import bcolz

import numpy as np
cimport numpy as np

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)
from zipline.data.baseloader import DataLoader


COLUMN_TYPES = {
    'open': np.float64,
    'high': np.float64,
    'low': np.float64,
    'close': np.float64,
    'volume': np.uint32,
}


class DailyEquityLoader(DataLoader):

    def __init__(self, daily_bar_path, daily_index_path, trading_days):
        self.daily_bar_table = bcolz.open(daily_bar_path, mode='r')
        self.daily_bar_index = shelve.open(daily_index_path)
        self.trading_days = trading_days

    def load_adjusted_array(self, columns, dates, assets):
        """
        Load each column with self.make_column.
        """
        nrows = len(dates)
        ncols = len(assets)

        data_arrays = {}
        for col in columns:
            data_arrays[col] = np.ndarray(
                shape=(nrows, ncols),
                dtype=COLUMN_TYPES[col])
            if data_arrays[col].dtype == np.float64:
                data_arrays[col][:] = np.nan

        start_pos = self.daily_bar_index['start_pos']
        start_day_offset = self.daily_bar_index['start_day_offset']

        date_offset = self.trading_days.searchsorted(dates[0])
        date_len = dates.shape[0]

        asset_indices = []

        for i, asset in enumerate(assets):
            start = start_pos[asset] - start_day_offset[asset] + \
                date_offset
            # what if negative?
            # or handle case goes over
            # may need end_day_offset
            end = start + date_len
            asset_indices.append(slice(start, end, 1))

        for col in columns:
            data_col = self.daily_bar_table[col][:]
            for i, asset_slice in enumerate(asset_indices):

                asset_data = data_col[asset_slice]

                if col != 'volume':
                    # Use int for check for better precision.
                    where_nan = asset_data[asset_data == 0]
                    asset_data = asset_data * 0.001
                    asset_data[where_nan] = np.nan

                data_arrays[col][:, i] = asset_data
            del data_col

        return[
            adjusted_array(
                data_arrays[col],
                NOMASK,
                {})
        for col in columns]
