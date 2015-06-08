import shelve

import pandas as pd

import bcolz

import numpy as np
cimport numpy as np

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)

cpdef _load_adjusted_array_from_bcolz(daily_bar_table, daily_bar_index,
                                      trading_days,
                                      columns,
                                      assets,
                                      dates):
    """
    Load each column from bcolsz table, @daily_bar_table.

    @daily_bar_index is an index of the start position and dates of each
    asset from the table.
    """
    nrows = dates.shape[0]
    ncols = len(assets)

    # Create return containers for each column.
    data_arrays = {}
    for col in columns:
        # Should we make the fill value a property of the Column?
        fill_value_map = {np.float32: np.nan,
                          np.uint32: 0}
        col_data = np.full(
            shape=(nrows, ncols),
            fill_value=fill_value_map[col.dtype],
            dtype=col.dtype)
        data_arrays[col.name] = col_data

    cdef dict start_pos = daily_bar_index['start_pos']
    cdef dict start_day_offset = daily_bar_index['start_day_offset']

    cdef np.intp_t date_offset = trading_days.searchsorted(dates[0])
    cdef np.intp_t date_len = dates.shape[0]

    cdef np.intp_t start, end
    cdef np.intp_t i

    cdef np.ndarray[dtype=np.uint8_t, ndim=1] mask = np.zeros(
        daily_bar_table.len, dtype=np.uint8)

    asset_indices = []
    for asset in assets:
        start = start_pos[asset] - \
                start_day_offset[asset] + \
                date_offset
        # what if negative?
        # or handle case goes over
        # may need end_day_offset
        end = start + date_len
        asset_indices.append((start, end))

    for col in columns:
        data_col = daily_bar_table[col.name]
        is_float = col.dtype == np.float32
        col_array = data_arrays[col.name]
        for i, asset_ix in enumerate(asset_indices):
            asset_data = data_col[asset_ix[0]:asset_ix[1]]

            if is_float:
                # Use int for nan check for better precision.
                where_nan = asset_data[asset_data == 0]
                # Data is stored as np.uint32 of equity pricing x 1000
                asset_data = asset_data * 0.001
                asset_data[where_nan] = np.nan

            col_array[:, i] = asset_data
        del data_col

    return[
        adjusted_array(
            data_arrays[col.name],
            NOMASK,
            {})
        for col in columns]
