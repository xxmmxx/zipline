import shelve
import pandas as pd

import bcolz
cimport cython
import numpy as np
cimport numpy as np

from zipline.data.adjusted_array import (
    adjusted_array,
    NOMASK,
)

@cython.boundscheck(False)
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

    cdef dict start_pos = daily_bar_index['start_pos']
    cdef dict start_day_offset = daily_bar_index['start_day_offset']
    cdef dict end_day_offset = daily_bar_index['end_day_offset']

    cdef np.intp_t date_offset = trading_days.searchsorted(dates[0])
    cdef np.intp_t date_len = dates.shape[0]

    cdef np.intp_t start, end
    cdef np.intp_t i

    asset_indices = []

    cdef np.intp_t asset_start, asset_start_day_offset, asset_end_day_offset

    for asset in assets:
        asset_start = start_pos[asset]
        asset_start_day_offset = start_day_offset[asset]
        asset_end_day_offset = end_day_offset[asset]
        start = asset_start - \
                asset_start_day_offset + \
                date_offset
        end = min(start + date_len, asset_start + asset_end_day_offset)
        asset_indices.append((start, end))

    for col in columns:
        data_col = daily_bar_table[col.name][:]
        col_array = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
        for i, asset_ix in enumerate(asset_indices):
            asset_data = data_col[asset_ix[0]:asset_ix[1]]

            # Asset data may not necessarily be the same shape as the number
            # of dates if the asset has an earlier end date.
            col_array[0:asset_data.shape[0], i] = asset_data

        if col.dtype == np.float32:
            # Use int for nan check for better precision.
            where_nan = col_array == 0
            col_array = col_array.astype(np.float32) * 0.001
            col_array[where_nan] = np.nan

        data_arrays[col.name] = col_array
        del data_col

    return[
        adjusted_array(
            data_arrays[col.name],
            NOMASK,
            {})
        for col in columns]
