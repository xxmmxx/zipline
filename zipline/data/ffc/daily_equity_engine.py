import shelve

import pandas as pd

import numpy as np

from zipline.data.ffc.daily_equity_loader import DailyEquityLoader

if __name__ == "__main__":
    import zipline.finance.trading
    env = zipline.finance.trading.TradingEnvironment.instance()

    # use trading calendar instead
    min_date = pd.Timestamp('2002-01-02', tz='UTC')
    td = env.trading_days

    mask = (td > '2014-01-01') & (td < '2014-12-31')

    dates = env.trading_days[mask]

    # generated from start_pos.keys()[0:8000]
    d = shelve.open('test_sids.shelf')
    assets = np.array(d['test_sids'], dtype=np.uint32)

    loader = DailyEquityLoader("./equity_daily_bars.bcolz",
                               "./daily_equity_index.shelf",
                               trading_days=td[td >= min_date])

    result = loader.load_adjusted_array(
        ['close', 'volume'],
        dates,
        assets,
    )

    assert True
