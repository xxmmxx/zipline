import shelve
import time

import pandas as pd

import numpy as np

from zipline.data.equities import USEquityPricing
from zipline.data.ffc.loaders.us_equity_pricing import \
    USEquityPricingLoader

if __name__ == "__main__":
    import zipline.finance.trading
    env = zipline.finance.trading.TradingEnvironment.instance()

    # use trading calendar instead
    min_date = pd.Timestamp('2002-01-02', tz='UTC')
    td = env.trading_days

    mask = (td > '2003-01-01') & (td < '2003-12-31')

    dates = env.trading_days[mask]

    # generated from start_pos.keys()[0:8000]
    d = shelve.open('./daily_equity_index.shelf')
    assets = np.array(sorted(d['start_pos'].keys())[0:8000], dtype=np.uint32)

    loader = USEquityPricingLoader("./equity_daily_bars.bcolz",
                                   "./daily_equity_index.shelf",
                                   trading_days=td[td >= min_date])

    before = time.time()
    result = loader.load_adjusted_array(
        [USEquityPricing.close, USEquityPricing.volume],
        dates,
        assets,
    )
    after = time.time()
    duration = after - before
    print "time in load_adjusted_array={0}".format(duration)

    print result[0]
    print result[1]
