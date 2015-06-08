import shelve

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
    print assets

    loader = USEquityPricingLoader("./equity_daily_bars.bcolz",
                                   "./daily_equity_index.shelf",
                                   trading_days=td[td >= min_date])

    result = loader.load_adjusted_array(
        [USEquityPricing.close, USEquityPricing.volume],
        dates,
        assets,
    )

    print result[0]
    print result[1]
    import pprint; import nose; nose.tools.set_trace()
