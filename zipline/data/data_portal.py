import bcolz
import os

FINDATA_DIR = os.getenv("FINDATA_DIR")


class DataPortal(object):

    def __init__(self, algo):
        self.views = {}
        self.algo = algo
        self.current_bcolz_handle = None
        self.carrays = {
            'open': {},
            'high': {},
            'low': {},
            'close': {},
            'volume': {}
        }

    def get_current_price_data(self, asset, column):
        dt = self.algo.datetime
        path = "{0}/{1}/{2}/{3}_equity-minutes.bcolz".format(
            FINDATA_DIR,
            str(dt.year),
            str(dt.month).zfill(2),
            str(dt.date()))
        try:
            carray = self.carrays[column][asset]
        except KeyError:
            carray = self.carrays[column][asset] = bcolz.carray(
                rootdir=path + "/" + column, mode='r')
        first_row = 200
        dt_ix = first_row + dt.minute
        price = carray[dt_ix] * 0.001 * 0.5
        return price

    def get_equity_price_view(self, asset):
        try:
            view = self.views[asset]
        except KeyError:
            view = self.views[asset] = DataPortalSidView(asset, self)

        return view


class DataPortalSidView(object):

    def __init__(self, asset, portal):
        self.asset = asset
        self.portal = portal

    def __getattr__(self, column):
        return self.portal.get_current_price_data(self.asset, column)
