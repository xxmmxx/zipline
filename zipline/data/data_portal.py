import bcolz
import os

FINDATA_DIR = os.getenv("FINDATA_DIR")


class DataPortal(object):

    def __init__(self, algo):
        self.views = {}
        self.algo = algo
        self.current_bcolz_handle = None

    def get_current_price_data(self, asset, column):
        dt = self.algo.datetime
        path = "{0}/{1}/{2}/{3}_equity-minutes.bcolz".format(
            FINDATA_DIR,
            str(dt.year),
            str(dt.month).zfill(2),
            str(dt.date()))
        table = bcolz.ctable(rootdir=path, mode='r')
        query = '(sid == {0}) & (dt == {1})'.format(
            int(asset),
            dt.value / 10e8)
        return table[query][column] * 0.001 * 0.5

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
