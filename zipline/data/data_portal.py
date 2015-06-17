class DataPortal(object):

    def __init__(self, algo):
        self.views = {}
        self.algo = algo

    def get_current_price_data(self, asset, column):
        return 50.0

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
