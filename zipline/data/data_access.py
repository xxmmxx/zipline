#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class DataAccess(object):

    def __init__(self):
        self.last_sale = {}  # sid -> TradeEvent
        self.last_split_ratio = {}  # sid -> ratio

    def last_sale_prices(self, assets):
        """
        Returns np.array

        dt, is currently ignored, but will be required when last_sale is
        no longer used.
        """
        return np.array([self.last_sale[asset].price for asset in assets],
                        dtype=float)

    def last_sale_dates(self, assets):
        """
        Returns np.array

        dt, is currently ignored, but will be required when last_sale is
        no longer used.
        """
        return np.array([self.last_sale[asset].dt for asset in assets],
                        dtype=float)
