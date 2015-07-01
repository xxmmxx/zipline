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

from abc import ABCMeta
from numbers import Integral
import numpy as np
import operator
import warnings
from functools32 import lru_cache

from logbook import Logger
import pandas as pd
from pandas.tseries.tools import normalize_date
from six import with_metaclass, string_types

from zipline.errors import (
    ConsumeAssetMetaDataError,
    InvalidAssetType,
    MultipleSymbolsFound,
    RootSymbolNotFound,
    SidAssignmentError,
    SidNotFound,
    SymbolNotFound,
    MapAssetIdentifierIndexError,
)
from zipline.assets._assets import (
    Asset, Equity, Future
)

log = Logger('assets.py')

# Expected fields for an Asset's metadata
ASSET_FIELDS = [
    'sid',
    'asset_type',
    'symbol',
    'root_symbol',
    'asset_name',
    'start_date',
    'end_date',
    'first_traded',
    'exchange',
    'notice_date',
    'expiration_date',
    'contract_multiplier',
    # The following fields are for compatibility with other systems
    'file_name',  # Used as symbol
    'company_name',  # Used as asset_name
    'start_date_nano',  # Used as start_date
    'end_date_nano',  # Used as end_date
]


def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


class AssetSQLWriter(object):

    def __init__(self, metadata=None, allow_sid_assignment=False,
                 fuzzy_char='_'):
        # This flag controls if the AssetFinder is allowed to generate its own
        # sids. If False, metadata that does not contain a sid will raise an
        # exception when building assets.
        self.allow_sid_assignment = allow_sid_assignment

        # The AssetFinder also holds a nested-dict of all metadata for
        # reference when building Assets
        self.metadata_cache = {}
        if metadata is not None:
            self.consume_metadata(metadata)

        self.fuzzy_char = fuzzy_char

    def write_sql(self, conn):
        # Should metadata_cache be renamed?
        equities_metadata = []
        futures_metadata = []
        for identifier, row in self.metadata_cache.iteritems():
            asset = self._spawn_asset(identifier=identifier, **row)
            if isinstance(asset, Equity):
                equities_metadata.append(asset.to_dict())
            elif isinstance(asset, Future):
                futures_metadata.append(asset.to_dict())
        if equities_metadata:
            equities_df = pd.DataFrame(equities_metadata).set_index('sid')

            try:
                equities_df['start_date'] = equities_df['start_date'].\
                    astype('datetime64[ns]').\
                    astype(np.int64)
            except KeyError:
                pass
            try:
                equities_df['end_date'] = equities_df['end_date'].\
                    astype('datetime64[ns]').\
                    astype(np.int64)
            except KeyError:
                pass

            equities_df['fuzzy'] = equities_df['symbol'].apply(
                lambda x: x.replace(self.fuzzy_char, ''))

            equities_df.to_sql('equities', conn)

        if futures_metadata:
            futures_df = pd.DataFrame(futures_metadata).set_index('sid')
            try:
                futures_df['start_date'] = futures_df['start_date'].\
                    astype('datetime64[ns]').\
                    astype(np.int64)
            except KeyError:
                pass
            try:
                futures_df['end_date'] = futures_df['end_date'].\
                    astype('datetime64[ns]').\
                    astype(np.int64)
            except KeyError:
                pass
            try:
                futures_df['expiration_date'] = futures_df['expiration_date'].\
                    astype('datetime64[ns]').\
                    astype(np.int64)
            except KeyError:
                pass

            futures_df.to_sql('futures', conn)

    def _next_free_sid(self):
        if len(self.cache) > 0:
            return max(self.cache.keys()) + 1
        return 0

    def _assign_sid(self, identifier):
        if hasattr(identifier, '__int__'):
            return identifier.__int__()
        if not self.allow_sid_assignment:
            raise SidAssignmentError(identifier=identifier)
        if isinstance(identifier, string_types):
            return self._next_free_sid()

    def insert_metadata(self, identifier, **kwargs):
        """
        Inserts the given metadata kwargs to the entry for the given
        identifier. Matching fields in the existing entry will be overwritten.
        :param identifier: The identifier for which to insert metadata
        :param kwargs: The keyed metadata to insert
        """
        entry = self.metadata_cache.get(identifier, {})

        for key, value in kwargs.items():
            # Do not accept invalid fields
            if key not in ASSET_FIELDS:
                continue
            # Do not accept Nones
            if value is None:
                continue
            # Do not accept empty strings
            if value == '':
                continue
            # Do not accept nans from dataframes
            if isinstance(value, float) and np.isnan(value):
                continue
            entry[key] = value

        self.metadata_cache[identifier] = entry

    def consume_identifiers(self, identifiers):
        """
        Consumes the given identifiers in to the metadata cache of this
        AssetFinder.
        """
        for identifier in identifiers:
            # Handle case where full Assets are passed in
            # For example, in the creation of a DataFrameSource, the source's
            # 'sid' args may be full Assets
            if isinstance(identifier, Asset):
                sid = identifier.sid
                metadata = identifier.to_dict()
                metadata['asset_type'] = identifier.__class__.__name__
                self.insert_metadata(identifier=sid, **metadata)
            else:
                self.insert_metadata(identifier)

    def consume_metadata(self, metadata):
        """
        Consumes the provided metadata in to the metadata cache. The
        existing values in the cache will be overwritten when there
        is a conflict.
        :param metadata: The metadata to be consumed
        """
        # Handle dicts
        if isinstance(metadata, dict):
            self._insert_metadata_dict(metadata)
        # Handle DataFrames
        elif isinstance(metadata, pd.DataFrame):
            self._insert_metadata_dataframe(metadata)
        # Handle readables
        elif hasattr(metadata, 'read'):
            self._insert_metadata_readable(metadata)
        else:
            raise ConsumeAssetMetaDataError(obj=metadata)

    def _insert_metadata_dataframe(self, dataframe):
        for identifier, row in dataframe.iterrows():
            self.insert_metadata(identifier, **row)

    def _insert_metadata_dict(self, dict):
        for identifier, entry in dict.items():
            self.insert_metadata(identifier, **entry)

    def _insert_metadata_readable(self, readable):
        for row in readable.read():
            # Parse out the row of the readable object
            metadata_dict = {}
            for field in ASSET_FIELDS:
                try:
                    row_value = row[field]
                    # Avoid passing placeholders
                    if row_value and (row_value is not 'None'):
                        metadata_dict[field] = row[field]
                except KeyError:
                    continue
                except IndexError:
                    continue
            # Locate the identifier, fail if not found
            if 'sid' in metadata_dict:
                identifier = metadata_dict['sid']
            elif 'symbol' in metadata_dict:
                identifier = metadata_dict['symbol']
            else:
                raise ConsumeAssetMetaDataError(obj=row)
            self.insert_metadata(identifier, **metadata_dict)

    def _spawn_asset(self, identifier, **kwargs):

        # Check if the sid is declared
        try:
            kwargs['sid']
            pass
        except KeyError:
            # If the identifier is not a sid, assign one
            kwargs['sid'] = self._assign_sid(identifier)
            # Update the metadata object with the new sid
            self.insert_metadata(identifier=identifier, sid=kwargs['sid'])

        # If the file_name is in the kwargs, it will be used as the symbol
        try:
            kwargs['symbol'] = kwargs.pop('file_name')
        except KeyError:
            pass

        # If the identifier coming in was a string and there is no defined
        # symbol yet, set the symbol to the incoming identifier
        try:
            kwargs['symbol']
            pass
        except KeyError:
            if isinstance(identifier, string_types):
                kwargs['symbol'] = identifier

        # If the company_name is in the kwargs, it may be the asset_name
        try:
            company_name = kwargs.pop('company_name')
            try:
                kwargs['asset_name']
            except KeyError:
                kwargs['asset_name'] = company_name
        except KeyError:
            pass

        # If dates are given as nanos, pop them
        try:
            kwargs['start_date'] = kwargs.pop('start_date_nano')
        except KeyError:
            pass
        try:
            kwargs['end_date'] = kwargs.pop('end_date_nano')
        except KeyError:
            pass
        try:
            kwargs['notice_date'] = kwargs.pop('notice_date_nano')
        except KeyError:
            pass
        try:
            kwargs['expiration_date'] = kwargs.pop('expiration_date_nano')
        except KeyError:
            pass

        # Process dates to Timestamps
        try:
            kwargs['start_date'] = pd.Timestamp(kwargs['start_date'], tz='UTC')
        except KeyError:
            pass
        try:
            kwargs['end_date'] = pd.Timestamp(kwargs['end_date'], tz='UTC')
        except KeyError:
            pass
        try:
            kwargs['notice_date'] = pd.Timestamp(kwargs['notice_date'],
                                                 tz='UTC')
        except KeyError:
            pass
        try:
            kwargs['expiration_date'] = pd.Timestamp(kwargs['expiration_date'],
                                                     tz='UTC')
        except KeyError:
            pass

        # Build an Asset of the appropriate type, default to Equity
        asset_type = kwargs.pop('asset_type', 'equity')
        if asset_type.lower() == 'equity':
            asset = Equity(**kwargs)
        elif asset_type.lower() == 'future':
            asset = Future(**kwargs)
        else:
            raise InvalidAssetType(asset_type=asset_type)

        return asset


class AssetFinder(object):

    def __init__(self, db_conn, fuzzy_char='_'):

        self.db_conn = db_conn

        self.fuzzy_char = fuzzy_char

    @lru_cache(maxsize=None)
    def retrieve_asset(self, sid):
        if isinstance(sid, Asset):
            return sid
        # TODO: Lookup SID here.
        # fetchone
        # select from securities where sid =
        # For now try both tables
        # first try equities then futures, down the road need a master table.
        try:
            asset = self.equity_for_id(sid)
            if asset is not None:
                return asset
        except:
            pass

        try:
            futures_contract = self.futures_contract_for_id(sid)
            if futures_contract is not None:
                return asset
        except:
            # TODO, make futures table in all case.
            pass

        raise SidNotFound(sid=sid)

    def lookup_symbol_resolve_multiple(self, symbol, as_of_date=None):
        """
        Return matching Asset of name symbol in database.

        If multiple Assets are found and as_of_date is not set,
        raises MultipleSymbolsFound.

        If no Asset was active at as_of_date, and allow_expired is False
        raises SymbolNotFound.
        """
        if as_of_date is not None:
            as_of_date = normalize_date(as_of_date)

        # TODO: Lookup SID
        # select from securities where symbol = '' and > as_of_date
        c = self.db_conn.cursor()
        c.row_factory = dict_factory
        fields = (
            'sid',
            'end_date',
            'start_date',
            'exchange',
            'symbol',
            'first_traded',
            'asset_name',
        )
        if as_of_date:
            t = (symbol, as_of_date.value)
            query = ("select {0} from equities where " +
                     "symbol=? and start_date>=? limit 1").format(
                         ", ".join(fields))
            c.execute(query, t)
            data = c.fetchone()

            if data is None:
                raise SymbolNotFound(symbol=symbol)
            else:
                return Equity(**data)
        else:
            t = (symbol,)
            query = ("select {0} from equities where symbol=?".format(
                ", ".join(fields)))
            c.execute(query, t)
            data = c.fetchone()

            if len(data) == 1:
                return Equity(**data[0])
            elif not data:
                raise SymbolNotFound(symbol=symbol)
            else:
                raise MultipleSymbolsFound(symbol=symbol,
                                           options=str(data))

    @lru_cache(maxsize=None)
    def lookup_symbol(self, symbol, as_of_date, fuzzy=None):
        """
        If a fuzzy string is provided, then we try various symbols based on
        the provided symbol.  This is to facilitate mapping from a broker's
        symbol to ours in cases where mapping to the broker's symbol loses
        information. For example, if we have CMCS_A, but a broker has CMCSA,
        when the broker provides CMCSA, it can also provide fuzzy='_',
        so we can find a match by inserting an underscore.
        """
        symbol = symbol.upper()
        as_of_date = normalize_date(as_of_date)

        if not fuzzy:
            try:
                return self.lookup_symbol_resolve_multiple(symbol, as_of_date)
            except SymbolNotFound:
                return None
        else:
            c = self.db_conn.cursor()
            fuzzer = symbol.replace(fuzzy, '')
            c.row_factory = dict_factory
            fields = (
                'sid',
#                'root_symbol',
                'end_date',
                'start_date',
                'exchange',
#                'expiration_date',
                'symbol',
#                'contract_multiplier',
#                'notice_date',
                'first_traded',
                'asset_name',
            )
            t = (fuzzer, as_of_date.value)
            query = ("select {0} from equities where fuzzy=? " +
                     "and start_date>=?").format(", ".join(fields))
            c.execute(query, t)
            data = c.fetchone()
            if data:
                return Equity(**data)

    def _sort_future_chains(self):
        """ Sort by increasing expiration date the list of contracts
        for each root symbol in the future cache.
        """
        exp_key = operator.attrgetter('expiration_date')

        for root_symbol in self.future_chains_cache:
            self.future_chains_cache[root_symbol].sort(key=exp_key)

    def _valid_contracts(self, root_symbol, as_of_date):
        """ Returns  a list of the currently valid futures contracts
        for a given root symbol, sorted by expiration date (the
        contracts are sorted when the AssetFinder is built).

        New, returns list of sids of contracts.
        """
        c = self.db_conn.cursor()
        t = {'root_symbol': root_symbol,
             'as_of_date': as_of_date.value}
        c.execute("""
        select sid, expiration_date from futures
        where root_symbol=:root_symbol
        and expiration_date > :as_of_date
        and start_date <= :as_of_date
        order by expiration_date asc
        """, t)
        return c.fetchall()

    def lookup_future_chain(self, root_symbol, as_of_date):
        """ Return the futures chain for a given root symbol.

        Parameters
        ----------
        root_symbol : str
            Root symbol of the desired future.
        as_of_date : pd.Timestamp
            Date at the time of the lookup.

        Returns
        -------
        [Future]
        """
        root_symbol.upper()
        as_of_date = normalize_date(as_of_date)
        return [self.futures_contract_for_id(sid)
                for sid, _ in
                self._valid_contracts(root_symbol, as_of_date)]

    def lookup_future_in_chain(self, root_symbol, as_of_date, contract_num=0):
        """ Find a specific contract in the futures chain for a given
        root symbol.

        Parameters
        ----------
        root_symbol : str
            Root symbol of the desired future.
        as_of_date : pd.Timestamp
            Date at the time of the lookup.
        contract_num : int
            1 for the primary contract, 2 for the secondary, etc.,
            relative to as_of_date.

        Returns
        -------
        Future
            The (contract_num)th contract in the futures chain. If none
            exits, returns None.
        """
        root_symbol.upper()
        as_of_date = normalize_date(as_of_date)

        valid_contracts = self._valid_contracts(root_symbol, as_of_date)

        if valid_contracts and contract_num >= 0:
            try:
                return self.futures_contract_for_id(
                    valid_contracts[contract_num][0])
            except IndexError:
                pass

        return None

    @lru_cache(maxsize=None)
    def futures_contract_for_id(self, contract_id):
        c = self.db_conn.cursor()
        t = (contract_id,)
        c.row_factory = dict_factory
        fields = (
            'sid',
            'root_symbol',
            'end_date',
            'start_date',
            'exchange',
            'expiration_date',
            'symbol',
            'contract_multiplier',
            'notice_date',
            'first_traded',
            'asset_name',
        )
        query = 'select {0} from futures where sid=?'.format(", ".join(fields))
        c.execute(query, t)
        data = c.fetchone()
        if data:
            return Future(**data)

    @lru_cache(maxsize=None)
    def equity_for_id(self, sid):
        c = self.db_conn.cursor()
        t = (sid,)
        c.row_factory = dict_factory
        fields = (
            'sid',
            'end_date',
            'start_date',
            'exchange',
            'symbol',
            'first_traded',
            'asset_name',
        )
        query = 'select {0} from equities where sid=?'.\
                format(", ".join(fields))
        c.execute(query, t)
        data = c.fetchone()
        if data:
            return Equity(**data)

    def lookup_future_by_expiration(self, root_symbol, as_of_date, ref_date):
        """ Find a specific contract in the futures chain by expiration
        date.

        Parameters
        ----------
        root_symbol : str
            Root symbol of the desired future.
        as_of_date : pd.Timestamp
            Date at the time of the lookup.
        ref_date : pd.Timestamp
            Reference point for expiration dates.

        Returns
        -------
        Future
            The valid contract the has the closest expiration date
            after ref_date. If none exists, returns None.
        """
        root_symbol.upper()
        as_of_date = normalize_date(as_of_date)
        ref_date = normalize_date(ref_date).value

        valid_contracts = self._valid_contracts(root_symbol, as_of_date)

        contracts_after_date = (sid for sid, expiration_date in
                                valid_contracts
                                if expiration_date > ref_date)
        contract_id = next(contracts_after_date, None)
        if contract_id is not None:
            return self.futures_contract_for_id(contract_id)

    @property
    def sids(self):
        return self.cache.keys()

    @property
    def assets(self):
        return self.cache.values()

    def _lookup_generic_scalar(self,
                               asset_convertible,
                               as_of_date,
                               matches,
                               missing):
        """
        Convert asset_convertible to an asset.

        On success, append to matches.
        On failure, append to missing.
        """
        try:
            if isinstance(asset_convertible, Asset):
                matches.append(asset_convertible)

            elif isinstance(asset_convertible, Integral):
                result = self.retrieve_asset(int(asset_convertible))
                if result is None:
                    raise SymbolNotFound(symbol=asset_convertible)
                matches.append(result)

            elif isinstance(asset_convertible, string_types):
                # Throws SymbolNotFound on failure to match.
                matches.append(
                    self.lookup_symbol_resolve_multiple(
                        asset_convertible,
                        as_of_date,
                    )
                )
            else:
                raise NotAssetConvertible(
                    "Input was %s, not AssetConvertible."
                    % asset_convertible
                )

        except SymbolNotFound:
            missing.append(asset_convertible)
            return None

    def lookup_generic(self,
                       asset_convertible_or_iterable,
                       as_of_date):
        """
        Convert a AssetConvertible or iterable of AssetConvertibles into
        a list of Asset objects.

        This method exists primarily as a convenience for implementing
        user-facing APIs that can handle multiple kinds of input.  It should
        not be used for internal code where we already know the expected types
        of our inputs.

        Returns a pair of objects, the first of which is the result of the
        conversion, and the second of which is a list containing any values
        that couldn't be resolved.
        """
        matches = []
        missing = []

        # Interpret input as scalar.
        if isinstance(asset_convertible_or_iterable, AssetConvertible):
            self._lookup_generic_scalar(
                asset_convertible=asset_convertible_or_iterable,
                as_of_date=as_of_date,
                matches=matches,
                missing=missing,
            )
            try:
                return matches[0], missing
            except IndexError:
                if hasattr(asset_convertible_or_iterable, '__int__'):
                    raise SidNotFound(sid=asset_convertible_or_iterable)
                else:
                    raise SymbolNotFound(symbol=asset_convertible_or_iterable)

        # Interpret input as iterable.
        try:
            iterator = iter(asset_convertible_or_iterable)
        except TypeError:
            raise NotAssetConvertible(
                "Input was not a AssetConvertible "
                "or iterable of AssetConvertible."
            )

        for obj in iterator:
            self._lookup_generic_scalar(obj, as_of_date, matches, missing)
        return matches, missing

    def map_identifier_index_to_sids(self, index, as_of_date):
        """
        This method is for use in sanitizing a user's DataFrame or Panel
        inputs.

        Takes the given index of identifiers, checks their types, builds assets
        if necessary, and returns a list of the sids that correspond to the
        input index.

        Parameters
        __________
        index : Iterable
            An iterable containing ints, strings, or Assets
        as_of_date : pandas.Timestamp
            A date to be used to resolve any dual-mapped symbols

        Returns
        _______
        List
            A list of integer sids corresponding to the input index
        """
        # This method assumes that the type of the objects in the index is
        # consistent and can, therefore, be taken from the first identifier
        first_identifier = index[0]

        # Ensure that input is AssetConvertible (integer, string, or Asset)
        if not isinstance(first_identifier, AssetConvertible):
            raise MapAssetIdentifierIndexError(obj=first_identifier)

        # If sids are provided, no mapping is necessary
        if isinstance(first_identifier, Integral):
            return index

        # If symbols or Assets are provided, construction and mapping is
        # necessary
        self.consume_identifiers(index)
        self.populate_cache()

        # Look up all Assets for mapping
        matches = []
        missing = []
        for identifier in index:
            self._lookup_generic_scalar(identifier, as_of_date,
                                        matches, missing)

        # Handle missing assets
        if len(missing) > 0:
            warnings.warn("Missing assets for identifiers: " + missing)

        # Return a list of the sids of the found assets
        return [asset.sid for asset in matches]


class AssetConvertible(with_metaclass(ABCMeta)):
    """
    ABC for types that are convertible to integer-representations of
    Assets.

    Includes Asset, six.string_types, and Integral
    """
    pass

AssetConvertible.register(Integral)
AssetConvertible.register(Asset)
# Use six.string_types for Python2/3 compatibility
for type in string_types:
    AssetConvertible.register(type)


class NotAssetConvertible(ValueError):
    pass
