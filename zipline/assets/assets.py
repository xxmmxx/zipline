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

from numbers import Integral
import numpy as np
import sqlite3
from sqlite3 import Row
import warnings

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
from abc import (
    ABCMeta,
    abstractmethod,
)

log = Logger('assets.py')

# Expected fields for an Asset's metadata
ASSET_FIELDS = [
    'sid',
    'asset_type',
    'symbol',
    'asset_name',
    'start_date',
    'end_date',
    'first_traded',
    'exchange',
    'notice_date',
    'root_symbol',
    'expiration_date',
    'contract_multiplier',
    # The following fields are for compatibility with other systems
    'file_name',  # Used as symbol
    'company_name',  # Used as asset_name
    'start_date_nano',  # Used as start_date
    'end_date_nano',  # Used as end_date
]


# Expected fields for an Asset's metadata
ASSET_TABLE_FIELDS = [
    'sid',
    'symbol',
    'asset_name',
    'start_date',
    'end_date',
    'first_traded',
    'exchange',
]


# Expected fields for an Asset's metadata
FUTURE_TABLE_FIELDS = ASSET_TABLE_FIELDS + [
    'root_symbol',
    'notice_date',
    'expiration_date',
    'contract_multiplier',
]

EQUITY_TABLE_FIELDS = ASSET_TABLE_FIELDS


# Create the query once from the fields, so that the join is not done
# repeatedly.
FUTURE_BY_SID_QUERY = 'select {0} from futures_contracts where sid=?'.format(
    ", ".join(FUTURE_TABLE_FIELDS))

EQUITY_BY_SID_QUERY = 'select {0} from equities where sid=?'.format(
    ", ".join(EQUITY_TABLE_FIELDS))


class AssetDBWriter(with_metaclass(ABCMeta)):

    def write_all(self, assets):
        """Top-level entry point for writing a new asset db.

        Parameters
        ----------
        assets
            The data to write to our database.
        """

        for sid, metadata in self.gen_data(assets):
            self.write_block(sid, **metadata)

    @abstractmethod
    def gen_data(self, assets):
        """
        Returns a generator yielding pairs of (sid, metadata).
        """
        return NotImplementedError()

    def init_db(self,
                connection_manager,
                fuzzy_char=None,
                allow_sid_assignment=True,
                constraints=False):
        """Connect to db and create tables.

        Parameters
        ----------
        connection_manager: DBConnectionManager
            Object containing SQLite connection information.
        fuzzy_char: string
            A string used in fuzzy character matching.
        allow_sid_assignment: boolean
            Allow sid assignment if sids not provided.
        constraints: boolean
            Create SQL ForeignKey and Index constraints.
        """
        self.db = connection_manager
        self.allow_sid_assignment = allow_sid_assignment
        self.fuzzy_char = fuzzy_char

        # This flag controls if the AssetDBWriter is allowed to generate its
        # own sids. If False, metadata that does not contain a sid will raiset
        # an exception when building assets.
        if allow_sid_assignment:
            self.end_date_to_assign = normalize_date(
                pd.Timestamp('now', tz='UTC'))

        c = self.db.cur

        # The AssetDBWriter holds a nested-dict of all metadata for
        # reference when building Assets
        self.metadata_cache = {}

        c.execute("""
            CREATE TABLE IF NOT EXISTS equities (
            sid INTEGER,
            symbol TEXT,
            asset_name TEXT,
            start_date INTEGER,
            end_date INTEGER,
            first_traded INTEGER,
            exchange TEXT,
            fuzzy TEXT
        )""")

        c.execute("""
            CREATE TABLE IF NOT EXISTS futures_exchanges (
            exchange_id INTEGER NOT NULL,
            exchange TEXT NOT NULL,
            timezone TEXT
        )""")

        c.execute("""
            CREATE TABLE IF NOT EXISTS futures_root_symbols (
            root_symbol_id INTEGER NOT NULL,
            root_symbol TEXT,
            sector TEXT,
            description TEXT,
            exchange TEXT{fk}
        )""".format(fk=", FOREIGN KEY(exchange) REFERENCES "
                    "futures_exchanges(exchange)"
                    if constraints else ""))

        c.execute("""
            CREATE TABLE IF NOT EXISTS futures_contracts (
            sid INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            root_symbol TEXT NOT NULL,
            asset_name TEXT,
            start_date INTEGER,
            end_date INTEGER,
            first_traded INTEGER,
            exchange TEXT,
            notice_date INTEGER,
            expiration_date INTEGER,
            contract_multiplier REAL{fk}
        )""".format(fk=", FOREIGN KEY(exchange) REFERENCES "
                    "futures_exchanges(exchange), "
                    "FOREIGN KEY(root_symbol) REFERENCES "
                    "futures_root_symbols(root_symbol)"
                    if constraints else ""))

        c.execute("""
            CREATE TABLE IF NOT EXISTS asset_router
            (sid integer,
            asset_type text
        )""")

        if constraints:

            c.execute('CREATE INDEX IF NOT EXISTS ix_equities_sid '
                      'ON equities(sid)')
            c.execute('CREATE INDEX IF NOT EXISTS ix_equities_symbol '
                      'ON equities(symbol)')
            c.execute('CREATE INDEX IF NOT EXISTS ix_equities_fuzzy '
                      'ON equities(fuzzy)')
            c.execute('CREATE INDEX IF NOT EXISTS ix_futures_exchanges_en '
                      'ON futures_exchanges(exchange_id)')
            c.execute('CREATE INDEX IF NOT EXISTS ix_futures_contracts_sid '
                      'ON futures_contracts(sid)')
            c.execute('CREATE INDEX IF NOT EXISTS ix_futures_root_symbols_id '
                      'ON futures_root_symbols(root_symbol_id)')
            c.execute('CREATE INDEX IF NOT EXISTS ix_asset_router_sid  '
                      'ON asset_router(sid)')

        self.db.conn.commit()

    # Previously _insert_metadata
    # Should rename exchange to exchange_name
    # Named 'exchange' in short term to be consistent with
    # the Futures object
    def write_block(self, identifier, **kwargs):
        """
        Inserts the given metadata kwargs to the entry for the given
        sid. Matching fields in the existing entry will be overwritten.
        """

        if identifier in self.metadata_cache:
            # Multiple pass insertion no longer supported.
            # This could and probably should raise an Exception, but is
            # currently just a short-circuit for compatibility with existing
            # testing structure in the test_algorithm module which creates
            # multiple sources which all insert redundant metadata.
            return

        entry = {}

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
            # Do not accept NaNs from dataframes
            if isinstance(value, float) and np.isnan(value):
                continue
            entry[key] = value

        # Check if the sid is declared
        try:
            entry['sid']
        except KeyError:
            # If the sid is not a sid, assign one
            if hasattr(identifier, '__int__'):
                entry['sid'] = identifier.__int__()
            else:
                if self.allow_sid_assignment:
                    # Assign the sid the value of its insertion order.
                    # This assumes that we are assigning values to all assets.
                    entry['sid'] = len(self.metadata_cache)
                else:
                    raise SidAssignmentError(identifier=identifier)

        # If the file_name is in the kwargs, it will be used as the symbol
        try:
            entry['symbol'] = entry.pop('file_name')
        except KeyError:
            pass

        # If the identifier coming in was a string and there is no defined
        # symbol yet, set the symbol to the incoming identifier
        try:
            entry['symbol']
            pass
        except KeyError:
            if isinstance(identifier, string_types):
                entry['symbol'] = identifier

        # If the company_name is in the kwargs, it may be the asset_name
        try:
            company_name = entry.pop('company_name')
            try:
                entry['asset_name']
            except KeyError:
                entry['asset_name'] = company_name
        except KeyError:
            pass

        # If dates are given as nanos, pop them
        try:
            entry['start_date'] = entry.pop('start_date_nano')
        except KeyError:
            pass
        try:
            entry['end_date'] = entry.pop('end_date_nano')
        except KeyError:
            pass
        try:
            entry['notice_date'] = entry.pop('notice_date_nano')
        except KeyError:
            pass
        try:
            entry['expiration_date'] = entry.pop('expiration_date_nano')
        except KeyError:
            pass

        # Process dates to Timestamps
        try:
            entry['start_date'] = pd.Timestamp(entry['start_date'], tz='UTC')
        except KeyError:
            # Set a default start_date of the EPOCH, so that all date queries
            # work when a start date is not provided.
            entry['start_date'] = pd.Timestamp(0, tz='UTC')
        try:
            # Set a default end_date of 'now', so that all date queries
            # work when a end date is not provided.
            entry['end_date'] = pd.Timestamp(entry['end_date'], tz='UTC')
        except KeyError:
            entry['end_date'] = self.end_date_to_assign
        try:
            entry['notice_date'] = pd.Timestamp(entry['notice_date'],
                                                tz='UTC')
        except KeyError:
            pass
        try:
            entry['expiration_date'] = pd.Timestamp(entry['expiration_date'],
                                                    tz='UTC')
        except KeyError:
            pass

        # Build an Asset of the appropriate type, default to Equity
        asset_type = entry.pop('asset_type', 'equity')
        if asset_type.lower() == 'equity':
            try:
                fuzzy = entry['symbol'].replace(self.fuzzy_char, '') \
                    if self.fuzzy_char else None
            except KeyError:
                fuzzy = None
            asset = Equity(**entry)
            c = self.db.cur
            t = (asset.sid,
                 asset.symbol,
                 asset.asset_name,
                 asset.start_date.value if asset.start_date else None,
                 asset.end_date.value if asset.end_date else None,
                 asset.first_traded.value if asset.first_traded else None,
                 asset.exchange,
                 fuzzy)
            c.execute("""
                INSERT INTO equities (
                sid,
                symbol,
                asset_name,
                start_date,
                end_date,
                first_traded,
                exchange,
                fuzzy)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """, t)

            t = (asset.sid,
                 'equity')
            c.execute("""
                INSERT INTO asset_router (
                sid, asset_type)
                VALUES(?, ?)
            """, t)

        elif asset_type.lower() == 'future':
            asset = Future(**entry)
            c = self.db.cur
            t = (asset.sid,
                 asset.symbol,
                 asset.asset_name,
                 asset.start_date.value if asset.start_date else None,
                 asset.end_date.value if asset.end_date else None,
                 asset.first_traded.value if asset.first_traded else None,
                 asset.exchange,
                 asset.root_symbol,
                 asset.notice_date.value if asset.notice_date else None,
                 asset.expiration_date.value
                 if asset.expiration_date else None,
                 asset.contract_multiplier)
            c.execute("""
                INSERT INTO futures_contracts(
                sid,
                symbol,
                asset_name,
                start_date,
                end_date,
                first_traded,
                exchange,
                root_symbol,
                notice_date,
                expiration_date,
                contract_multiplier)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, t)

            t = (asset.sid,
                 'future')
            c.execute("""
                INSERT INTO asset_router (
                sid,
                asset_type)
                VALUES(?, ?)
            """, t)
        else:
            raise InvalidAssetType(asset_type=asset_type)

        self.metadata_cache[identifier] = entry

        self.db.conn.commit()

    def consume_identifiers(self, identifiers):
        """
        Consumes the given identifiers in to the metadata cache of this
        AssetDBWriter, and adds to database.
        """
        for identifier in identifiers:
            # Handle case where full Assets are passed in
            # For example, in the creation of a DataFrameSource, the source's
            # 'sid' args may be full Assets
            if isinstance(identifier, Asset):
                sid = identifier.sid
                metadata = identifier.to_dict()
                metadata['asset_type'] = identifier.__class__.__name__
                self.write_block(sid, **metadata)
            else:
                self.write_block(identifier)


class NullAssetDBWriter(AssetDBWriter):

    def gen_data(self, __):
        for i in iter(()):
            yield


class AssetDBWriterFromDictionary(AssetDBWriter):
    """ An implementation of AssetDBWriter for use
        with dictionaries.

        Expects a dictionary to be passed to gen_data
        with the following format:

        {id_0: {start_date : ...}, id_1: {start_data: ...}, ...}
    """

    def gen_data(self, dict):
        """
        Returns a generator yielding pairs of (identifier, metadata)
        """
        for identifier, metadata in dict.items():
            yield identifier, metadata


class AssetDBWriterFromDataFrame(AssetDBWriter):
    """ An implementation of AssetDBWriter for use
        with pandas DataFrames.

        Expects dataframe to be passed to gen_data
        to have the following structure:
            * column names must be the metadata fields
            * index must be the different asset identifiers
            * array contents should be the metadata value
    """

    def gen_data(self, dataframe):
        """
        Returns a generator yielding pairs of (identifier, metadata)
        """
        for identifier, row in dataframe.iterrows():
            yield identifier, row.to_dict()


class AssetDBWriterFromReadable(AssetDBWriter):
    """ An implementation of AssetDBWriter for use
        with objects with a 'read' property.

        If an the objects read method must return rows
        containing at least one of 'sid' or 'symbol' along
        with the other metadata fields.
    """

    def gen_data(self, readable):
        """
        Returns a generator yielding pairs of (identifier, metadata)
        """
        for row in readable.read():
            id_metadata = {}
            for field in ASSET_FIELDS:
                try:
                    row_value = row[field]
                    # Avoid passing placeholder strings
                    if row_value and (row_value != 'None'):
                        id_metadata[field] = row[field]
                except KeyError:
                    continue
                except IndexError:
                    continue
            if 'sid' in id_metadata:
                identifier = id_metadata['sid']
                del id_metadata['sid']
            elif 'symbol' in id_metadata:
                identifier = id_metadata['symbol']
                del id_metadata['symbol']
            else:
                raise ConsumeAssetMetaDataError(obj=row)
            yield identifier, id_metadata


class DBConnectionManager(object):
    """
    Class responsible for managing our connection to
    the database. The AssetDBWriter and AssetFinder
    share an instance of this class, to ensure they
    reference the same data.
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.text_factory = str
        self.cur = self.conn.cursor()

    def execute(self, query, t=()):
        """
        Execute query and commit it to our database.

        Parameters
        ----------
        query: string
            SQL query string to be executed.
        t: tuple of parameters to add to SQL string.
        """
        self.cur.execute(query, t)
        self.conn.commit()
        return self.cur

    def close(self):
        """
        Close our connection to the database.
        """
        self.conn.close()

    def __del__(self):
        self.conn.close()


def asset_finder(metadata,
                 allow_sid_assignment=True,
                 fuzzy_char=None,
                 db_path=':memory:',
                 create_table=True):
    """ Create an instance of the AssetFinder linked
        to an instance of AssetDBWriter relevant to
        processing metadata.
    """

    asset_writer = get_relevant_writer(metadata)

    return AssetFinder(metadata, allow_sid_assignment, fuzzy_char,
                       db_path, create_table, asset_writer)


def get_relevant_writer(metadata):
    """ Create an instance of AssetDBWriter relevant to
        processing metadata.
    """

    if isinstance(metadata, dict):
        return AssetDBWriterFromDictionary()
    elif isinstance(metadata, pd.DataFrame):
        return AssetDBWriterFromDataFrame()
    elif hasattr(metadata, 'read'):
        return AssetDBWriterFromReadable()
    else:
        raise ConsumeAssetMetaDataError(obj=metadata)


class AssetFinder(object):

    def __init__(self,
                 metadata=None,
                 allow_sid_assignment=True,
                 fuzzy_char=None,
                 db_path=':memory:',
                 create_table=True,
                 asset_writer=None):

        self.fuzzy_char = fuzzy_char
        self.allow_sid_assignment = allow_sid_assignment

        self.db = DBConnectionManager(db_path)

        if asset_writer is None:
            if metadata is None:
                # Default datasource is dictionary
                self.asset_writer = AssetDBWriterFromDictionary()
            else:
                self.asset_writer = get_relevant_writer(metadata)
        else:
            self.asset_writer = asset_writer

        # Create table and read in metadata.
        # Should we use flags like 'r', 'w', instead?
        # What we need to support is:
        # - A 'throwaway' mode where the metadata is read each run.
        # - A 'write' mode where the data is written to the provided db_path
        # - A 'read' mode where the asset finder uses a prexisting db.
        if create_table:
            self.asset_writer.init_db(self.db,
                                      fuzzy_char,
                                      allow_sid_assignment)
            if metadata is not None:
                self.asset_writer.write_all(metadata)

        # Cache for lookup of assets by sid, the objects in the asset lookp may
        # be shared with the results from equity and future lookup caches.
        #
        # The top level cache exists to minimize lookups on the asset type
        # routing.
        #
        # The caches are read through, i.e. accessing an asset through
        # retrieve_asset, _retrieve_equity etc. will populate the cache on
        # first retrieval.
        self._asset_cache = {}
        self._equity_cache = {}
        self._future_cache = {}

        self._asset_type_cache = {}

    def clear_metadata(self):
        """
        Used for testing.
        """
        # Clear the asset writers metadata cache
        self.asset_writer.metadata_cache = {}
        # Close the database connection
        self.db.close()
        # Create new database connection in memory.
        self.db = DBConnectionManager(db_path=':memory:')
        # Initialise the database with the same connection
        # params as used by the AssetFinder.
        self.asset_writer.init_db(self.db,
                                  self.fuzzy_char,
                                  self.allow_sid_assignment)

    def asset_type_by_sid(self, sid):
        """
        Retrieve the asset type of a given sid.
        """
        try:
            return self._asset_type_cache[sid]
        except KeyError:
            pass

        c = self.db.cur
        # Python 3 compatibility required forcing to int for sid = 0.
        t = (int(sid),)
        query = 'SELECT asset_type FROM asset_router WHERE sid=:sid'
        c.execute(query, t)
        data = c.fetchone()
        if data is None:
            return

        asset_type = data[0]
        self._asset_type_cache[sid] = asset_type

        return asset_type

    def retrieve_asset(self, sid, default_none=False):
        """
        Retrieve the Asset object of a given sid.
        """
        if isinstance(sid, Asset):
            return sid

        try:
            asset = self._asset_cache[sid]
        except KeyError:
            asset_type = self.asset_type_by_sid(sid)
            if asset_type == 'equity':
                asset = self._retrieve_equity(sid)
            elif asset_type == 'future':
                asset = self._retrieve_futures_contract(sid)
            else:
                asset = None

            self._asset_cache[sid] = asset

        if asset is not None:
            return asset
        elif default_none:
            return None
        else:
            raise SidNotFound(sid=sid)

    def _retrieve_equity(self, sid):
        """
        Retrieve the Equity object of a given sid.
        """
        try:
            return self._equity_cache[sid]
        except KeyError:
            pass

        c = self.db.cur
        c.row_factory = Row
        t = (int(sid),)
        c.execute(EQUITY_BY_SID_QUERY, t)
        data = dict(c.fetchone())
        if data:
            if data['start_date']:
                data['start_date'] = pd.Timestamp(data['start_date'], tz='UTC')

            if data['end_date']:
                data['end_date'] = pd.Timestamp(data['end_date'], tz='UTC')

            if data['first_traded']:
                data['first_traded'] = pd.Timestamp(
                    data['first_traded'], tz='UTC')

            equity = Equity(**data)
        else:
            equity = None

        self._equity_cache[sid] = equity
        return equity

    def _retrieve_futures_contract(self, sid):
        """
        Retrieve the Future object of a given sid.
        """
        try:
            return self._future_cache[sid]
        except KeyError:
            pass

        c = self.db.cur
        t = (int(sid),)
        c.row_factory = Row
        c.execute(FUTURE_BY_SID_QUERY, t)
        data = dict(c.fetchone())
        if data:
            if data['start_date']:
                data['start_date'] = pd.Timestamp(data['start_date'], tz='UTC')

            if data['end_date']:
                data['end_date'] = pd.Timestamp(data['end_date'], tz='UTC')

            if data['first_traded']:
                data['first_traded'] = pd.Timestamp(
                    data['first_traded'], tz='UTC')

            if data['notice_date']:
                data['notice_date'] = pd.Timestamp(
                    data['notice_date'], tz='UTC')

            if data['expiration_date']:
                data['expiration_date'] = pd.Timestamp(
                    data['expiration_date'], tz='UTC')

            future = Future(**data)
        else:
            future = None

        self._future_cache[sid] = future
        return future

    def lookup_symbol_resolve_multiple(self, symbol, as_of_date=None):
        """
        Return matching Asset of name symbol in database.

        If multiple Assets are found and as_of_date is not set,
        raises MultipleSymbolsFound.

        If no Asset was active at as_of_date, and allow_expired is False
        raises SymbolNotFound.
        """
        if as_of_date is not None:
            as_of_date = pd.Timestamp(normalize_date(as_of_date))

        c = self.db.cur

        if as_of_date:
            # If one SID exists for symbol, return that symbol
            t = (symbol, as_of_date.value, as_of_date.value)
            query = ("SELECT sid FROM equities "
                     "WHERE symbol=? "
                     "AND start_date<=? "
                     "AND end_date>=?")
            c.execute(query, t)
            candidates = c.fetchall()

            if len(candidates) == 1:
                return self._retrieve_equity(candidates[0][0])

            # If no SID exists for symbol, return SID with the
            # highest-but-not-over end_date
            if len(candidates) == 0:
                t = (symbol, as_of_date.value)
                query = ("SELECT sid FROM equities "
                         "WHERE symbol=? "
                         "AND start_date<=? "
                         "ORDER BY end_date DESC "
                         "LIMIT 1")
                c.execute(query, t)
                data = c.fetchone()

                if data:
                    return self._retrieve_equity(data[0])

            # If multiple SIDs exist for symbol, return latest start_date with
            # end_date as a tie-breaker
            if len(candidates) > 1:
                t = (symbol, as_of_date.value)
                query = ("SELECT sid FROM equities "
                         "WHERE symbol=? " +
                         "AND start_date<=? " +
                         "ORDER BY start_date DESC, end_date DESC " +
                         "LIMIT 1")
                c.execute(query, t)
                data = c.fetchone()

                if data:
                    return self._retrieve_equity(data[0])

            raise SymbolNotFound(symbol=symbol)

        else:
            t = (symbol,)
            query = ("SELECT sid FROM equities WHERE symbol=?")
            c.execute(query, t)
            data = c.fetchall()

            if len(data) == 1:
                return self._retrieve_equity(data[0][0])
            elif not data:
                raise SymbolNotFound(symbol=symbol)
            else:
                options = []
                for row in data:
                    sid = row[0]
                    asset = self._retrieve_equity(sid)
                    options.append(asset)
                raise MultipleSymbolsFound(symbol=symbol,
                                           options=options)

    def lookup_symbol(self, symbol, as_of_date, fuzzy=False):
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
            c = self.db.cur
            fuzzy = symbol.replace(self.fuzzy_char, '')
            t = (fuzzy, as_of_date.value, as_of_date.value)
            query = ("SELECT sid FROM EQUITIES "
                     "WHERE fuzzy=? " +
                     "AND start_date<=? " +
                     "AND end_date>=?")
            c.execute(query, t)
            candidates = c.fetchall()

            # If one SID exists for symbol, return that symbol
            if len(candidates) == 1:
                return self._retrieve_equity(candidates[0][0])

            # If multiple SIDs exist for symbol, return latest start_date with
            # end_date as a tie-breaker
            if len(candidates) > 1:
                t = (symbol, as_of_date.value)
                query = ("SELECT sid FROM equities "
                         "WHERE symbol=? " +
                         "AND start_date<=? " +
                         "ORDER BY start_date desc, end_date desc" +
                         "LIMIT 1")
                c.execute(query, t)
                data = c.fetchone()
                if data:
                    return self._retrieve_equity(data[0])

    def lookup_future_chain(self, root_symbol, as_of_date, knowledge_date):
        """ Return the futures chain for a given root symbol.

        Parameters
        ----------
        root_symbol : str
            Root symbol of the desired future.
        as_of_date : pd.Timestamp
            Date at which the chain determination is rooted. I.e. the
            existing contract whose notice date is first after this
            date is the primary contract, etc.
        knowledge_date : pd.Timestamp
            Date for determining which contracts exist for inclusion in
            this chain. Contracts exist only if they have a start_date
            on or before this date.

        Returns
        -------
        list
            A list of Future objects, the chain for the given
            parameters.

        Raises
        ------
        RootSymbolNotFound
            Raised when a future chain could not be found for the given
            root symbol.
        """
        c = self.db.cur
        t = {'root_symbol': root_symbol,
             'as_of_date': as_of_date.value,
             'knowledge_date': knowledge_date.value}
        c.execute("""
        SELECT sid FROM futures_contracts
        WHERE root_symbol=:root_symbol
        AND :as_of_date < notice_date
        AND start_date <= :knowledge_date
        ORDER BY notice_date ASC
        """, t)
        sids = [r[0] for r in c.fetchall()]
        if not sids:
            # Check if root symbol exists.
            c.execute("""
            SELECT COUNT(sid) FROM futures_contracts
            WHERE root_symbol=:root_symbol
            """, t)
            count = c.fetchone()[0]
            if count == 0:
                raise RootSymbolNotFound(root_symbol=root_symbol)
            else:
                # If symbol exists, return empty future chain.
                return []
        return [self._retrieve_futures_contract(sid) for sid in sids]

    @property
    def sids(self):
        c = self.db.cur
        query = 'SELECT sid FROM asset_router'
        c.execute(query)
        return [r[0] for r in c.fetchall()]

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
        self.asset_writer.consume_identifiers(index)

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

    def consume_identifiers(self, identifiers):
        """
        Consumes the provided identifiers, passing them to
        the asset writer to be added to the database.

        Parameters
        ----------
        identifiers
            The data to be consumed.
        """
        self.asset_writer.consume_identifiers(identifiers)

    def consume_metadata(self, metadata):
        """
        Consumes the provided metadata, passing it to
        the asset writer to be added to the database.

        Parameters
        ----------
        metadata
            The data to be consumed.
        """

        self.asset_writer.write_all(metadata)

    def insert_metadata(self, identifier, **kwargs):
        """
        Insert information for a single identifier.
        """

        self.asset_writer.write_block(identifier, **kwargs)


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
for _type in string_types:
    AssetConvertible.register(_type)


class NotAssetConvertible(ValueError):
    pass
