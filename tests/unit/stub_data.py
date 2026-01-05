import datetime
import statistics
from math import prod

import numpy as np
import pandas as pd

from portfolio_mgmt.core.constants import WEEKS_IN_YEAR
from portfolio_mgmt.core.data import DataAccess

BASE_PRICE = 100
RETURNS = [0.02, -0.01, 0.02, 0.01, -0.005, 0.014, 0.01, -0.005, 0.014]
PRICES = [BASE_PRICE * prod([1 + r for r in RETURNS[:i]]) for i in range(len(RETURNS) + 1)]
PRICES_START_DATE = pd.to_datetime("2020-01-03").date()
PRICES_END_DATE = pd.to_datetime("2020-03-06").date()
PRICES_MEAN = 103.69
PRICES_STDEV = 2.24
RETURNS_MEAN = statistics.mean(RETURNS) * WEEKS_IN_YEAR
RETURNS_STDEV = statistics.stdev(RETURNS) * np.sqrt(WEEKS_IN_YEAR)
RETURNS_GEO_MEAN = (statistics.geometric_mean(1 + r for r in RETURNS)) ** WEEKS_IN_YEAR - 1

TICKER = "TEST_TICKER"
TICKER_2 = "TEST_TICKER_2"
TICKER_3 = "TEST_TICKER_3"
TICKER_4 = "TEST_TICKER_4"


def sample_prices():
    """10-week sample prices fixture.

    Provides a DataFrame of weekly Close prices (Fridays) used by multiple ETF tests.
    The values are deterministic so tests can assert returns, start/end dates and simple
    statistics. The fixture also includes a couple of inline sanity assertions that verify
    the mean and (sample) standard deviation of the series remain as expected.
    """
    date_list = [PRICES_START_DATE + datetime.timedelta(weeks=x)
                 for x in range((PRICES_END_DATE - PRICES_START_DATE).days // 7 + 1)]
    df = pd.DataFrame(
        {"Close": PRICES},
        index=date_list,
    )
    assert df.index.min() == PRICES_START_DATE
    assert df.index.max() == PRICES_END_DATE
    assert round(df["Close"].mean(), 2) == PRICES_MEAN
    assert round(df["Close"].std(), 2) == PRICES_STDEV
    return df


def sample_etf_series(ticker: str):
    keys = [
        "fundName",
        "fundSize",
        "inceptionDate",
        "ter",
        "ticker",
        "index",
        "asset_class",
        "grouping",
        "returns_5y",
        "holdings",
        "category",
    ]
    values = [
        "My Fund",
        1e6,
        "2019-01-01",
        0.005,
        ticker,
        "IDX",
        "Equity",
        "Grouping",
        0.12,
        100,
        "Category",
    ]
    s = pd.Series(dict(zip(keys, values, strict=True)))
    s.name = f"ISIN-{ticker}"
    return s


def sample_countries():
    # DataFrame indexed by country name with a 'weight' column
    df = pd.DataFrame({"weight": [0.6, 0.4], "isin": ["ISIN1", "ISI2"]}, index=["Germany", "USA"])
    return df


class StubDataAccess(DataAccess):
    """A lightweight Test double for DataAccess used in tests.

    This stub mirrors the minimal interface expected by the code under test and
    returns empty but well-typed objects when no data is provided. Tests may
    still override `engine.lib.etf.DataAccess` with a lambda returning a
    custom instance when they need to inject specific series/dataframes.
    """

    def __init__(self):
        # prices_df should be a DataFrame with a DateTimeIndex and a 'Close' column
        self._prices = {
            TICKER: sample_prices(),
            TICKER_2: sample_prices() * 2,
            TICKER_3: sample_prices().drop(sample_prices().index[:2])
        }
        self._etf = {
            TICKER: sample_etf_series(TICKER),
            TICKER_2: sample_etf_series(TICKER_2),
            TICKER_3: sample_etf_series(TICKER_3)
        }
        self._countries = sample_countries()

    def get_etf_prices(self, ticker, align=True, smooth_spikes=True):
        # Return a copy so tests can mutate without affecting other tests
        return self._prices[ticker].copy()

    def get_etf_countries(self, ticker):
        return self._countries.copy()

    def get_etf(self, ticker):
        return self._etf[ticker]

