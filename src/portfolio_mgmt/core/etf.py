"""Module defining the Etf class representing an Exchange Traded Fund (ETF) and related functionality.

This modules provides the functionality to create ETF instances from ticker symbols and operate on their data,
including prices, returns, country allocations, and risk-reward metrics.

The ETF is the core building block of a portfolio in this library.
"""

import datetime
from typing import Any

import pandas as pd

from .environment import Environment
from .risk_reward import RiskReward


class Etf:
    """Class representing an Exchange Traded Fund (ETF).

    An ETF is a type of investment fund that is traded on stock exchanges and holds assets such as stocks, commodities,
    or bonds.

    This class provides methods to access ETF data, including prices, returns, allocations to countries,
    and risk-reward metrics.

    Attributes:
        isin: The International Securities Identification Number of the ETF. This is the unique identifier of the
              security.
        fund_name: The name of the ETF.
        fund_size: The size of the fund in million euros.
        inception_date: The date when the ETF was launched.
        ter: The Total Expense Ratio of the ETF. This is the annual management fee charged by the fund.
        ticker: The commonly used ticker symbol of the ETF. The ticker is an exchange-specific identifier used to
                designate a security trading at a specific exchange. An ETF can have multiple tickers if it is listed
                on multiple exchanges. We currently only support a single ticker per ETF, the most common one.
        index: The index that the ETF aims to replicate.
        asset_class: The asset class of the ETF (e.g., equity, bond, commodity).
        grouping: The grouping of the ETF (e.g., global, regional, country).
        returns_5y: The annualized return of the ETF over the past 5 years.
        holdings: The number of individual securities held by the ETF.
        category: The category of the ETF (e.g., equity, bond, commodity).

    See also DataAccessCSV.get_etf_data() method.
    """
    def __init__(self, isin: str, fundName: str, fundSize: float, inceptionDate: str, ter: float, ticker: str, # noqa: N803
                 index: str, asset_class: str, grouping: str, returns_5y: float, holdings: int, category: str):
        """Initialize an ETF instance."""
        self.isin                = isin
        self.fund_name           = fundName
        self.fund_size           = fundSize
        self.inception_date      = inceptionDate
        self.ter                 = ter
        self.ticker              = ticker
        self.index               = index
        self.asset_class         = asset_class
        self.grouping            = grouping
        self.returns_5y          = returns_5y
        self.holdings            = holdings
        self.category            = category

    @classmethod
    def _data_access(cls):
        return Environment.current().data_access()

    def _unbounded_prices(self) -> 'pd.Series[float]':
        """Return the whole prices history available for this ETF. Prices at close."""
        return self._data_access().get_etf_prices(self.ticker)["Close"]

    def _bounded_prices(self):
        """Return bounded prices based on the current environment start and end dates.

        Always use this method, which respects the current environment.
        """
        prices = self._unbounded_prices()
        env = Environment.current()
        if env.start_date is not None:
            prices = prices.loc[env.start_date:]
        if env.end_date is not None:
            prices = prices.loc[:env.end_date]
        return prices

    @property
    def prices(self):
        """Get the prices of the ETF, bounded by the current environment's start and end dates.

        Currently, prices are weekly and at close, with adjustment for splits and dividends.

        Returns:
            A Series containing the weekly prices of the ETF. The index is the date.
        """
        return self._bounded_prices()

    @property
    def returns(self) -> 'pd.Series[float]':
        """Get the returns of the ETF, bounded by the current environment's start and end dates.

        Currently, returns are weekly and based on the prices at close. Using weekly returns is important to
        avoid correlation issues that arise with shorter timeframes and products trading in different timezones.

        Returns:
            A Series containing the weekly returns of the ETF. The index is the date.
            The series has one entry less than the prices series, which is needed to start the returns calculation.
        """
        return Etfs([self]).returns[0]

    @property
    def prices_start_date(self) -> datetime.date:
        """Get the first date for which prices are available, considering the current environment."""
        return self.prices.index.min()

    @property
    def prices_end_date(self) -> datetime.date:
        """Get the last date for which prices are available, considering the current environment."""
        return self.prices.index.max()

    @property
    def risk_reward(self):
        """Get the annualized risk-reward profile of the ETF considering the current environment.

        The risk-reward profile is calculated based on the returns of the ETF and represents metrics such as
        the Sharpe ratio, volatility, and annual return.

        Returns:
            RiskReward: An instance containing the risk-reward metrics.
        """
        return RiskReward.from_weekly_returns(self.returns)

    @property
    def countries(self) -> pd.DataFrame:
        """Get the country allocation of the ETF.

        ETFs can have exposure to multiple countries. This property returns those countries and their weights
        which represent the level of exposure of an ETF to the given countries.

        Returns:
            A DataFrame containing the countries and the respective weights that the ETF invests in.
            The index is the country name. Columns:
            - weight: The weight that the ETF invests in the country.
            - isin: The ISIN code of the ETF that invests in the country.

        """
        return self._data_access().get_etf_countries(self.ticker)

    @property
    def last_price(self):
        """Get the last available price of the ETF, considering the current environment."""
        return self.prices.iloc[-1]

    @classmethod
    def from_ticker(cls, ticker: str) -> 'Etf':
        """Create an ETF instance from its ticker symbol.

        Args:
            ticker (str): The ticker symbol of the ETF.

        Returns:
            An instance to the ETF corresponding to the given ticker.
        """
        data = cls._data_access().get_etf(ticker)
        etf = cls(**(data.to_dict()), isin=str(data.name))
        return etf

    def to_dict(self) -> dict[str, Any]:
        """Convert the ETF instance to a dictionary representation.

        Returns:
            A dictionary containing the ETF's attributes.
        """
        return {
            **self.__dict__,
            "prices_start_date": self.prices_start_date,
            "prices_end_date": self.prices_end_date,
            "risk_reward": self.risk_reward.to_dict(),
            "countries": [{ "country": country.Index, "weight": country.weight }
                          for country in self.countries.itertuples()],
            "last_price": self.last_price,
        }

    def __str__(self):
        """Return a string representation of the ETF instance."""
        return (f"{self.fund_name} ({self.ticker}:{self.category}),"
                f" prices from {self.prices_start_date} to {self.prices_end_date} @ C{self.last_price:.2f},"
                f" SR {self.risk_reward.sharpe:.2f}")


class Etfs(list[Etf]):
    """Class representing a collection of ETFs.

    This is mostly a convenience class to provide a common place for certain methods.
    """

    @property
    def tickers(self):
        """Get the tickers of the ETFs in the collection.

        Returns:
            A list of ticker symbols for the ETFs.
        """
        return [etf.ticker for etf in self]

    @property
    def returns(self):
        """Get the returns of the ETFs in the collection.

        This methods uses the prices property of each ETF to calculate the returns. The returns are not concatenated,
        but rather, they are treated independently and returned as a list. This means that any misalignment in dates
        between the ETF prices will **not** result in NaN returns.

        Returns:
            A list of Series containing the returns of the ETFs. The Series are in the same order as the ETFs in the
            collection and the length of each depends on the price data available for that ETF.
        """
        returns = [etf.prices.pct_change().dropna() for etf in self]
        return returns

    @property
    def returns_df(self):
        """Get the returns of the ETFs in the collection as a DataFrame.

        Returns:
            A DataFrame containing the returns of the ETFs. The index is the concatenation of all the dates
            available for the collection of ETFs. If an ETF does not have returns for a specific date,
            the value will be NaN (i.e., no truncation happens). The columns are the ticker symbols.
        """
        return pd.DataFrame(dict(zip(self.tickers, self.returns, strict=True)))
