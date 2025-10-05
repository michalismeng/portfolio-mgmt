"""Provides functionality to read ETF-related data and prices from CSV files.

Notes on the current example dataset:
- The example dataset is not exhaustive and only contains a small number of ETFs.
- A representative portfolio can be constructed from the available ETFs.
- The example prices are fetched from Yahoo Finance and stored in `weekly_prices.csv` file.
- A ticker symbol might be available in multiple exchanges, in case of multiple listings. The one with the most
  historical data is chosen.
- The adjusted closing price is used as the "closing price". This price is adjusted for dividends and stock splits
  as per the Yahoo Finance API.
"""

from pathlib import Path

import pandas as pd


class DataAccessCSV:
    """Class to load ETF data from CSV files located in a specified directory.

    This class provides methods to extract data from CSV files, including ETF information, country allocations,
    and weekly prices and returns them as pandas DataFrames.

    By default, it operates on the example dataset provided in this package, in the 'examples/data' directory.

    The dataset should contain the following files:
    - etfs.csv: Information about the available ETFs.
    - countries.csv: Information about which countries ETFs invest in.
    - weekly-prices.csv: Weekly prices for the ETFs.
    """

    _DEFAULT_DATA_DIR = "examples/data"
    _ETF_DATA_FILE = "etfs.csv"
    _COUNTRY_DATA_FILE = "countries.csv"
    _PRICES_DATA_FILE = "weekly-prices.csv"

    def __init__(self, data_dir: str = _DEFAULT_DATA_DIR):
        """Initialize a new DataAccessCSV instance.

        For more information on the expected CSV format, refer to the methods of the class that read those files.

        Args:
            data_dir: The directory where the CSV files are located. If not provided, defaults to 'data'.
        """
        self.data_dir = Path.resolve(Path(data_dir))

    def get_etf_data(self):
        """Get all the available ETF data.

        Read all the available ETFs from the CSV file.

        Returns:
            A DataFrame with ETF information, indexed by isin. Columns:
            - fundName: The name of the ETF.
            - fundSize: The size of the ETF fund in million EUR.
            - inceptionDate: The date the ETF was created.
            - ter: The total annual expense ratio of the ETF.
            - ticker: The ticker symbol of the ETF.
            - index: The index the ETF is tracking.
            - asset_class: The asset class of the ETF.
            - grouping: The grouping of the ETF. Subcategory of asset_class.
            - returns_5y: The 5-year return of the ETF, if available.
            - holdings: The number of holdings of the ETF.
            - category: The category of the ETF. Subcategory of asset_class.
        """
        etf_path = Path(self.data_dir) / self._ETF_DATA_FILE
        etfs = pd.read_csv(etf_path, index_col=0)
        return etfs

    def get_countries_data(self):
        """Get information about which countries ETFs invest in.

        Read all the available countries and their corresponding ETFs from the CSV file.

        Returns:
            A DataFrame with country information, indexed by country code. Columns:
            - weight: The weight that the ETF invests in the country.
            - isin: The ISIN code of the ETF that invests in the country.
        """
        countries_path = Path(self.data_dir) / self._COUNTRY_DATA_FILE
        countries = pd.read_csv(countries_path, index_col=0)
        return countries

    def get_weekly_prices(self):
        """Get all the available weekly prices.

        Read all the available weekly prices from the CSV file.

        The example prices are fetched from Yahoo Finance and stored in `weekly_prices.csv` file. A ticker symbol
        might be available in multiple exchanges, in case of multiple listings. The one with the most historical
        data is chosen.

        The closing price should be adjusted for dividends and stock splits.

        Returns:
            A DataFrame with weekly prices, indexed by date. Columns:
            - Ticker: The ticker symbol of the ETF in YFinance notation.
            - Close: The closing price of the ETF for the week designated by the date.
            - Volume: The weekly trading volume of the ETF.
        """
        weekly_prices_path = Path(self.data_dir) / self._PRICES_DATA_FILE
        weekly_prices = pd.read_csv(weekly_prices_path, index_col=0)
        return weekly_prices
