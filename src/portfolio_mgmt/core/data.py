"""Data abstraction layer for portfolio management.

This module provides a singleton class which standardizes access to various data sources.

The main functionality includes:
- Searching for ETFs by category, index, or grouping
- Retrieving ETF prices
- Retrieving country data in which ETFs invest in
- Combining all the above as an ETF dataframe

The data is currently sourced from CSV files using the DataAccessCSV class.
"""

import datetime
import logging
from abc import ABC, ABCMeta

import numpy as np
import pandas as pd

from .csv_data import DataAccessCSV

logger = logging.getLogger(__name__)


class SingletonMeta(ABCMeta):
    """Singleton metaclass allowing only one instance of a class and it's subclasses."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """If an instance of the class does not exist, create one. Otherwise, return the existing instance."""
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(ABC, metaclass=SingletonMeta):
    """Abstract singleton class."""

    pass


class DataAccess(Singleton):
    """Singleton class for accessing portfolio management data."""

    _initialized = False

    def __init__(self):
        """Initialize the DataAccess singleton instance.

        Load all data and store them in memory for fast access. The current implementation uses the DataAccessCSV
        class to load the example data from CSV files.
        """
        if self._initialized:
            return
        self._initialized = True
        self.data_layer = DataAccessCSV()

        self.etfs = self.data_layer.get_etf_data()
        logger.info(f"Loaded {len(self.etfs)} ETFs after excluding instruments")
        self.countries = self.data_layer.get_countries_data()
        weekly_prices = self.data_layer.get_weekly_prices()
        logger.info(f"Loaded {len(weekly_prices)} weekly prices")
        self.weekly_prices_dict = {ticker: row for ticker, row in weekly_prices.groupby("Ticker")}

    def find_etfs_by_category(self, category: str):
        """Find ETFs by a specific category.

        Arguments:
            category: A string representing the category to search for.

        Returns:
            A DataFrame containing the ETFs that exactly match the given category.
        """
        return self.etfs[self.etfs["category"] == category]

    def find_etf_for_index(self, indices=None, squeeze="first", match="contains"):
        """Find ETFs for a given index or list of indices.

        Arguments:
            indices: A string or list of strings representing the indices to search for.
                     Matched with OR and case insensitive.
            squeeze: If "first", return the first ETF found. If "choose", return a random ETF.
                     Leave as None to return all ETFs.
            match: A string indicating the type of match to perform. Can be "contains" or "exact".
                   If "contains", the index name must contain the given string (case insensitive).
                   If "exact", the index name must exactly match the given string (case sensitive).

        Returns:
            A DataFrame containing the ETFs that match the given indices.
        """
        if indices is None:
            indices = []
        if isinstance(indices, str):
            indices = [indices]

        pattern = "|".join(indices)
        # Escape the pattern for regex
        if match == "contains":
            pattern = (
                pattern.replace("(", "\\(")
                .replace(")", "\\)")
                .replace("+", "\\+")
                .replace("?", "\\?")
                .replace(".", "\\.")
            )
            etfs = self.etfs[self.etfs["index"].str.contains(pattern, case=False, na=False)]
        elif match == "exact":
            etfs = self.etfs[self.etfs["index"] == pattern]

        if squeeze is None:
            return etfs
        if squeeze == "first":
            return etfs.head(1)
        if squeeze == "choose":
            return etfs.sample(1)

    def get_etf(self, ticker) -> pd.Series:
        """Get the ETF data for a given ticker.

        Args:
            ticker: The ticker symbol of the ETF.

        Returns:
            A Series with the ETF data for the given ticker. The name of the Series is the ISIN of the ETF.
            For the fields of the Series, see the underlying 'data_layer.get_etf_data()' method.
        """
        etf: pd.DataFrame = self.etfs[self.etfs["ticker"] == ticker]
        if etf.empty:
            raise ValueError(f"No ETF found for ticker: {ticker}")
        return pd.Series(etf.squeeze())

    def get_etf_prices(self, ticker, align=True, smooth_spikes=True):
        """Get the weekly prices for a given ETF ticker.

        Args:
            ticker: The ticker symbol of the ETF.
            align: If True, align the indices to the same weekly period, starting on Monday.
                   This is necessary for concatenating weekly prices from multiple ETFs, which might not all start
                   on the same date, thus resulting in NaN values.
            smooth_spikes: If True, apply spike smoothing to the prices.

        Returns:
            A DataFrame with weekly prices for the given ETF, indexed by date. Columns:
            - Ticker: The ticker symbol of the ETF in YFinance notation.
            - Close: The closing price of the ETF for the week designated by the date.
            - Volume: The weekly trading volume of the ETF.
        """
        etf = self.etfs[self.etfs["ticker"] == ticker].head(1)
        if etf.empty:
            raise ValueError(f"No ETF found with ticker: '{ticker}'")
        prices = self.weekly_prices_dict.get(ticker, None)
        if prices is None or prices.empty:
            raise ValueError(f"No prices found for ticker: '{ticker}'")

        if align:
            prices = prices.copy()
            prices.index = pd.to_datetime(prices.index).to_period("W-SUN").start_time

        if smooth_spikes:
            prices = self._robust_spike_smoothing(prices)

        return prices

    def get_etf_countries(self, ticker):
        """Get the countries associated with a given ETF.

        Arguments:
            ticker: The ticker symbol of the ETF.

        Returns:
            A DataFrame containing the countries associated with the given ticker. The index is the country name.
            Columns:
            - weight: The weight that the ETF invests in the country.
            - isin: The ISIN code of the ETF that invests in the country.
        """
        isin = self.etfs[self.etfs["ticker"] == ticker].head(1)
        if len(isin) == 0:
            raise ValueError(f"No ISIN found for ticker: {ticker}")
        isin = isin.index[0]
        return self.countries[self.countries["isin"] == isin]

    def _robust_spike_smoothing(
        self, prices, spike_threshold=0.2, extreme_spike_threshold=0.5, window_size=5, min_periods=3
    ):
        """Robust method for smoothing price spikes that result from problematic data.

        This method implements multiple techniques:
        1. Statistical outlier detection using z-scores and IQR
        2. Adaptive thresholds based on volatility
        3. Context-aware smoothing that preserves legitimate market movements
        4. Multiple smoothing techniques (rolling mean, median, and exponential)
        5. Exclusion of known volatile periods (e.g., COVID-19)

        Arguments:
            prices: DataFrame with price data
            spike_threshold: Base threshold for identifying spikes (default 20%)
            extreme_spike_threshold: Threshold for extreme spikes requiring different treatment
            window_size: Size of rolling window for smoothing
            min_periods: Minimum periods required for rolling calculations

        Returns:
            DataFrame with smoothed prices

        Notes:
            Generated by ChatGPT :)
        """
        if prices.empty:
            return prices

        prices = prices.copy()
        returns = prices["Close"].pct_change().dropna()

        if returns.empty:
            return prices

        # Define periods to exclude from smoothing (legitimate high volatility periods)
        exclusion_periods = [
            (datetime.datetime(2019, 12, 1), datetime.datetime(2020, 6, 30)),  # COVID-19
            (datetime.datetime(2008, 9, 1), datetime.datetime(2009, 3, 31)),  # Financial Crisis
            (datetime.datetime(2001, 9, 1), datetime.datetime(2001, 12, 31)),  # 9/11 & Dot-com
        ]

        # 1. Statistical outlier detection using multiple methods
        spikes_mask = self._detect_statistical_outliers(returns, spike_threshold)

        # 2. Adaptive threshold based on rolling volatility
        rolling_vol = returns.rolling(window=20, min_periods=10).std()
        adaptive_threshold = rolling_vol * 3  # 3 standard deviations
        adaptive_spikes = returns.abs() > adaptive_threshold

        # Combine spike detection methods
        all_spikes = spikes_mask | adaptive_spikes

        # 3. Filter out spikes during exclusion periods
        dt = pd.to_datetime(returns.index)
        exclusion_mask = pd.Series(False, index=returns.index)

        for start, end in exclusion_periods:
            period_mask = (start <= dt) & (dt <= end)
            exclusion_mask = exclusion_mask | period_mask

        # Only smooth spikes that are NOT in exclusion periods
        spikes_to_smooth = all_spikes & ~exclusion_mask
        spike_indices = returns[spikes_to_smooth].index

        if not spike_indices.empty:
            # 4. Apply different smoothing techniques based on spike severity
            for idx in spike_indices:
                spike_magnitude = abs(returns.loc[idx])

                if spike_magnitude > extreme_spike_threshold:
                    # For extreme spikes, use median-based approach (more robust to outliers)
                    smoothed_price = self._median_based_smoothing(prices, idx, window_size)
                else:
                    # For moderate spikes, use adaptive weighted smoothing
                    smoothed_price = self._adaptive_weighted_smoothing(prices, idx, window_size, min_periods)

                # Apply the smoothed price
                prices.loc[idx, "Close"] = smoothed_price

        return prices

    def _detect_statistical_outliers(self, returns, base_threshold):
        """Detect outliers using multiple statistical methods.

        Notes:
            Generated by ChatGPT :)
        """
        # Method 1: Simple threshold
        threshold_spikes = (returns > base_threshold) | (returns < -base_threshold)

        # Method 2: Z-score based detection
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        z_score_spikes = z_scores > 3

        # Method 3: IQR-based detection
        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1
        iqr_spikes = (returns < (q1 - 1.5 * iqr)) | (returns > (q3 + 1.5 * iqr))

        # Combine methods (spike must be detected by at least 2 methods)
        spike_votes = threshold_spikes.astype(int) + z_score_spikes.astype(int) + iqr_spikes.astype(int)
        return spike_votes >= 2

    def _median_based_smoothing(self, prices, spike_idx, window_size):
        """Robust smoothing using median for extreme outliers.

        Notes:
            Generated by ChatGPT :)
        """
        # Get surrounding prices
        idx_pos = prices.index.get_loc(spike_idx)
        start_idx = max(0, idx_pos - window_size // 2)
        end_idx = min(len(prices), idx_pos + window_size // 2 + 1)

        surrounding_prices = prices.iloc[start_idx:end_idx]["Close"]
        # Exclude the spike itself from calculation
        surrounding_prices = surrounding_prices.drop(spike_idx, errors="ignore")

        if len(surrounding_prices) >= 2:
            return surrounding_prices.median()
        else:
            # Fallback to simple rolling mean if not enough data
            return prices["Close"].rolling(window_size, min_periods=1).mean().loc[spike_idx]

    def _adaptive_weighted_smoothing(self, prices, spike_idx, window_size, min_periods):
        """Adaptive weighted smoothing that gives more weight to closer observations.

        Notes:
            Generated by ChatGPT :)
        """
        idx_pos = prices.index.get_loc(spike_idx)
        start_idx = max(0, idx_pos - window_size // 2)
        end_idx = min(len(prices), idx_pos + window_size // 2 + 1)

        surrounding_prices = prices.iloc[start_idx:end_idx]["Close"]

        if len(surrounding_prices) >= min_periods:
            # Create weights that favor closer observations
            center = len(surrounding_prices) // 2
            weights = np.exp(-0.5 * np.square(np.arange(len(surrounding_prices)) - center))

            # Exclude the spike itself
            if spike_idx in surrounding_prices.index:
                spike_pos = surrounding_prices.index.get_loc(spike_idx)
                weights = np.delete(weights, spike_pos)
                surrounding_prices = surrounding_prices.drop(spike_idx)

            if len(surrounding_prices) > 0:
                return np.average(surrounding_prices, weights=weights)

        # Fallback to exponential smoothing
        return prices["Close"].ewm(span=window_size).mean().loc[spike_idx]
