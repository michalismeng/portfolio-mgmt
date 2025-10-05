"""Module for calculating risk and reward metrics for a portfolio."""
import math

import pandas as pd

from .constants import WEEKS_IN_YEAR


class RiskReward:
    """Class to hold risk and reward metrics for a portfolio.

    The main metric is the sharpe ratio, which expresses the risk-adjusted return of the portfolio.
    It is calculated as the geometric mean return divided by the standard deviation of returns.

    Attributes:
        mu (float): Arithmetic mean return of the portfolio (annualized).
        g_mu (float): Geometric mean return of the portfolio (annualized).
        stdev (float): Standard deviation of returns (annualized).
        sharpe (float): Sharpe ratio of the portfolio.
    """
    def __init__(self, mu: float, stdev: float, g_mu: float | None=None):
        """Initialize the RiskReward object.

        The geometric mean return can be optionally provided. If not provided, it is calculated
        using an approximation based on the arithmetic mean and standard deviation.

        All statistics are expected to be annualized.
        """
        self.mu = mu
        self.stdev = stdev
        self.g_mu = g_mu if g_mu is not None else self.mu - 0.5 * self.stdev ** 2

        if self.stdev == 0:
            raise ValueError("Standard deviation cannot be zero.")

        self.sharpe = self.g_mu / self.stdev

    def to_dict(self):
        """Convert the RiskReward object to a dictionary."""
        return {
            "mu": self.mu,
            "g_mu": self.g_mu,
            "stdev": self.stdev,
            "sharpe": self.sharpe,
        }

    @classmethod
    def from_weekly_returns(cls, returns: 'pd.Series[float]') -> 'RiskReward':
        """Create a RiskReward object from a series of weekly returns.

        Args:
            returns (pd.Series[float]): Series of weekly returns.

        **Important:**
            No check is made to ensure the returns are weekly. The caller is responsible
            for ensuring this and if the returns are not weekly, calculations will be incorrect.

        Returns:
            RiskReward: RiskReward object with calculated metrics.
        """
        mu = returns.mean() * WEEKS_IN_YEAR
        stdev = returns.std() * math.sqrt(WEEKS_IN_YEAR)
        g_mu_weekly = pd.to_numeric((returns + 1).prod()) ** (1 / len(returns)) - 1
        g_mu = (1 + g_mu_weekly) ** WEEKS_IN_YEAR - 1
        return RiskReward(mu=mu, g_mu=g_mu, stdev=stdev)

