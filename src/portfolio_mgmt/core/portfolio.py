"""Module for managing investment portfolios."""

import datetime
from enum import Enum
from typing import Any

from cachetools import cached
from cachetools.keys import hashkey
import numpy as np
import pandas as pd
from pypfopt import DiscreteAllocation

from portfolio_mgmt.core.etf import Etf, Etfs

from .environment import Environment
from .nodes import DictRepresentation, PortfolioNode, PortfolioNodeETF
from .optimization import optimize_formal
from .risk_reward import RiskReward
from .transaction import Transactions

import hashlib
import json

class OptimizationMethod(Enum):
    """The optimization method to use for the ETF weights of the portfolio."""

    PRICE_BASED = "price_based"
    """Use price-based optimization, where the optimal weights are determined
    based on the historical returns of the ETFs."""

    MANUAL = "manual"
    """Use manual optimization, where the optimal weights are set manually by the user."""


class Portfolio:
    """Represents a user's investment portfolio.

    The portfolio holds the desired investments hierarchy in a tree-like form and the transactions that the user has
    performed.

    It provides methods to optimize and compare with the actual holdings, as well as get the current market value.

    All methods respect the current Environment, which can be used to set a specific date range for the calculations,
    for example, to get the market value of the portfolio at a specific date.

    Attributes:
        name: Name of the portfolio.
        root: The root node of the portfolio's investment hierarchy.
        optimization_method: The method used for optimizing the portfolio weights.
        transactions: List of transactions associated with the portfolio.
    """

    _MARKET_VALUE_COLUMNS = [
        "ticker",
        "shares",
        "avg_price",
        "cost_basis",
        "date",
        "price",
        "market_value",
        "current_weight",
        "unrealized_pl",
    ]
    _PORTFOLIO_IMPLEMENT_COLUMNS = [
        "ticker",
        "shares",
        "optimal_weight",
        "allocation_weight",
        "date",
        "price",
        "market_value",
    ]
    _CASH_TICKER = "$CASH"

    def __init__(self, name: str, root: PortfolioNode,
                 optimization_method: OptimizationMethod = OptimizationMethod.PRICE_BASED,
                 transactions: Transactions | None = None):
        """Initialize the Portfolio."""
        self.root = root
        self.name = name
        self.optimization_method = optimization_method
        self.transactions = transactions if transactions is not None else Transactions()
        self.no_trade_zone = (100 / len(self._etf_nodes) / 2 if self._etf_nodes else 0) / 100

    def to_dict(self, repr=DictRepresentation.LITE) -> dict:
        """Convert the portfolio to a dictionary representation.

        Args:
            repr: Representation format. Possible values:
                "lite": Returns only the most necessary information. Use this when storing the object in the database.
                "full": Returns the full information. Use this for detailed views.
                See DictRepresentation enum for details.

        Returns:
            A dictionary representation of the portfolio.
        """
        return {
            "root": self.root.to_dict(repr=repr),
            "name": self.name,
            "optimization_method": self.optimization_method.value,
            "transactions": self.transactions.to_tuple_list(),
        }

    def state_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        """Create a Portfolio instance from a dictionary representation."""
        opt_method = data.get("optimization_method", OptimizationMethod.PRICE_BASED.value)
        if isinstance(opt_method, str):
            opt_method = OptimizationMethod(opt_method)

        return cls(
            root=PortfolioNode.from_dict(data["root"]),
            name=data.get("name", ""),
            optimization_method=opt_method,
            transactions=Transactions.from_list(data.get("transactions", [])),
        )

    @property
    def _etf_nodes(self) -> list[PortfolioNodeETF]:
        """Get a list of all ETF nodes in the portfolio."""
        return self.root.get_etf_nodes()

    @property
    def etfs(self) -> Etfs:
        """Get all ETFs in the portfolio."""
        return self.root.get_etfs()

    @staticmethod
    def _calculate_avg_base_price(transactions: pd.DataFrame):
        """Calculate current position and average base price per ticker.

        transactions: DataFrame with columns:
            - 'ticker': stock symbol
            - 'quantity': positive for buys, negative for sells
            - 'price': transaction price per share

        Returns:
            DataFrame with ['ticker', 'shares', 'cost_basis', 'avg_price']
        """
        results = []

        for ticker, df in transactions.groupby("ticker"):
            total_shares = 0
            total_cost = 0.0

            for _, row in df.iterrows():
                qty = row["quantity"]
                price = row["price"]

                if qty > 0:
                    total_cost += qty * price
                    total_shares += qty
                else:
                    sell_qty = -qty
                    if sell_qty > total_shares:
                        raise ValueError(f"Sell quantity exceeds current holdings for {ticker}")

                    avg_price = total_cost / total_shares
                    total_cost -= sell_qty * avg_price
                    total_shares -= sell_qty

            avg_price = total_cost / total_shares if total_shares > 0 else 0.0

            results.append({"ticker": ticker, "shares": total_shares, "cost_basis": total_cost, "avg_price": avg_price})

        if results:
            return pd.DataFrame(results).set_index("ticker")
        else:
            return pd.DataFrame(results, columns=["ticker", "shares", "cost_basis", "avg_price"]).set_index("ticker")

    @staticmethod
    def _compare_frames(df1: pd.DataFrame, df2: pd.DataFrame, key: str, value: str, value2: str | None = None):
        """Compare two dataframes on a string key column and compute the difference in values.

        Parameters:
            df1, df2: DataFrames
            key: column name (string identifier)
            value: column name
            value2: Optional column name for the second dataframe. If provided, must be different than value.

        Returns:
            DataFrame with [key, diff]
        """
        # This is the case where the 'value' column appears in both dataframes. Use suffixes
        if value2 is None:
            merged = pd.merge(df1, df2, on=key, how="outer", suffixes=("_df1", "_df2")).fillna(
                {f"{value}_df1": 0, f"{value}_df2": 0}
            )
            merged["diff"] = merged[f"{value}_df1"] - merged[f"{value}_df2"]
        else:
            # This is the case where the value columns are different in the two dataframes
            if value == value2:
                raise ValueError("When providing 'value' and 'value2' parameters, they must be different.")

            merged = pd.merge(df1, df2, on=key, how="outer", suffixes=("_df1", "_df2")).fillna({value: 0, value2: 0})
            merged["diff"] = merged[value] - merged[value2]

        return merged[[key, "diff"]].set_index(key)

    def optimize(self):
        """Optimize the portfolio weights based on the selected optimization method.

        Price-based optimization: Use historical returns to determine optimal weights.
        Get the returns of each ETF in the portfolio, perform the formal optimization
        and assign the weights to the ETF nodes. Then propagate those weights upwards
        to the root node. The weights are guaranteed to sum to 100% by the format optimization process.

        Manual optimization: Do nothing, as the weights are already set manually by the user.
        """
        if self.optimization_method == OptimizationMethod.PRICE_BASED:
            etfs = self.root.get_etfs()
            # Calculate mean and std of returns for the whole history for each ETF.
            rrs = [RiskReward.from_weekly_returns(r) for r in etfs.returns]
            stds = [r.stdev for r in rrs]
            gmus = [r.g_mu for r in rrs]
            # Calculate correlation only when we have data from all ETFs
            corr = etfs.returns_df.dropna().corr().values
            if len(gmus) > 0:
                weights, _ = optimize_formal(gmus, stds, corr)
            else:
                weights = dict(zip(range(len(etfs)), np.zeros(len(etfs)), strict=True))

            self.root.clear_weights()

            for node, weight in zip(self.root.get_etf_nodes(), weights.values(), strict=True):
                node.weight = weight

            self.root.propagate_weights_upward()
            self.root.normalize()

    def consolidate_transactions(self):
        """Consolidate transactions for the given portfolio.

        Aggregate buys and sells per ticker to get an overview of current holdings, the average base price
        and the base cost.

        Returns:
            A DataFrame where the index is the ticker and columns are:
                'shares': Total number of shares currently held, after aggregation of buys and sells.
                'cost_basis': The total cost to acquire the shares.
                'avg_price': The average price per share paid.
        """
        df = self.transactions.to_df()
        env = Environment.current()
        if env.start_date is not None:
            df = df[df["date"].dt.date >= env.start_date]
        if env.end_date is not None:
            df = df[df["date"].dt.date <= env.end_date]
        return self._calculate_avg_base_price(df)

    @property
    def mvalue(self) -> float:
        """Calculate the market value of the portfolio including cash."""
        return self.market_value()["market_value"].sum()

    @property
    def cash(self) -> float:
        """Calculate the current cash component of the portfolio.

        The current amount of cash in the portfolio is calculated as the total amount
        of cash added to the portfolio minus the cost basis of all non-cash related transactions.
        """
        consolidated_transactions = self.consolidate_transactions()
        if self._CASH_TICKER not in consolidated_transactions.index:
            cash_tx = 0
        else:
            cash_tx = consolidated_transactions.loc[self._CASH_TICKER, "cost_basis"]
            consolidated_transactions.drop(self._CASH_TICKER, inplace=True)
        return cash_tx - consolidated_transactions["cost_basis"].sum()

    @staticmethod
    def calculation_cache_key(p: 'Portfolio'):
        return hashkey(p.state_hash(), Environment.current().state_hash())

    @cached(cache={}, key=calculation_cache_key)
    def market_value(self):
        """Calculate the market value of each position in the portfolio.

        The market value of the portfolio is its current liquidation value, that is
        the cash amount resulting from selling all open positions.

        The market value includes a cash component (denoted by the $CASH symbol), which is the difference
        between the cash-related transactions and the total cost basis of non-cash transactions.

        Returns:
            A DataFrame containing details about market value of each ETF in the portfolio.
            The index is the ticker symbol.
            The columns are:
                date: The date of the prices based on which the market value is calculated
                shares: Number of shares owned for the position
                cost_basis: Total cost to acquire the shares
                avg_price: Average price paid to acquire the shares
                price: The price used to calculate the market value
                market_value: The market value of the position (shares * price)
                unrealized_pl: The profit or loss not yet realized for this position (market value - cost basis)
                current_weight: The weight of the position relative to all other positions in the portoflio.
                                Useful to compare with the optimal weight for the position.

        Notes:
            To get the market value of the whole portfolio, sum the 'market_value' field of each component.
        """
        consolidated_transactions = self.consolidate_transactions()
        if not len(consolidated_transactions):
            return pd.DataFrame([], columns=self._MARKET_VALUE_COLUMNS).set_index("ticker")

        cash_component = self.cash
        consolidated_transactions.drop(self._CASH_TICKER, inplace=True, errors="ignore")

        mv_data = []
        for ticker, row in consolidated_transactions.iterrows():
            etf = Etf.from_ticker(str(ticker))
            last_price = etf.last_price
            last_date = etf.prices_end_date
            market_value = row["shares"] * last_price
            mv_data.append(
                {
                    "ticker": ticker,
                    "date": last_date,
                    "shares": row["shares"],
                    "cost_basis": row["cost_basis"],
                    "avg_price": row["avg_price"],
                    "price": last_price,
                    "market_value": market_value,
                    "unrealized_pl": market_value - row["cost_basis"],
                }
            )

        # Add cash component as a separate entry (P&L is always 0 for cash)
        mv_data.append(
            {
                "ticker": self._CASH_TICKER,
                "date": max([item["date"] for item in mv_data])
                if mv_data
                else datetime.datetime.now().strftime("%Y-%m-%d"),
                "shares": cash_component,
                "cost_basis": cash_component,
                "avg_price": 1,
                "price": 1,
                "market_value": cash_component,
                "unrealized_pl": 0.0,
            }
        )

        for entry in mv_data:
            entry["current_weight"] = entry["market_value"] / sum(entry["market_value"] for entry in mv_data)

        assert set(mv_data[0].keys()) == set(self._MARKET_VALUE_COLUMNS)

        return pd.DataFrame(mv_data, columns=self._MARKET_VALUE_COLUMNS).set_index("ticker")

    @cached(cache={}, key=calculation_cache_key)
    def implement(self, investment_amount: float | None = None) -> tuple[pd.DataFrame, Any]:
        """Implement the portfolio by calculating the discrete allocation of shares to buy.

        Use the assigned weights and instrument prices to determine how many shares of each ETF to buy.
        The allocation is done using a greedy algorithm to maximize the use of the available investment amount.

        Args:
            investment_amount: Optional total amount to be invested. If None, use the portfolio's current market value.

        Returns:
            A tuple containing:
                A DataFrame with ETF buy details. The index is the ticker symbol.
                The leftover amount that couldn't be allocated, due to buying non-fractional shares.
        """
        etfs = self._etf_nodes
        market_value: float = self.market_value()["market_value"].sum()

        if not etfs:
            raise ValueError("Portfolio has no ETFs.")
        if investment_amount is None and market_value is None:
            raise ValueError("Portfolio has no investment amount set and no amount override provided.")

        weights = {etf_node.ticker: etf_node.portfolio_weight for etf_node in etfs}
        prices_dict = {etf_node.ticker: etf_node.etf.last_price for etf_node in etfs}

        amount = investment_amount if investment_amount is not None else market_value

        da = DiscreteAllocation(weights, pd.Series(prices_dict), int(amount))
        buy_list, leftover = da.greedy_portfolio()

        result = []
        for etf_node in etfs:
            ticker = etf_node.ticker
            shares_to_buy = buy_list.get(ticker, 0)
            money_invested = shares_to_buy * prices_dict[ticker]
            allocation_weight = money_invested / amount if amount > 0 else 0

            result.append(
                {
                    "ticker": ticker,
                    "date": etf_node.etf.prices_end_date,
                    "price": prices_dict[ticker],
                    "optimal_weight": weights[ticker],
                    "allocation_weight": allocation_weight,
                    "market_value": money_invested,
                    "shares": shares_to_buy,
                }
            )

        assert set(result[0].keys()) == set(self._PORTFOLIO_IMPLEMENT_COLUMNS)

        return pd.DataFrame(result, columns=self._PORTFOLIO_IMPLEMENT_COLUMNS).set_index("ticker"), leftover

    @cached(cache={}, key=calculation_cache_key)
    def drift(self) -> pd.DataFrame:
        """Calculate the drift between the current holdings and the desired portfolio.

        Portfolio drift occurs when the current allocation weights have drifted away from the optimal ones
        and thus the portfolio needs rebalancing.

        This function calculates detailed metrics about each position's drift based on the market value
        of the portfolio and it's implementation.

        The implementation of the optimal portfolio uses the total market value of the holdings.
        This essentialy assumes a virtual full liquidation of the holdings and then buying the optimal portfolio
        with all the available cash.

        Returns:
            A tuple containing:
                A DataFrame with the drift. The index is the ticker symbol.
        """
        mv = self.market_value()
        buy_list, _ = self.implement()

        # Drop the cash component to avoid showing a drift in cash. The implementation portfolio ideally holds 0 cash.
        mv = mv.drop([self._CASH_TICKER])

        # Temporarily reset the ticker index, because _compare_frames requires columns.
        _buy_list = buy_list.reset_index()
        _mv = mv.reset_index()

        drift = self._compare_frames(_buy_list, _mv, key="ticker", value="shares").rename(
            columns={"diff": "shares_diff"}
        )
        different_weights = self._compare_frames(
            _buy_list, _mv, key="ticker", value="optimal_weight", value2="current_weight"
        ).rename(columns={"diff": "weights_diff"})
        difference_mv = self._compare_frames(_buy_list, _mv, key="ticker", value="market_value").rename(
            columns={"diff": "mvalue_diff"}
        )
        drift = pd.merge(drift, different_weights, how="inner", left_index=True, right_index=True)
        drift = pd.merge(drift, difference_mv, how="inner", left_index=True, right_index=True)

        drift["mvalue_diff"] = drift["mvalue_diff"].abs()

        cash_component = self.market_value().loc[self._CASH_TICKER]["market_value"].sum()
        minimum_trade_value = max(150, cash_component * 0.1 / 100 if cash_component else 0)

        drift["outside_no_trade_zone"] = drift["weights_diff"].apply(lambda w: abs(w) > self.no_trade_zone / 2)
        drift["more_than_minimum_fee"] = drift["mvalue_diff"].apply(lambda mv: mv >= minimum_trade_value)
        drift["can_trade"] = drift["outside_no_trade_zone"] & drift["more_than_minimum_fee"]

        return drift
