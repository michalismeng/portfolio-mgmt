import json
import os
from datetime import datetime

import pandas as pd

from ..core.environment import Environment
from ..core.nodes import PortfolioNode
from ..core.portfolio import OptimizationMethod, Portfolio
from .utils import BaseCLI


class PortfolioCommands(BaseCLI):
    PORTFOLIO_ERROR_MSG = "No active portfolio. Please load a portfolio first or create a new one."

    def _str_to_datetime(self, value: str):
        return datetime.strptime(value, "%Y-%m-%d")

    def create(self, name: str, opt_method: str = OptimizationMethod.PRICE_BASED.value):
        """Create a new portfolio.

        Arguments:
            name: Name of the portfolio
            opt_method: Optimization method to use
        """
        valid_methods = [m.value for m in OptimizationMethod]
        if opt_method not in valid_methods:
            print(f"Invalid optimization method '{opt_method}'. Available methods: {', '.join(valid_methods)}")
            return

        root = PortfolioNode("Root", 1)
        self.active_portfolio = Portfolio(name, root, OptimizationMethod(opt_method))
        print(f"Portfolio {name} created")

    def save(self, filename: str):
        """Save the portfolio to a file as json."""
        with self.active_portfolio_safe(self.PORTFOLIO_ERROR_MSG) as p:
            with open(os.path.abspath(filename), "w") as f:
                json.dump(p.to_dict(), f, default=str)

    def load(self, filename: str):
        """Load the portfolio from a json file."""
        with open(os.path.abspath(filename)) as f:
            data = json.loads(f.read())
        self.active_portfolio = Portfolio.from_dict(data)

    def view(self, format="table"):
        """View the portfolio tree."""
        with self.active_portfolio_safe(self.PORTFOLIO_ERROR_MSG) as p:
            self._print_format(p.root.to_pd(), format)

    def set_opt_method(self, method: OptimizationMethod | str):
        """Set the optimization method for the portfolio."""
        if isinstance(method, str):
            valid_methods = [m.value for m in OptimizationMethod]
            if method not in valid_methods:
                raise ValueError(f"Invalid optimization method '{method}'. Available methods: {', '.join(valid_methods)}")
            method = OptimizationMethod(method)

        with self.active_portfolio_safe(self.PORTFOLIO_ERROR_MSG) as p:
            p.optimization_method = method
            print(f"Optimization method set to '{method.value}'.")

    def normalize(self):
        """Normalize portfolio weights to sum to 100%."""
        with self.active_portfolio_safe(self.PORTFOLIO_ERROR_MSG) as p:
            p.root.normalize()
            print("Portfolio normalized.")

    def optimize(self):
        """Optimize weights for the leaf ETFs."""
        with self.active_portfolio_safe(self.PORTFOLIO_ERROR_MSG) as p:
            if p.optimization_method == OptimizationMethod.MANUAL:
                print("Manual optimization method is set. Weights are user-defined. Nothing to optimize.")
                return
            elif p.optimization_method == OptimizationMethod.PRICE_BASED:
                p.optimize()

    def info(self):
        """Get information on the portfolio."""
        with self.active_portfolio_safe(self.PORTFOLIO_ERROR_MSG) as p:
            print(f"Portfolio: {p.name}")
            print(f"Optimization method: {p.optimization_method.value}")
            print(f"Number of ETFs: {len(p.etfs)}")
            try:
                returns = p.root.returns()
                profile = p.root.risk_reward()
                if profile:
                    print(f"Risk-Reward for {p.name}:")
                    print(f"  Mean: {profile.mu:.2%}")
                    print(f"  Geometric Mean: {profile.g_mu:.2%}")
                    print(f"  Stdev: {profile.stdev:.2%}")
                    print(f"  Sharpe: {profile.sharpe:.2f}")
                    print(f"Calculated based on returns available from {returns.index.min().date()} to"
                          f" {returns.index.max().date()}.")
                else:
                    print(f"No risk-reward profile found for {p.root.name}.")
            except Exception:
                print("Risk-reward profile could not be calculated.")
                pass

    @staticmethod
    def _fmt_int(v):
        return "" if pd.isna(v) else f"{v:.0f}"

    @staticmethod
    def _fmt_2(v):
        return "" if pd.isna(v) else f"{v:.2f}"

    @staticmethod
    def _fmt_pct2(v):
        return "" if pd.isna(v) else f"{v:.2%}"

    @staticmethod
    def _fmt_date(v):
        """Format dates for display; return empty string for NaN/NaT."""
        if pd.isna(v):
            return ""
        if isinstance(v, (pd.Timestamp, datetime)):
            try:
                return v.strftime("%Y-%m-%d")
            except Exception:
                return str(v)
        return str(v)

    @staticmethod
    def _format_market_value(mv: pd.DataFrame) -> pd.DataFrame:
        """Format market value info for pretty printing."""
        mv = mv.sort_values(by="current_weight", ascending=False)

        mv['shares'] = mv['shares'].apply(PortfolioCommands._fmt_int)
        mv['avg_price'] = mv['avg_price'].apply(PortfolioCommands._fmt_2)
        mv['price'] = mv['price'].apply(PortfolioCommands._fmt_2)
        mv['cost_basis'] = mv['cost_basis'].apply(PortfolioCommands._fmt_2)
        mv['market_value'] = mv['market_value'].apply(PortfolioCommands._fmt_2)
        mv['current_weight'] = mv['current_weight'].apply(PortfolioCommands._fmt_pct2)
        mv['unrealized_pl'] = mv['unrealized_pl'].apply(PortfolioCommands._fmt_2)
        mv["date"] = mv["date"].apply(PortfolioCommands._fmt_date)

        mv = mv.rename(columns={"price": "Price", "shares": "Shares", "avg_price": "Avg Price",
                                "cost_basis": "Cost Basis", "date": "Price Date", "market_value": "Market Value",
                                "current_weight": "Current Weight", "unrealized_pl": "Unrealized P&L"})
        mv.index.name = "Ticker"
        return mv

    @staticmethod
    def _format_buy_list(buy_list: pd.DataFrame) -> pd.DataFrame:
        """Format the buy list for pretty printing."""
        buy_list = buy_list.sort_values(by="optimal_weight", ascending=False)

        buy_list['shares'] = buy_list['shares'].apply(PortfolioCommands._fmt_int)
        buy_list['price'] = buy_list['price'].apply(PortfolioCommands._fmt_2)
        buy_list['market_value'] = buy_list['market_value'].apply(PortfolioCommands._fmt_2)
        buy_list['optimal_weight'] = buy_list['optimal_weight'].apply(PortfolioCommands._fmt_pct2)
        buy_list['allocation_weight'] = buy_list['allocation_weight'].apply(PortfolioCommands._fmt_pct2)
        buy_list['date'] = buy_list['date'].apply(PortfolioCommands._fmt_date)

        buy_list = buy_list.rename(columns={"price": "Price", "shares": "Shares", "date": "Price Date",
                                            "market_value": "Market Value", "optimal_weight": "Optimal Weight",
                                            "allocation_weight": "Allocation Weight"})
        buy_list.index.name = "Ticker"
        return buy_list

    @staticmethod
    def _format_drift(drift: pd.DataFrame) -> pd.DataFrame:
        def format_instruction(shares: float, outside_no_trade_zone: bool, minimum_fee: bool):
            if pd.isna(shares):
                return ""
            if shares == 0:
                return "Do nothing"
            if not outside_no_trade_zone:
                return "Do nothing (inside no trade zone)"
            if not minimum_fee:
                return "Do nothing (trade value below minimum fee)"
            shares_str = f"{abs(shares):.0f} share{'s' if abs(shares) > 1 else ''}"
            return f"Buy {shares_str}" if shares > 0 else f"Sell {shares_str}"

        def format_trade_zone(value: bool | None):
            if pd.isna(value):
                return ""
            return "Outside" if value else "Inside"

        drift["instruction"] = (drift[["shares_diff", "outside_no_trade_zone", "more_than_minimum_fee"]]
                                .apply(lambda x: format_instruction(x.iloc[0], x.iloc[1], x.iloc[2]), axis=1))
        drift["outside_no_trade_zone"] = drift["outside_no_trade_zone"].apply(format_trade_zone)

        drift = drift.sort_values(by="shares_diff", key=abs, ascending=False)
        drift['shares_diff'] = drift['shares_diff'].apply(PortfolioCommands._fmt_int)
        drift['weights_diff'] = drift['weights_diff'].apply(PortfolioCommands._fmt_2)
        drift['mvalue_diff'] = drift['mvalue_diff'].apply(PortfolioCommands._fmt_2)

        drift = drift.rename(columns={"shares_diff": "Shares Diff", "mvalue_diff": "Market Value Diff",
                                      "weights_diff": "Weights Diff", "outside_no_trade_zone": "No Trade Zone",
                                      "more_than_minimum_fee": "Minimum Value Met", "can_trade": "Can Trade",
                                      "instruction": "Instruction"})
        drift.index.name = "Ticker"
        return drift

    def _print_market_value_info(self, mv, when):
        if mv.empty:
            print("No market value data available.")
        else:
            mv = self._format_market_value(mv)
            print(f"Holdings as of {when if when else datetime.now().strftime("%Y-%m-%d")}:")
            print()
            self._pretty_print_df(mv)
            print()
            print(f"Total portfolio cost: {mv['Cost Basis'].astype(float).sum():.2f}")
            print(f"Unrealized P&L: {mv['Unrealized P&L'].astype(float).sum():.2f}")
            print(f"Total cash position: {mv.loc['$CASH', 'Market Value']}")
            print(f"Total market value: {mv['Market Value'].astype(float).sum():.2f}")


    def _print_buy_list(self, buy_list, leftover, when):
        buy_list = self._format_buy_list(buy_list)
        print(f"Portfolio implementation as of {when if when else datetime.now().strftime("%Y-%m-%d")}:")
        print()
        self._pretty_print_df(buy_list)
        print()
        print(f"Total portfolio cost: {buy_list["Market Value"].astype(float).sum():.2f}")
        print(f"Total cash position: {leftover:.2f}")
        print(f"Total market value: {(buy_list["Market Value"].astype(float).sum() + leftover):.2f}")


    def _print_drift(self, drift, when, portfolio: Portfolio):
        transactions_count = len(drift[drift["can_trade"]])
        drift = self._format_drift(drift)
        print(f"Portfolio drift as of {when if when else datetime.now().strftime("%Y-%m-%d")}:")
        print()
        self._pretty_print_df(drift)
        print()
        print(f"To rebalance your holdings you need to perform {transactions_count} transactions.")
        print(f"The total value of the transactions will be {drift[drift["Can Trade"]]["Market Value Diff"].astype(float).abs().sum():.2f}")
        print(f"No trade zone set at: {portfolio.no_trade_zone:.2f} percentage points. Weight diffs larger than {portfolio.no_trade_zone / 2:.2f} in absolute value will generate a trade.")


    def market_value(self, when: datetime | str | None = None):
        """Get a summary of the current market value of the portfolio."""
        if isinstance(when, str):
            when = self._str_to_datetime(when)

        with self.active_portfolio_safe() as p, Environment.use(Environment.clone(end_date=when)):
            mv = p.market_value()
        self._print_market_value_info(mv, when)

    holdings = market_value

    def implement(self, amount: float | None = None, when: datetime | str | None = None):
        """Get the shares to buy for the portfolio assigned weights."""
        if isinstance(when, str):
            when = self._str_to_datetime(when)

        with self.active_portfolio_safe() as p, Environment.use(Environment.clone(end_date=when)):
            buy_list, leftover = p.implement(amount)
        self._print_buy_list(buy_list, leftover, when)


    def drift(self, when: datetime | str | None = None):
        """Calculate current holdings drift from the desired portfolio."""
        if isinstance(when, str):
            when = self._str_to_datetime(when)

        with self.active_portfolio_safe() as p:
            with Environment.use(Environment.clone(end_date=when)):
                drift = p.drift()
            self._print_drift(drift, when, p)