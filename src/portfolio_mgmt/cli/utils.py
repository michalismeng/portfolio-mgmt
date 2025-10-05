import json
from collections.abc import Generator
from contextlib import contextmanager

import pandas as pd

from ..core.nodes import PortfolioNode
from ..core.portfolio import Portfolio


class CLIContext:
    _portfolio: Portfolio | None = None
    _node: PortfolioNode | None = None

    @classmethod
    def set_portfolio(cls, value: Portfolio | None):
        cls._portfolio = value
        cls._node = None

    @classmethod
    def get_portfolio(cls):
        return cls._portfolio

    @classmethod
    def set_node(cls, value: PortfolioNode | None):
        cls._node = value

    @classmethod
    def get_node(cls):
        return cls._node


class BaseCLI:
    @property
    def active_portfolio(self):
        return CLIContext.get_portfolio()

    @active_portfolio.setter
    def active_portfolio(self, value: Portfolio | None):
        CLIContext.set_portfolio(value)

    @property
    def active_node(self):
        return CLIContext.get_node()

    @active_node.setter
    def active_node(self, value: PortfolioNode | None):
        CLIContext.set_node(value)

    @contextmanager
    def active_portfolio_safe(self, error_msg="Active portfolio not set") -> Generator[Portfolio, None, None]:
        if not self.active_portfolio:
            raise ValueError(error_msg)
        yield self.active_portfolio

    @contextmanager
    def active_node_safe(self) -> Generator[PortfolioNode, None, None]:
        with self.active_portfolio_safe():
            if not self.active_node:
                raise ValueError("Active node not set")
            yield self.active_node

    def _print_format(self, df: pd.DataFrame, format="table"):
        if format == "table":
            self._pretty_print_df(df)
        elif format == "json":
            print(json.dumps(df.to_dict(), indent=2))
        else:
            print(f"Unknown format '{format}'. Acceptable values: {'table', 'json'}")


    def _pretty_print_df(self, df):
        with pd.option_context(
            'display.max_colwidth', None,
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', None,
            'display.expand_frame_repr', True
            ):
            print(df)

