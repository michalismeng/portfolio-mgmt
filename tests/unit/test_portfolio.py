from datetime import date, datetime

import pytest
from dateutil.relativedelta import relativedelta

from portfolio_mgmt.core.environment import Environment
from portfolio_mgmt.core.nodes import PortfolioNode
from portfolio_mgmt.core.portfolio import Portfolio
from portfolio_mgmt.core.transaction import Transaction

from .stub_data import BASE_PRICE, PRICES, PRICES_END_DATE, PRICES_START_DATE, TICKER, TICKER_2


def to_datetime(date: date) -> datetime:
    return datetime.combine(date, datetime.min.time())

def test_transaction_consolidation():
    root = PortfolioNode("ROOT", weight=1)
    portfolio = Portfolio("test", root)

    # Simulate adding transactions
    portfolio.transactions.append(Transaction(TICKER, to_datetime(PRICES_START_DATE + relativedelta(days=-2)), 10,
                                              BASE_PRICE))
    portfolio.transactions.append(Transaction(TICKER, to_datetime(PRICES_START_DATE + relativedelta(days=-1)), 5,
                                              2 * BASE_PRICE))
    portfolio.transactions.append(Transaction(TICKER_2, to_datetime(PRICES_START_DATE + relativedelta(days=1)), 8,
                                              BASE_PRICE))

    consolidated = portfolio.consolidate_transactions()
    assert len(consolidated) == 2  # Two tickers

    assert consolidated.loc[TICKER, "shares"] == 15
    assert consolidated.loc[TICKER, "cost_basis"], 10 * BASE_PRICE + 5 * 2 * BASE_PRICE == 2
    assert round(consolidated.loc[TICKER, "avg_price"], 2) == round((10 * BASE_PRICE + 5 * 2 * BASE_PRICE) / 15, 2)

    assert consolidated.loc[TICKER_2, "shares"] == 8
    assert consolidated.loc[TICKER_2, "cost_basis"] == 8 * BASE_PRICE

    # Simulate sell
    portfolio.transactions.append(Transaction(TICKER_2, to_datetime(PRICES_START_DATE + relativedelta(days=2)), -8,
                                              BASE_PRICE))
    consolidated = portfolio.consolidate_transactions()
    assert len(consolidated) == 2

    assert consolidated.loc[TICKER_2, "shares"] == 0
    assert consolidated.loc[TICKER_2, "cost_basis"] == 0

def test_transaction_consolidation_bounded():
    root = PortfolioNode("ROOT", weight=1)
    portfolio = Portfolio("test", root)

    # Simulate adding transactions
    portfolio.transactions.append(Transaction(portfolio._CASH_TICKER,
                                              to_datetime(PRICES_END_DATE + relativedelta(days=4)),
                                              10 * BASE_PRICE, 1))
    portfolio.transactions.append(Transaction(TICKER, to_datetime(PRICES_END_DATE + relativedelta(days=6)),
                                              8, BASE_PRICE))

    with Environment.use(Environment.clone(end_date=PRICES_END_DATE)):
        assert portfolio.consolidate_transactions().empty
        assert portfolio.mvalue == 0

    with Environment.use(Environment.clone(end_date=PRICES_END_DATE + relativedelta(days=4))):
        assert len(portfolio.consolidate_transactions()) == 1
        assert portfolio.cash == portfolio.mvalue == 10 * BASE_PRICE

    with Environment.use(Environment.clone(end_date=PRICES_END_DATE + relativedelta(days=6))):
        assert len(portfolio.consolidate_transactions()) == 2
        assert portfolio.cash == 10 * BASE_PRICE - 8 * BASE_PRICE


def test_portfolio_cash():
    root = PortfolioNode("ROOT", weight=1)
    portfolio = Portfolio("test", root)

    # Simulate adding transactions
    # Add cash
    portfolio.transactions.append(Transaction(portfolio._CASH_TICKER,
                                              to_datetime(PRICES_START_DATE + relativedelta(days=-2)),
                                              BASE_PRICE * 20, 1))
    assert portfolio.cash == BASE_PRICE * 20
    # Buy stock
    portfolio.transactions.append(Transaction(TICKER,
                                              to_datetime(PRICES_START_DATE + relativedelta(days=-1)), 10,
                                              BASE_PRICE))
    assert portfolio.cash == BASE_PRICE * 20 - 10 * BASE_PRICE
    assert portfolio.mvalue == portfolio.cash + PRICES[-1] * 10

    # Add cash
    portfolio.transactions.append(Transaction(portfolio._CASH_TICKER,
                                              to_datetime(PRICES_END_DATE + relativedelta(days=1)),
                                              BASE_PRICE * 20, 1))
    assert portfolio.cash == BASE_PRICE * 20 - 10 * BASE_PRICE + BASE_PRICE * 20
    assert portfolio.mvalue == portfolio.cash + PRICES[-1] * 10


def test_portfolio_transactions():
    root = PortfolioNode("ROOT", weight=1)
    portfolio = Portfolio("test", root)

    # Simulate adding transactions
    initial_cash = BASE_PRICE * 20
    # Add cash
    portfolio.transactions.append(Transaction(portfolio._CASH_TICKER,
                                              to_datetime(PRICES_START_DATE + relativedelta(days=-2)),
                                              initial_cash, 1))
    assert portfolio.cash == initial_cash
    # Buy stock
    portfolio.transactions.append(Transaction(TICKER, to_datetime(PRICES_START_DATE + relativedelta(days=-1)), 10,
                                              BASE_PRICE))
    assert portfolio.cash == initial_cash - 10 * BASE_PRICE
    assert portfolio.mvalue == portfolio.cash + PRICES[-1] * 10

    # Add cash
    extra_cash = BASE_PRICE * 10
    portfolio.transactions.append(Transaction(portfolio._CASH_TICKER,
                                              to_datetime(PRICES_END_DATE + relativedelta(days=1)),
                                              extra_cash, 1))
    assert portfolio.cash == initial_cash - 10 * BASE_PRICE + extra_cash
    assert portfolio.mvalue == portfolio.cash + PRICES[-1] * 10

    # Sell position
    portfolio.transactions.append(Transaction(TICKER, to_datetime(PRICES_END_DATE + relativedelta(days=2)), -10,
                                              2 * BASE_PRICE))
    assert portfolio.cash == initial_cash + extra_cash


def test_short_sell_fails():
    root = PortfolioNode("ROOT", weight=1)
    portfolio = Portfolio("test", root)

    # Short sell fails when consolidating transactions
    with pytest.raises(ValueError):
        portfolio.transactions.append(Transaction(TICKER, to_datetime(PRICES_START_DATE + relativedelta(days=-1)), -10,
                                                  BASE_PRICE))
        portfolio.consolidate_transactions()
