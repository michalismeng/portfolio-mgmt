import datetime

from portfolio_mgmt.core.environment import Environment
from portfolio_mgmt.core.etf import Etf, Etfs

from .stub_data import (
    PRICES,
    PRICES_END_DATE,
    PRICES_START_DATE,
    RETURNS,
    RETURNS_GEO_MEAN,
    RETURNS_MEAN,
    RETURNS_STDEV,
    TICKER,
    TICKER_2,
    TICKER_3,
)


def test_from_ticker():
    etf = Etf.from_ticker(TICKER)
    assert etf.isin == f"ISIN-{TICKER}"
    assert etf.ticker == TICKER
    assert etf.holdings == 100


def test_prices_unbounded():
    etf = Etf.from_ticker(TICKER)
    assert len(etf.prices) == len(PRICES)
    assert etf.prices.tolist() == PRICES


def test_prices_bounded():
    etf = Etf.from_ticker(TICKER)
    with Environment.use(Environment.clone(start_date=PRICES_START_DATE + datetime.timedelta(days=7))):
        assert len(etf.prices) == len(PRICES) - 1
        assert etf.prices_start_date >= PRICES_START_DATE
        assert etf.prices_end_date == PRICES_END_DATE
        assert etf.prices.tolist() == PRICES[1:]
        assert etf.last_price == PRICES[-1]

    with Environment.use(Environment.clone(end_date=PRICES_END_DATE - datetime.timedelta(days=7))):
        assert len(etf.prices) == len(PRICES) - 1
        assert etf.prices_start_date == PRICES_START_DATE
        assert etf.prices_end_date <= PRICES_END_DATE
        assert etf.prices.tolist() == PRICES[:-1]
        assert etf.last_price == PRICES[-2]

    with Environment.use(Environment.clone(start_date=PRICES_START_DATE + datetime.timedelta(days=7),
                                           end_date=PRICES_END_DATE - datetime.timedelta(days=7))):
        assert len(etf.prices) == len(PRICES) - 2
        assert etf.prices_start_date >= PRICES_START_DATE
        assert etf.prices_end_date <= PRICES_END_DATE
        assert etf.prices.tolist() == PRICES[1:-1]
        assert etf.last_price == PRICES[-2]


def test_returns_unbounded():
    etf = Etf.from_ticker(TICKER)
    assert len(etf.returns) == len(RETURNS)
    assert etf.returns.round(3).tolist() == RETURNS


def test_returns_bounded():
    etf = Etf.from_ticker(TICKER)
    with Environment.use(Environment.clone(start_date=PRICES_START_DATE + datetime.timedelta(days=7))):
        assert len(etf.returns) == len(RETURNS) - 1
        assert etf.returns.round(3).tolist() == RETURNS[1:]
        assert etf.prices_start_date == PRICES_START_DATE + datetime.timedelta(days=7)

    with Environment.use(Environment.clone(end_date=PRICES_END_DATE - datetime.timedelta(days=7))):
        assert len(etf.returns) == len(RETURNS) - 1
        assert etf.returns.round(3).tolist() == RETURNS[:-1]
        assert etf.prices_end_date == PRICES_END_DATE - datetime.timedelta(days=7)

    with Environment.use(Environment.clone(start_date=PRICES_START_DATE + datetime.timedelta(days=7),
                                           end_date=PRICES_END_DATE - datetime.timedelta(days=7))):
        assert len(etf.returns) == len(RETURNS) - 2
        assert etf.returns.round(3).tolist() == RETURNS[1:-1]
        assert etf.prices_start_date == PRICES_START_DATE + datetime.timedelta(days=7)
        assert etf.prices_end_date == PRICES_END_DATE - datetime.timedelta(days=7)


def test_risk_reward():
    etf = Etf.from_ticker(TICKER)
    assert round(etf.risk_reward.mu, 3) == round(RETURNS_MEAN, 3)
    assert round(etf.risk_reward.stdev, 3) == round(RETURNS_STDEV, 3)
    assert round(etf.risk_reward.sharpe, 3) == round(RETURNS_GEO_MEAN / RETURNS_STDEV, 3)


def test_countries():
    etf = Etf.from_ticker(TICKER)
    assert len(etf.countries) == 2
    assert set(etf.countries.index) == set(["Germany", "USA"])


def test_multiple_prices():
    etf1 = Etf.from_ticker(TICKER)
    etf2 = Etf.from_ticker(TICKER_2)
    x = Etfs([etf1, etf2])

    assert set(x.tickers) == set([TICKER, TICKER_2])
    assert len(x.returns) == 2

    assert all(x[1].prices == [2 * p for p in PRICES])
    assert all(x.returns[0].round(3) == RETURNS)
    assert all(x.returns[1].round(3) == RETURNS)

    # Assert same risk-reward profile, since the returns are the same
    assert x[0].risk_reward.sharpe == x[1].risk_reward.sharpe


def test_multiple_prices_unaligned():
    # ETF 3 misses the first two price data points.
    etf1 = Etf.from_ticker(TICKER)
    etf3 = Etf.from_ticker(TICKER_3)

    assert len(etf3.prices) == len(PRICES) - 2
    assert etf3.prices_start_date == etf1.prices_start_date + datetime.timedelta(days=7 * 2)
    assert etf3.prices_end_date == etf1.prices_end_date

    # Ensure the Etfs class doesn't truncate start dates
    x = Etfs([etf1, etf3])

    assert all(x[1].prices == PRICES[2:])
    assert all(x.returns[0].round(3) == RETURNS)
    assert all(x.returns[1].round(3) == RETURNS[2:])

    assert x.returns_df.shape == (len(RETURNS), 2)

