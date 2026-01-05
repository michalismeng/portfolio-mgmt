"""Module for managing portfolio transactions."""
from datetime import datetime

import pandas as pd
from dateutil import parser


class Transaction:
    """Represents a single transaction in the portfolio.

    Attributes:
        ticker: Stock symbol of the transaction.
        date: Date of the transaction.
        quantity: Number of shares bought (positive) or sold (negative).
        price: Price per share at which the transaction was executed.
    """

    COLUMNS = ["ticker", "date", "quantity", "price"]

    def __init__(self, ticker: str, date: datetime, quantity: float, price: float):
        """Initialize a Transaction object.

        The quantity is positive for a buy and negative for a sell.

        Args:
            ticker (str): Stock symbol of the transaction.
            date (datetime): Date of the transaction.
            quantity (float): Number of shares bought (positive) or sold (negative).
            price (float): Price per share at which the transaction was executed.

        Notes:
            To register cash additions or withdrawals, either in the form of funds or dividends,
            use a special ticker "$CASH" with the quantity representing the amount of cash added.
        """
        self.ticker = ticker
        self.date = date
        self.quantity = quantity
        self.price = price

        assert set(vars(self).keys()) == set(self.COLUMNS)

    def to_tuple(self) -> tuple[str, datetime, float, float]:
        """Convert the Transaction object to a tuple."""
        return (self.ticker, self.date, self.quantity, self.price)

    @classmethod
    def from_tuple(cls, data: tuple[str, datetime | str, float, float]) -> 'Transaction':
        """Create a Transaction object from a tuple."""
        if len(data) != 4:
            raise ValueError(f"Expected tuple of length 4, got {len(data)}")
        if not isinstance(data[0], str):
            raise ValueError(f"Expected string for ticker, got {type(data[0])}")
        if not isinstance(data[1], datetime) and not isinstance(data[1], str):
            raise ValueError(f"Expected datetime for date, got {type(data[1])}")
        if not isinstance(data[2], (int, float)):
            raise ValueError(f"Expected number for quantity, got {type(data[2])}")
        if not isinstance(data[3], (int, float)):
            raise ValueError(f"Expected number for price, got {type(data[3])}")

        date = parser.parse(data[1]) if isinstance(data[1], str) else data[1]
        return cls(ticker=data[0], date=date, quantity=data[2], price=data[3])


class Transactions(list[Transaction]):
    """A list of transactions with utility methods.

    See the Transaction object for more details.
    """

    def to_df(self) -> pd.DataFrame:
        """Convert the list of transactions to a pandas DataFrame."""
        df = pd.DataFrame([tx.to_tuple() for tx in self], columns=Transaction.COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def to_tuple_list(self) -> list[tuple]:
        """Convert the list of transactions to a list of tuples."""
        return [tx.to_tuple() for tx in self]

    @classmethod
    def from_list(cls, data: list[tuple]) -> 'Transactions':
        """Create a Transactions object from a list of tuples."""
        return cls([Transaction.from_tuple(tx) for tx in data])
