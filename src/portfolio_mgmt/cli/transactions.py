from datetime import datetime

from dateutil.parser import parse

from ..core.transaction import Transaction
from .utils import BaseCLI


class TransactionCommands(BaseCLI):
    PORTFOLIO_ERROR_MSG = "No active portfolio. Please load a portfolio first or create a new one."

    def create(self, ticker: str, date: datetime | str, amount: float, price: float):
        """Register a transaction.

        Args:
            ticker: The ticker symbol. For a cash transaction, use the special symbol $CASH.
            date: The date of the transaction. Must be in format yyyy-mm-dd.
            amount: Amount of shares bought.
            price: Price per share.
        """
        with self.active_portfolio_safe(self.PORTFOLIO_ERROR_MSG) as p:
            if isinstance(date, str):
                date = parse(date)
            p.transactions.append(Transaction(ticker, date, amount, price))
            p.transactions.sort(key=lambda tx: tx.date)
            print(f"Transaction registered: {'buy' if amount > 0 else 'sell'} {abs(amount)} of {ticker} at {price} on {date.date()}")

    def list(self, format="table"):
        """List all transactions for the current portfolio."""
        with self.active_portfolio_safe(self.PORTFOLIO_ERROR_MSG) as p:
            transactions = p.transactions.to_df()
            if transactions.empty:
                print("No transactions found for this portfolio.")
            else:
                self._print_format(transactions, format)

    def delete(self, index: int):
        """Delete a transaction by its index: delete_transaction <index>"""
        with self.active_portfolio_safe(self.PORTFOLIO_ERROR_MSG) as p:
            if 0 <= index < len(p.transactions):
                del p.transactions[index]
                print(f"Transaction at index {index} deleted.")
            else:
                print(f"Index {index} out of range. No transaction deleted.")
