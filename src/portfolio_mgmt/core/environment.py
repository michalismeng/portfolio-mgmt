"""Module to manage the environment for portfolio calculations.

The Environment class serves as a central place to store settings used throughout the portfolio management library.
"""
import datetime
from contextlib import contextmanager

from .data import DataAccess


class _MissingType:
    """Sentinel for a missing argument."""
    def __repr__(self):
        return "<MISSING>"


_MISSING = _MissingType()


type Maybe[T] = T | None | _MissingType


class Environment:
    """Class to manage the environment for portfolio calculations.

    Serves as a central place to store settings used throughout the portfolio management library.

    Attributes:
        start_date: The start date for calculations. If set, all calculations will respect d >= start_date.
        end_date: The end date for calculations. If set, all calculations will respect d <= end_date.
        data_access: The class responsible for accessing data, defaulting to DataAccess.

    Changing the start/end dates makes it possible to simulate a portfolio at a specific point in time
    (e.g., get past market value), or to backtest strategies over a specific period.

    Example:
        >>> env = Environment(start_date="2024-01-01", end_date="2024-12-31")
        >>> Environment.push(env)
        >>> Etf.from_ticker("SXR8").prices # Prices from 2024-01-01 to 2024-12-31, despite more data available
    """
    _environments: 'list[Environment]' = []

    def __init__(self, start_date: datetime.date | str | None = None, end_date: datetime.date | str | None = None,
                 data_access: type[DataAccess] | None = None):
        """Initialize the environment with optional start/end dates and the data access class."""
        if isinstance(start_date, str):
            start_date = self._str_to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = self._str_to_datetime(end_date)

        self.start_date = start_date
        self.end_date = end_date

        # Instance to the class responsible for accessing data
        self.data_access = DataAccess if data_access is None else data_access

    @staticmethod
    def _str_to_datetime(date: str):
        return datetime.datetime.strptime(date, "%Y-%m-%d").date()

    @classmethod
    def push(cls, env: 'Environment'):
        """Add a new environment to the stack of active environments.

        Args:
            env (Environment): The environment to add to the stack.
        """
        cls._environments.append(env)

    @classmethod
    def pop(cls) -> 'Environment':
        """Remove and return the most recently added environment from the stack.

        Returns:
            Environment: The environment that was removed from the stack.
        """
        return cls._environments.pop()

    @classmethod
    @contextmanager
    def use(cls, env: 'Environment'):
        """Context manager to set the active environment temporarily.

        This context manager pushes the given environment onto the stack of active environments
        when entering the context and pops it off when exiting, ensuring that the previous environment
        is restored.

        Useful for temporarily changing the environment, for example to calculate the portfolio market value
        at a specific (end) date or for backtesting.

        Args:
            env (Environment): The environment to set as active within the context.

        Yields:
            Environment: The environment that is active within the context.

        Example:
            >>> etf = Etf.from_ticker("SXR8")
            >>> with Environment.use(Environment(end_date="2024-12-31")):
            >>>    # Code here uses the new environment
            >>>    print(etf.returns)   # ... returns until 2024-12-31, even though more data is available
            >>> # Outside the context, the previous environment is restored.
            >>> print(etf.returns)      # ... returns with all available data
        """
        cls.push(env)
        try:
            yield env
        finally:
            cls.pop()

    @classmethod
    def current(cls):
        """Get the currently active environment.

        Environments are managed in a stack, so this returns the most recently added environment.
        If no environment has been set, a default environment is created and returned.

        Returns:
            Environment: The currently active environment.
        """
        if not cls._environments:
            cls._environments.append(cls())
        return cls._environments[-1]

    @classmethod
    def clone(cls, start_date: Maybe[datetime.date | str] = _MISSING, end_date: Maybe[datetime.date | str] = _MISSING,
                   data_access: Maybe[type[DataAccess]] = _MISSING) -> 'Environment':
        """Create a copy of the current environment and patch it with the provided values.

        Args:
            start_date: New start date to set. If _MISSING, the current value is retained.
            end_date: New end date to set. If _MISSING, the current value is retained.
            data_access: New data access class to set. If _MISSING, the current value is retained.

        Returns:
            Environment: A new instance of Environment with patched values.
        """
        current_env = cls.current()
        start_date = start_date if not isinstance(start_date, _MissingType) else current_env.start_date
        end_date = end_date if not isinstance(end_date, _MissingType) else current_env.end_date
        data_access = data_access if not isinstance(data_access, _MissingType) else current_env.data_access

        return Environment(start_date=start_date, end_date=end_date, data_access=data_access)
