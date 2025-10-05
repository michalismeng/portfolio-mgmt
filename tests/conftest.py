import pytest

from portfolio_mgmt.core.environment import Environment

from .unit.stub_data import StubDataAccess


@pytest.fixture(scope="session", autouse=True)
def stub_data_access():
    """Session-scoped fixture that replaces engine.lib.etf.DataAccess with StubDataAccess.

    This runs once for the whole pytest session (autouse) so tests get the stubbed
    DataAccess by default. The original class is restored after the session ends.
    """
    Environment.push(env=Environment(data_access=StubDataAccess))
    yield StubDataAccess

