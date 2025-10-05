import datetime

from portfolio_mgmt.core.environment import Environment
from portfolio_mgmt.core.nodes import PortfolioNode, PortfolioNodeETF

from .stub_data import PRICES_START_DATE, RETURNS, RETURNS_MEAN, RETURNS_STDEV, TICKER, TICKER_2


def test_etf_node_returns():
    node = PortfolioNodeETF(TICKER, 1)
    assert node.returns().round(3).tolist() == RETURNS
    with Environment.use(Environment.clone(start_date=PRICES_START_DATE + datetime.timedelta(days=7))):
        assert node.returns().round(3).tolist() == RETURNS[1:]


def test_node_returns():
    node = PortfolioNode("TEST", weight=1)
    node.add_child(PortfolioNodeETF(TICKER, 0.5))
    node.add_child(PortfolioNodeETF(TICKER_2, 0.5))

    assert node.returns().round(3).tolist() == RETURNS


def test_node_risk_reward():
    node = PortfolioNode("TEST", weight=1)
    node.add_child(PortfolioNodeETF(TICKER, 0.5))
    node.add_child(PortfolioNodeETF(TICKER_2, 0.5))

    assert round(node.risk_reward().mu, 3) == round(RETURNS_MEAN, 3)
    assert round(node.risk_reward().stdev, 3) == round(RETURNS_STDEV, 3)


def test_node_max_depth():
    node = PortfolioNode("TEST", weight=1)
    node2 = PortfolioNode("TEST_2", weight=1)
    node3 = PortfolioNode("TEST_3", weight=1)
    node4 = PortfolioNode("TEST_4", weight=1)
    node.add_child(node2)
    node2.add_child(node3)
    node3.add_child(node4)

    assert node.max_depth() == 3


def test_node_propagate_weights_upwards():
    node = PortfolioNode("TEST", weight=None)

    assert node.weight is None

    # Create 4 nodes, each with 0.25 weight. Ensure it propagates to 1 to the parent.
    for i in range(2, 6):
        child = PortfolioNode(f"TEST_{i}", weight=0.25)
        node.add_child(child)

    node.propagate_weights_upward()
    assert node.weight == 1


def test_node_weight_up_to_now():
    node = PortfolioNode("TEST", weight=1)
    node2 = PortfolioNode("TEST_2", weight=0.5)
    node3 = PortfolioNode("TEST_3", weight=0.25)
    node4 = PortfolioNode("TEST_4", weight=0.5)
    node.add_child(node2)
    node2.add_child(node3)
    node3.add_child(node4)

    assert node.portfolio_weight == 1
    assert node2.portfolio_weight == 0.5
    assert node3.portfolio_weight == 0.125
    assert node4.portfolio_weight == 0.0625


def test_node_rebalance():
    # Set up imbalanced nodes (children weights don't add up to 100%)
    node = PortfolioNode("TEST", weight=1)
    node2 = PortfolioNode("TEST_2", weight=0.1)
    node3 = PortfolioNode("TEST_3", weight=0.1)
    node4 = PortfolioNode("TEST_4", weight=0.1)
    node5 = PortfolioNode("TEST_5", weight=0.1)
    node6 = PortfolioNode("TEST_6", weight=0.1)
    node.add_child(node2)
    node.add_child(node3)
    node.add_child(node4)
    node.add_child(node5)
    node2.add_child(node6)

    node.normalize(recursive=False)

    # Ensure non-recursive rebalance, only affects immediate children
    assert node2.weight == 0.25
    assert node3.weight == 0.25
    assert node4.weight == 0.25
    assert node5.weight == 0.25
    assert node6.weight == 0.1

    node.normalize()

    # Ensure recursive rebalance affects all nodes
    assert node2.weight == 0.25
    assert node3.weight == 0.25
    assert node4.weight == 0.25
    assert node5.weight == 0.25
    assert node6.weight == 1
