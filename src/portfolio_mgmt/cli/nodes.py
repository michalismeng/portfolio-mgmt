from ..core.nodes import NodeType, PortfolioNode, PortfolioNodeETF
from ..core.portfolio import OptimizationMethod
from .utils import BaseCLI


class NodeCommandsBase(BaseCLI):

    def add(self, new_node: PortfolioNode, weight: float | None = None, parent: str | None=None):
        """Add a new node to the portfolio.

        Add a node to the portfolio and optimize it, if the optimization method is not set to manual.

        Args:
            name: The name of the new node to add.
            weight: The weight of the new node. Must be None, if portfolio optimization is not manual.
            parent: The name of the parent node, under which the new node will be created. Leave None to use the
                    currently active node.
        """
        if weight is not None and (weight < 0 or weight > 1):
            raise ValueError(f"Weight must be between 0 and 1 (inclusive). Given weight, {weight}, is outside permitted range.")

        with self.active_portfolio_safe() as p:
            if p.optimization_method != OptimizationMethod.MANUAL and weight is not None:
                raise ValueError(f"Weights can be user-defined only when optimization method is set to manual. Currently set to '{p.optimization_method.value}'.")

            if parent:
                parent_node = p.root.find_by_name(parent)
                if not parent_node:
                    raise ValueError(f"The given parent node '{parent}' was not found in the portfolio.")
            else:
                with self.active_node_safe() as n:
                    parent_node = n

            parent_node.add_child(new_node)

            # Optimize the portfolio if the optimization method is not manual
            if p.optimization_method != OptimizationMethod.MANUAL:
                print("Optimizing portfolio...")
                p.optimize()

            print(f"Node '{new_node.name}' added to the portfolio with weight: {new_node.weight:.0%}.")


class NodeCommands(NodeCommandsBase):

    def add(self, name: str, weight: float | None = None, parent: str | None=None):
        """Add a new node to the portfolio.

        Add a node to the portfolio and optimize it, if the optimization method is not set to manual.

        Args:
            name: The name of the new node to add.
            weight: The weight of the new node. Must be None, if portfolio optimization is not manual.
            parent: The name of the parent node, under which the new node will be created. Leave None to use the
                    currently active node.
        """
        new_node = PortfolioNode(name, weight=weight)
        super().add(new_node, weight=weight, parent=parent)

    def delete(self, path: str | None = None):
        """Remove a node from the portfolio.

        Remove a node from the portfolio and optimize it, if the optimization method is not set to manual.

        Args:
            path: Path to the node to delete.
        """
        with self.active_portfolio_safe() as p:
            if path:
                node = p.root.find_by_path(path)
            else:
                with self.active_node_safe() as n:
                    node = n

        if not node:
            print(f"The given path '{path}' was not found in the portfolio.")
            return

        if node.parent is None:
            print(f"You cannot remove the root node from the portfolio.")
        else:
            node.parent.remove(node.name)

    def use(self, path: str | None = None):
        """Set the currently active node.

        Args:
            path: The path to the node to set as the currently active node. Leave None to use the portfolio root.
        """
        with self.active_portfolio_safe() as p:
            if not path:
                active_node = p.root
            else:
                node = p.root.find_by_path(path)
                if not node:
                    print(f"The given path '{path}' was not found in the portfolio.")
                    return
                else:
                    active_node = node
            self.active_node = active_node
            print(f"Node '{active_node.name}' set as the currently active node.")

    def info(self, name: str | None = None):
        """Get information for a node.

        Args:
            name: The name of the node to get information about. Leave None to use the currently active node.

        """
        with self.active_portfolio_safe() as p:
            if name:
                node = p.root.find_by_name(name)
            else:
                with self.active_node_safe() as n:
                    node = n
        if node:
            if node.type == NodeType.ETF:
                print(f"Node: {node.name}. This is an ETF node.")
            elif node.type == NodeType.NODE:
                print(f"Node: {node.name}")
            elif node.type == NodeType.COUNTRY:
                print(f"Node: {node.name}. This is a Country node.")

            chain = node.chain()
            print(f"Path: {" > ".join([f"{n.name} ({n.portfolio_weight:.0%})" for n in reversed(chain)])}")
            print(f"Level weight {node.weight:.0%}, portfolio weight {node.portfolio_weight:.0%}")

            try:
                returns = node.returns()
                profile = node.risk_reward()
                if profile:
                    print(f"Risk-Reward for {node.name}:")
                    print(f"  Mean: {profile.mu:.2%}")
                    print(f"  Geometric Mean: {profile.g_mu:.2%}")
                    print(f"  Stdev: {profile.stdev:.2%}")
                    print(f"  Sharpe: {profile.sharpe:.2f}")
                    print(f"Calculated based on returns available from {returns.index.min().date()} to"
                          f" {returns.index.max().date()}.")
                else:
                    print(f"No risk-reward profile found for {node.name}.")
            except Exception as e:
                print("Risk-reward profile could not be calculated.")
                pass
        else:
            print(f"The given node '{name}' was not found in the portfolio.")


class EtfCommands(NodeCommandsBase):

    def add(self, ticker: str, weight: float | None = None, parent: str | None=None):
        """Add a new ETF to the portfolio.

        Add an ETF to the portfolio and optimize it, if the optimization method is not set to manual.

        Args:
            ticker: The ticker symbol of the ETF to add.
            weight: The weight of the new ETF node. Must be None, if portfolio optimization is not manual.
            parent: The name of the parent node, under which the new ETF node will be created. Leave None to use the
                    currently active node.
        """
        new_node = PortfolioNodeETF(ticker, weight=weight)
        super().add(new_node, weight=weight, parent=parent)
