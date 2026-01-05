"""Module defining portfolio nodes and the portfolio tree structure.

This module provides the classes and methods to create and manage a portfolio tree,
including different types of nodes. Currently supported node types are:

- PortfolioNode: A generic node that can have children. This is the basis for all other node types. It's
                 useful for grouping nodes together and representing regions or groups of investments. One
                 cannot invest directly in this node.
- PortfolioNodeETF: A node representing an ETF. This node containts ETF information and is considered
                    directly investable.
- CountryNode: A leaf node representing a country within an ETF. This node is used to represent the
               country-level exposure of an ETF.
"""
from enum import Enum

import numpy as np
import pandas as pd

from .etf import Etf, Etfs
from .risk_reward import RiskReward


class DictRepresentation(Enum):
    """Enumeration of dictionary representation types.

    Used when serializing objects to dictionaries. Available options are:
    - LITE: Minimal information, suitable for overviews and database storage. No redundant information is included.
    - FULL: Full information, suitable for detailed views and API responses.
    """
    LITE = "lite"
    FULL = "full"


class NodeType(Enum):
    """Enumeration of node types."""
    NODE = "node"
    ETF = "etf"
    COUNTRY = "country"


class PortfolioNodeList(list['PortfolioNode']):
    """A list of portfolio nodes with additional methods for portfolio calculations."""

    def returns(self) -> 'list[pd.Series[float]]':
        """Get the weighted returns list of the node collection.

        The weighted returns list is the returns of each node in the list multiplied by the node's assigned weight.
        This method is useful to calculate the returns of a parent node, as the sum of the weighted returns.

        Returns:
            A list of the weighted returns, one series for each of the nodes in the list, in order they were inserted.
            If a node doesn't have a weight assigned, NaN values are returned for that node.
        """
        return [node.returns() * (node.weight if node.weight else np.nan) for node in self]

    def normalize(self, target_weight=1.0, recursive=True):
        """Normalize all node weights to sum to the given target weight.

        Args:
            target_weight: The target weight to normalize to. Default is 1.0 (100%).
            recursive: Set to True to normalize all nodes recursively. Set to False to normalize only the current level.
                       Default is True.

        Notes:
            See PortfolioNode.normalize() for more details.
        """
        total_weight = sum(node.weight for node in self if node.weight is not None)
        for node in self:
            if node.weight is not None and total_weight > 0:
                node.weight *= target_weight / total_weight
            if recursive:
                node.normalize(target_weight)


class PortfolioNode:
    """Base class for portfolio nodes.

    This class represents a node in a portfolio tree structure.
    Each node can have children, and it can be a leaf node or a non-leaf node.

    All calculations respect the current environment, for example, the start and end dates.
    """

    def __init__(self, name: str, weight: float | None=0, parent: 'PortfolioNode | None'=None):
        """Initialize a portfolio node.

        Arguments:
            name: Name of the node.
            weight: Weight of the node in the portfolio.
            parent: Parent node.
            level: Level of the node in the hierarchy.
        """
        self._name = name
        self.weight = weight
        self.children = PortfolioNodeList([])
        self.parent = parent

    @property
    def type(self) -> NodeType:
        """Get the type of the node.

        Subclasses should override this property to return the correct type. Using a type allows serialization
        and deserialization of nodes.

        Returns:
            The type of the node.
        """
        return NodeType.NODE

    def add_child(self, node: 'PortfolioNode'):
        """Add a child to this node.

        Before adding, checks are performed to ensure the child is a PortfolioNode and that this node can accept
        children.

        Args:
            node: The child node to add.

        Notes:
            No normalization of weights is performed when adding a child. Use the normalize() method to adjust weights
            after adding or removing children.
        """
        if not isinstance(node, PortfolioNode):
            raise ValueError("Child must be a PortfolioNode")
        if not self.can_add_child:
            raise ValueError("Cannot add child to this node")
        self.children.append(node)
        node.parent = self
        return node

    def remove(self, name: str) -> 'PortfolioNode | None':
        """Remove a child node by name.

        Args:
            name: The name of the child node to remove.

        Returns:
            The removed node, or None if not found.

        Notes:
            No normalization of weights is performed when removing a child. Use the normalize() method to adjust weights
            after adding or removing children.
        """
        for i, child in enumerate(self.children):
            if child.name == name:
                del self.children[i]
                return child
            else:
                # If child has children, recursively search them
                c = child.remove(name)
                if c:
                    return c
        return None

    def normalize(self, target_weight=1.0, recursive=True):
        """Normalize node so all child weights sum to target_weight.

        This method is useful to ensure weights add up to 100% at a given level of the portfolio tree or
        recursively across all levels. If weights do not add up to 100%, any calculations performed on the portfolio
        will be invalid.

        Args:
            target_weight: The target weight to normalize to. Default is 1.0 (100%).
            recursive: Set to True to normalize all nodes recursively. Set to False to normalize only the current level.
                       Default is True.
        """
        self.children.normalize(target_weight=target_weight, recursive=recursive)

    def returns(self, ignore_empty=False) -> 'pd.Series[float]':
        """Get the returns of this portfolio node considering the current environment's start and end dates.

        The returns on any date, D, are calculated as the weighted sum of the returns of each child node on that date.
        Returns of children are concatenated and truncated to the intersection of all dates, so that all children have
        non-NaN data when the calculation is performed. NaN values are not allowed in the returns series.

        Currently, returns are weekly and based on the prices at close. Using weekly returns is important to
        avoid correlation issues that arise with shorter timeframes and products trading in different timezones.

        Args:
            ignore_empty: Set to true to ignore children without weights. This is useful, if the portfolio isn't full,
                          meaning it has nodes without leaf ETFs. In that case, the returns of the whole portfolio
                          would be empty.

        Returns:
            A series of the weekly returns of the node calculated as the weighted sum of the returns of its children.
        """
        weighted_returns = [w for w in self.children.returns() if w.any()] if ignore_empty else self.children.returns()
        weighted_returns = (pd.concat(weighted_returns, axis=1).dropna().sum(axis=1) if weighted_returns
                                                                                     else pd.Series(dtype=float))
        return weighted_returns

    def risk_reward(self) -> RiskReward:
        """Get the annualized risk-reward profile of the ETF considering the current environment.

        The risk-reward profile is calculated based on the returns of the node and represents metrics such as
        the Sharpe ratio, volatility, and annual return.

        Returns:
            The risk and reward profile of the node.
        """
        return RiskReward.from_weekly_returns(self.returns())

    def is_leaf(self) -> bool:
        """Return whether the node is a leaf node (has no children)."""
        return not self.children

    def chain(self) -> 'list[PortfolioNode]':
        """Get the chain of nodes from the current one to the root.

        Returns:
            A list of the chain of nodes from the current node to the root in this order. The list contains at least
            one node, the current one.
        """
        chain: list[PortfolioNode] = []
        temp = self
        while temp is not None:
            chain.append(temp)
            temp = temp.parent
        return chain

    @property
    def name(self):
        """Get the name of the node."""
        return self._name

    def full_name(self):
        """Get the full name of the node, including weight if available."""
        return f"{self.name} {self.weight:.0%}" if self.weight is not None else self.name

    @property
    def can_add_child(self):
        """Return whether this node can have children."""
        return True

    @property
    def portfolio_weight(self) -> float | None:
        """Get the portfolio-level weight of the node.

        The portfolio weight is the weight of the node with respect to the whole portfolio.
        It expresses the exposure/invested amount of the whole portfolio to the given node and it's underlying assets.
        For example, if a node has a 20% portfolio weight, it means that 20% of the whole portfolio is exposed to
        the assets under that node.

        This method recursively gets the weights of the parent nodes until the root of the portfolio
        and multiplies them, thus calculating the weight of the given node with respect to the whole
        portfolio.

        Returns:
            The weight of the node with the respect to the whole portfolio.
        """
        if self.parent is None:
            return self.weight
        parent = self.parent.portfolio_weight
        return self.weight * parent if self.weight is not None and parent is not None else None

    def _to_paths_fixed_depth(self, max_depth, weight_up_to_now: float | None=1.0, path=None):
        """Internal method to get all paths from this node to its leaf nodes, padded to max_depth."""
        path = path or []
        full_path = path + [self.full_name()]

        current_weight = None if weight_up_to_now is None or self.weight is None else weight_up_to_now * self.weight
        if self.is_leaf():
            padded_path = full_path + [None] * (max_depth - len(full_path))
            return [(padded_path, current_weight)]

        paths = []
        for child in self.children:
            paths.extend(child._to_paths_fixed_depth(max_depth, current_weight, full_path))
        return paths

    def to_pd(self) -> pd.DataFrame:
        """Convert the portfolio tree to a pandas DataFrame.

        A multi-index DataFrame is returned, where each level of the index corresponds to a level in the portfolio
        tree. The DataFrame contains the weights of each node relative to the whole portfolio, i.e., weights are
        multiplied at each level.

        Returns:
            A DataFrame representing the portfolio tree with weights.
        """
        max_depth = self.max_depth()
        paths_with_weights = self._to_paths_fixed_depth(max_depth)
        index = pd.MultiIndex.from_tuples(map(lambda x: x[0], paths_with_weights))
        df = pd.DataFrame({
            "Weight": [p[1] for p in paths_with_weights]
        }, index=index)
        return df.sort_index()

    def max_depth(self, depth=0) -> int:
        """Get the maximum depth starting from this node.

        The depth of a node is defined as the number of edges from the node to the deepest leaf node.
        Call this method on the root node to get the total depth of the portfolio tree.

        By default, the depth is zero-based. Use the 'depth' argument to set the starting depth, N, and get a N-based
        depth.

        Args:
            depth: The current depth in the recursion. Default is 0.

        Returns:
            The maximum depth of the tree starting from this node.
        """
        if self.is_leaf():
            return depth
        return max(child.max_depth(depth + 1) for child in self.children)

    def find_by_name(self, name: str) -> 'PortfolioNode | None':
        """Find a node by its name in the subtree rooted at this node.

        If there are multiple nodes with the given name, find the first one.

        Args:
            name: The name of the node to find.

        Returns:
            The first node with the given name, or None if not found.
        """
        if self.name == name:
            return self
        for child in self.children:
            found = child.find_by_name(name)
            if found:
                return found
        return None

    def find_by_path(self, path: str) -> 'PortfolioNode | None':
        """Find a node by path in the subtree rooted at this node.

        Args:
            path: A path of names to look for. The path is of the form: 'Name1 > Name2 > Name3 ...'.

        Returns:
            The node at the given path, or None if not found.
        """
        items = [p.strip() for p in path.split(">")]

        if len(items) == 0:
            return None
        if items[0] != self.name:
            return None
        else:
            if len(items) == 1:
                return self
            for child in self.children:
                found = child.find_by_path(">".join(items[1:]))
                if found:
                    return found
        return None

    def propagate_weights_upward(self):
        """Propagate child weights to this node.

        Sum the children weights and assign this to the current node's weight.
        Assumes on the portfolio scale that the leaf node weights sum up to 100%.

        This method is useful after setting the leaf node weights to their optimal values
        and then propagating these weights upward in the portfolio tree (in a bottom-up fashion).
        """
        if any(child.weight is None for child in self.children):
            raise ValueError("All children must have assigned weights to propagate them")

        for child in self.children:
            child.propagate_weights_upward()

        # Assign weight only when the node has children. Otherwise the sum will be 0.
        if not self.is_leaf():
            self.weight = sum(child.weight for child in self.children) # type: ignore - child weights checked above

    def get_etfs(self) -> Etfs:
        """Get all ETFs from the ETF nodes that are descendants of this node."""
        etf_nodes = self._get_etf_nodes([])
        return Etfs([node.etf for node in etf_nodes])

    def get_etf_nodes(self) -> list['PortfolioNodeETF']:
        """Get all ETF nodes that are descendants of this node."""
        return self._get_etf_nodes([])

    def _get_etf_nodes(self, result: list['PortfolioNodeETF']) -> list['PortfolioNodeETF']:
        """Helper method to recursively collect all ETF nodes."""
        for child in self.children:
            child._get_etf_nodes(result)
        return result

    def clear_weights(self):
        """Set all weights in this subtree to zero."""
        self.weight = 0.0
        for child in self.children:
            child.clear_weights()

    def to_dict(self, repr=DictRepresentation.LITE) -> dict:
        """Convert the node to a dictionary representation.

        Args:
            repr: The representation format. See DictRepresentation for available options.

        Returns:
            A dictionary representation of the node.
        """
        basic = {
            "name": self.name,
            "weight": self.weight,
            "type": self.type.value,
            "children": [child.to_dict(repr) for child in self.children]
        }

        if repr == DictRepresentation.FULL and self.returns().any():
            basic["risk_reward"] = self.risk_reward().to_dict()
        return basic

    @staticmethod
    def from_dict(data: dict) -> 'PortfolioNode':
        """Create a PortfolioNode (or subclass) from a dictionary representation."""
        if data["type"] == "node":
            node = PortfolioNode(name=data["name"], weight=data.get("weight"))
        elif data["type"] == "etf":
            node = PortfolioNodeETF(ticker=data["ticker"], weight=data.get("weight"))
        else:
            valid_types = [m.value for m in NodeType]
            raise ValueError(f"Unexpected portfolio node type '{data["type"]}'."
                             f" Supported types are: {', '.join(valid_types)}")

        for child_data in data.get("children", []):
            node.add_child(PortfolioNode.from_dict(child_data))
        return node


class PortfolioNodeETF(PortfolioNode):
    """A portfolio node representing an ETF."""

    def __init__(self, ticker: str, weight: float | None=None, parent: PortfolioNode | None=None):
        """Initialize an ETF portfolio node.

        Args:
            ticker: The ticker symbol of the ETF. This is used as the id of the node.
            weight: The weight of the ETF in the portfolio.
            parent: The parent node of this ETF node.
        """
        self.etf = self._etf_from_ticker(ticker)
        super().__init__(ticker, weight=weight, parent=parent)
        self.ticker = ticker
        self.countries = self.etf.countries
        if self.countries.empty:
            # Alternatives don't have countries, so we create a default one
            self.countries = pd.DataFrame(columns=["country", "weight", "isin"]).set_index("country")
            self.countries.loc["Other"] = [1, None]
        self._set_countries(self.countries)

    def _etf_from_ticker(self, ticker: str):
        return Etf.from_ticker(ticker)

    def is_leaf(self):
        """Return whether the node is a leaf node."""
        return True

    def _set_countries(self, countries: pd.DataFrame):
        for country, weight, _ in countries.itertuples(index=True):
            child = CountryNode(country, weight=weight, parent=self)
            self.children.append(child)

    def to_dict(self, repr=DictRepresentation.LITE) -> dict:
        """Convert the ETF node to a dictionary representation.

        Args:
            repr: The representation format. See DictRepresentation for available options.

        Returns:
            A dictionary representation of the ETF node.
        """
        basic = super().to_dict(repr=repr)

        if repr == DictRepresentation.LITE:
            basic["children"] = []

        basic["ticker"] = self.ticker

        if repr == DictRepresentation.FULL:
            basic["etf"] = self.etf.to_dict()
        return basic

    def _get_etf_nodes(self, result: list['PortfolioNodeETF']) -> list['PortfolioNodeETF']:
        result.append(self)
        return result

    def full_name(self):
        """Get the full name of the ETF node, truncated to 35 characters."""
        return f"{self._name[:35]} ({self.ticker}): {self.weight:.0%}"

    @property
    def can_add_child(self):
        """Return whether this node can have children."""
        return False

    @property
    def type(self) -> NodeType:
        """Get the type of the node."""
        return NodeType.ETF

    def _to_paths_fixed_depth(self, max_depth, weight_up_to_now=1.0, path=None):
        path = path or []

        padded_path = path + [None] * (max_depth - len(path) - 1)
        full_path = padded_path + [self.full_name()]
        current_weight = None if weight_up_to_now is None or self.weight is None else weight_up_to_now * self.weight

        paths = []
        for child in self.children:
            paths.extend(child._to_paths_fixed_depth(max_depth, current_weight, full_path))
        return paths

    def returns(self, ignore_empty=False):
        """Get the returns of this node considering the current environment's start and end date.

        The returns of an ETF node are simply the returns of the underlying ETF.

        Returns:
            The returns of the ETF node.
        """
        return self.etf.returns

    def clear_weights(self):
        """Weights of country children cannot be cleared."""
        self.weight: float = 0

class CountryNode(PortfolioNode):
    """A leaf node representing a country within an ETF."""
    @property
    def type(self) -> NodeType:
        """Get the type of the node."""
        return NodeType.COUNTRY
