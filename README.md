# Portfolio Management Tools

A Python library and CLI for building, managing, and optimizing investment portfolios. Create hierarchical portfolio structures, track transactions, and analyze risk-reward metrics using real market data.

## Features

- **Hierarchical Portfolio Management**: Build multi-level portfolio structures with asset allocation at each level
- **ETF Support**: Integrate with multiple ETFs with automatic price history retrieval and risk metrics
- **Transaction Tracking**: Record and track all buy/sell transactions with cost basis calculations
- **Portfolio Optimization**: Optimize portfolio weights using historical returns or manual allocation
- **Risk-Reward Analysis**: Calculate returns, volatility, Sharpe ratios, and other risk metrics
- **CLI & REPL**: Interactive command-line interface for portfolio operations
- **Data Management**: Built-in CSV data access for ETF metadata and pricing

## Installation

### Requirements

- Python 3.12+
- Dependencies: numpy, pandas, pyportfolioopt

### Setup

```bash
# Clone the repository
git clone git@github.com:michalismeng/portfolio-mgmt.git
cd portfolio-mgmt

# Install the package in development mode
pip install -e .

# (Optional) Install CLI dependencies
pip install -e ".[cli]"
```

## Quick Start

### Using the CLI

Start the interactive REPL:

```bash
portfolio-mgmt
```

You'll see the REPL prompt:

```
Welcome to the Portfolio CLI. Type help or ? to list commands.
>
```

Common commands:

```
# Create a new portfolio
portfolio create "My Portfolio"

# Add nodes (asset classes/categories) to the portfolio
node add Equities 0.8 Root
node add Bonds 0.1 Root

# Add ETFs to nodes
etf add SXR8 1 Equities
etf add EUNU 1 Bonds

# View portfolio structure
portfolio view

# Save portfolio to file
portfolio save my_portfolio.json
```

### Running Command Scripts

Execute a series of commands from a file:

```bash
portfolio-mgmt filename examples/portfolio.txt
```

Or inline:

```bash
portfolio-mgmt "portfolio create MyPort, node add Equities 0.8 Root, etf add SXR8 1 Equities"
```

### Python API

```python
from portfolio_mgmt.core.portfolio import Portfolio, OptimizationMethod
from portfolio_mgmt.core.nodes import PortfolioNode, PortfolioNodeETF
from portfolio_mgmt.core.etf import Etf

# Create a portfolio
root = PortfolioNode(name="Root", allocation=1.0, parent=None)
portfolio = Portfolio(
    name="My Portfolio",
    root=root,
    optimization_method=OptimizationMethod.PRICE_BASED
)

# Add nodes and ETFs, run optimizations
# ... (see example files for detailed usage)
```

## Project Structure

```
src/portfolio_mgmt/
├── core/                 # Core portfolio management logic
│   ├── portfolio.py      # Portfolio class and management
│   ├── nodes.py          # Portfolio node hierarchy
│   ├── etf.py            # ETF representation and data access
│   ├── transaction.py    # Transaction tracking
│   ├── risk_reward.py    # Risk and reward calculations
│   ├── optimization.py   # Portfolio optimization
│   ├── data.py           # Data access interfaces
│   ├── csv_data.py       # CSV-based data access
│   └── environment.py    # Environment and context management
│
└── cli/                  # Command-line interface
    ├── main.py           # REPL entry point
    ├── portfolio.py      # Portfolio CLI commands
    ├── nodes.py          # Node CLI commands
    ├── etf.py            # ETF CLI commands
    ├── transactions.py   # Transaction CLI commands
    └── utils.py          # CLI utilities
```

## Data

Example data files are included in `examples/data/`:

- `countries.csv`: Country reference data
- `etfs.csv`: ETF metadata (ISIN, fund size, TER, etc.)
- `weekly-prices.csv`: Historical weekly price data for ETFs

These are used by the CSV data access layer to populate portfolio data.

## Usage Examples

See `examples/portfolio.txt` for a complete walkthrough:

```plaintext
portfolio create "My Global Portfolio" manual
node add Equities 0.8 Root
node add Bonds 0.1 Root
etf add SXR8 1 Equities
etf add EUNU 1 Bonds
portfolio view
portfolio save my_portfolio.json
```

## Testing

```bash
# Install test dependencies
pip install -e ".[tests]"

# Run tests
pytest tests/
```

## Development

Install development dependencies:

```bash
pip install -e ".[cli,tests,lint]"
```

Run linting:

```bash
ruff check src/ tests/
```
