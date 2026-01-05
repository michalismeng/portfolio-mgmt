"""Module for portfolio optimization using PyPortfolioOpt."""
import numpy as np
from pypfopt import EfficientFrontier, objective_functions


def _correlation_to_covariance(corr_matrix: np.ndarray, std_devs: np.ndarray) -> np.ndarray:
    """Convert a correlation matrix to a covariance matrix.

    Args:
        corr_matrix (np.ndarray): Square correlation matrix (n x n)
        std_devs (np.ndarray): 1D array of standard deviations (length n)

    Returns:
        np.ndarray: Covariance matrix (n x n)
    """
    corr_matrix = np.asarray(corr_matrix)
    std_devs = np.asarray(std_devs)

    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    if corr_matrix.shape[0] != std_devs.shape[0]:
        raise ValueError("Length of std_devs must match the size of the correlation matrix.")

    d = np.diag(std_devs)
    cov_matrix = d @ corr_matrix @ d
    return cov_matrix


def optimize_formal(g_mu: list[float] | np.ndarray, stds: list[float] | np.ndarray, corr: np.ndarray):
    """Perform formal portfolio optimization, maximizing the Sharpe ratio.

    This method uses the PyPortfolioOpt library to perform formal portfolio optimization, maximizing the sharpe ratio.
    In addition, it uses an L2 regularization component to the objective function, in order to prevent negligible
    weights, thus avoiding extreme weights to some extent.

    Args:
        g_mu: A list of the geometric means of the returns of the ETFs.
        stds: A list of the standard deviations of the returns of the ETFs.
        corr: A square correlation matrix of the returns of the ETFs.

    Returns:
        raw_weights: The optimized weights of the ETFs.
        performance: A tuple containing the expected annual return, annual volatility, and Sharpe ratio
    """
    if not isinstance(g_mu, np.ndarray):
        g_mu = np.array(g_mu)
    if not isinstance(stds, np.ndarray):
        stds = np.array(stds)

    s = _correlation_to_covariance(corr, stds)
    ef = EfficientFrontier(g_mu, s)
    # Add gamma to prevent negligible weights
    ef.add_objective(objective_functions.L2_reg, gamma=0.5)
    raw_weights = ef.max_sharpe()
    return raw_weights, ef.portfolio_performance(verbose=False)

