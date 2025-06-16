import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    def optimize(self, df, method='mean_variance'):
        """Optimize portfolio weights using various methods."""
        close_cols = [col for col in df.columns if 'Adj Close' in col]
        if not close_cols:
            # Handle single ticker case
            if 'Adj Close' in df.columns:
                close_cols = ['Adj Close']
            else:
                raise ValueError(f"No 'Adj Close' columns found for optimization. Columns: {list(df.columns)}")

        closes = df[close_cols]
        returns = closes.pct_change().dropna()

        if returns.empty:
            raise ValueError("No valid returns data after processing.")

        if method == 'mean_variance':
            return self._mean_variance_optimization(returns, close_cols)
        elif method == 'equal_weight':
            return self._equal_weight_optimization(close_cols)
        elif method == 'risk_parity':
            return self._risk_parity_optimization(returns, close_cols)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _mean_variance_optimization(self, returns, close_cols):
        """Mean-variance optimization using cvxpy."""
        try:
            mu = returns.mean().values
            Sigma = returns.cov().values
            n = len(mu)

            if n == 0:
                raise ValueError("No assets available for optimization.")

            # Handle single asset case
            if n == 1:
                weights = pd.Series([1.0], index=[self._get_ticker_name(close_cols[0])], name='Weights')
                return weights

            w = cp.Variable(n)
            objective = cp.Maximize(mu @ w - 0.5 * cp.quad_form(w, Sigma))
            constraints = [cp.sum(w) == 1, w >= 0]
            problem = cp.Problem(objective, constraints)

            problem.solve()

            if problem.status != cp.OPTIMAL:
                print("Optimization failed, using equal weights")
                return self._equal_weight_optimization(close_cols)

            weights = pd.Series(w.value,
                                index=[self._get_ticker_name(col) for col in close_cols],
                                name='Weights')
            return weights

        except Exception as e:
            print(f"Mean-variance optimization failed: {e}, using equal weights")
            return self._equal_weight_optimization(close_cols)

    def _equal_weight_optimization(self, close_cols):
        """Equal weight optimization."""
        n = len(close_cols)
        equal_weight = 1.0 / n
        weights = pd.Series([equal_weight] * n,
                            index=[self._get_ticker_name(col) for col in close_cols],
                            name='Weights')
        return weights

    def _risk_parity_optimization(self, returns, close_cols):
        """Risk parity optimization - weight inversely to volatility."""
        try:
            volatilities = returns.std()
            inv_vol = 1 / volatilities
            weights = inv_vol / inv_vol.sum()

            weights.index = [self._get_ticker_name(col) for col in close_cols]
            weights.name = 'Weights'
            return weights

        except Exception as e:
            print(f"Risk parity optimization failed: {e}, using equal weights")
            return self._equal_weight_optimization(close_cols)

    def _get_ticker_name(self, col):
        """Extract ticker name from column name."""
        if '_' in col:
            return col.split('_')[0]
        else:
            return 'Stock'
