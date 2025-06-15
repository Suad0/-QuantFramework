import numpy as np
import cvxpy as cp
import pandas as pd


class PortfolioOptimizer:
    def optimize(self, df):
        """Optimize portfolio weights using mean-variance optimization."""
        close_cols = [col for col in df.columns if 'Adj Close' in col]
        if not close_cols:
            raise ValueError(f"No 'Adj Close' columns found for optimization. Columns: {df.columns}")

        closes = df[close_cols]
        returns = closes.pct_change().dropna()
        if returns.empty:
            raise ValueError("No valid returns data after processing.")

        mu = returns.mean().values
        Sigma = returns.cov().values
        n = len(mu)
        if n == 0:
            raise ValueError("No assets available for optimization.")

        w = cp.Variable(n)
        objective = cp.Maximize(mu @ w - 0.5 * cp.quad_form(w, Sigma))
        constraints = [cp.sum(w) == 1, w >= 0]
        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError("Optimization failed to converge.")

        weights = pd.Series(w.value, index=[col.split('_')[0] for col in close_cols], name='Weights')
        return weights
