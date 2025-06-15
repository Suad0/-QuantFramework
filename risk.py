import pandas as pd
import numpy as np


class RiskAnalyzer:
    def calculate_risk(self, returns, weights):
        """Calculate risk metrics (VaR, volatility)."""
        portfolio_returns = returns
        if isinstance(returns, pd.DataFrame):
            portfolio_returns = (returns * weights).sum(axis=1)

        portfolio_returns = portfolio_returns.dropna()
        if portfolio_returns.empty:
            raise ValueError("No valid portfolio returns for risk calculation.")

        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized volatility
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)  # 95% VaR
        return {
            'Volatility': volatility,
            'VaR_95': var_95
        }
