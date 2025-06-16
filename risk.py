import pandas as pd
import numpy as np


class RiskAnalyzer:
    def calculate_risk(self, returns, weights=None):
        """Calculate comprehensive risk metrics."""

        # Handle different input types
        if isinstance(returns, pd.DataFrame) and weights is not None:
            # If we have multiple assets and weights, calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
        elif isinstance(returns, pd.Series):
            # If we already have portfolio returns
            portfolio_returns = returns
        elif isinstance(returns, pd.DataFrame):
            # If we have multiple assets but no weights, use equal weights
            portfolio_returns = returns.mean(axis=1)
        else:
            raise ValueError("Invalid returns input type")

        portfolio_returns = portfolio_returns.dropna()

        if portfolio_returns.empty:
            raise ValueError("No valid portfolio returns for risk calculation.")

        # Calculate risk metrics
        risk_metrics = {}

        # Basic statistics
        risk_metrics['Mean_Return'] = portfolio_returns.mean() * 252  # Annualized
        risk_metrics['Volatility'] = portfolio_returns.std() * np.sqrt(252)  # Annualized

        # Value at Risk
        risk_metrics['VaR_95'] = np.percentile(portfolio_returns, 5) * np.sqrt(252)  # 95% VaR
        risk_metrics['VaR_99'] = np.percentile(portfolio_returns, 1) * np.sqrt(252)  # 99% VaR

        # Expected Shortfall (Conditional VaR)
        var_95_daily = np.percentile(portfolio_returns, 5)
        risk_metrics['CVaR_95'] = portfolio_returns[portfolio_returns <= var_95_daily].mean() * np.sqrt(252)

        # Sharpe Ratio (assuming risk-free rate of 0)
        if risk_metrics['Volatility'] > 0:
            risk_metrics['Sharpe_Ratio'] = risk_metrics['Mean_Return'] / risk_metrics['Volatility']
        else:
            risk_metrics['Sharpe_Ratio'] = 0

        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        risk_metrics['Max_Drawdown'] = drawdown.min()

        # Skewness and Kurtosis
        risk_metrics['Skewness'] = portfolio_returns.skew()
        risk_metrics['Kurtosis'] = portfolio_returns.kurtosis()

        # Downside Deviation (volatility of negative returns)
        negative_returns = portfolio_returns[portfolio_returns < 0]
        if len(negative_returns) > 0:
            risk_metrics['Downside_Deviation'] = negative_returns.std() * np.sqrt(252)
        else:
            risk_metrics['Downside_Deviation'] = 0

        # Sortino Ratio
        if risk_metrics['Downside_Deviation'] > 0:
            risk_metrics['Sortino_Ratio'] = risk_metrics['Mean_Return'] / risk_metrics['Downside_Deviation']
        else:
            risk_metrics['Sortino_Ratio'] = 0

        # Calmar Ratio
        if risk_metrics['Max_Drawdown'] != 0:
            risk_metrics['Calmar_Ratio'] = risk_metrics['Mean_Return'] / abs(risk_metrics['Max_Drawdown'])
        else:
            risk_metrics['Calmar_Ratio'] = 0

        print("Risk Metrics Calculated:")
        for metric, value in risk_metrics.items():
            if isinstance(value, float):
                if 'Ratio' in metric:
                    print(f"{metric}: {value:.3f}")
                else:
                    print(f"{metric}: {value:.4f}")

        return risk_metrics
