"""
Portfolio service for the application layer.
Provides interface to portfolio optimization and risk management functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import existing infrastructure components
from ...infrastructure.optimization.portfolio_optimizer import PortfolioOptimizer
from ...infrastructure.risk.risk_manager import RiskManager
from ...infrastructure.risk.var_calculator import VaRCalculator


class PortfolioService:
    """Service for portfolio operations."""
    
    def __init__(self):
        self.optimizer = PortfolioOptimizer()
        self.risk_manager = RiskManager()
        self.var_calculator = VaRCalculator()
    
    def optimize_portfolio(self, data: pd.DataFrame, method: str = 'mean_variance') -> Dict[str, float]:
        """Optimize portfolio weights using specified method."""
        try:
            # Calculate expected returns and covariance matrix from data
            price_columns = [col for col in data.columns if '_Close' in col]
            if not price_columns:
                # Fallback: create equal weights for available symbols
                symbols = self._extract_symbols_from_data(data)
                n_assets = len(symbols)
                return {symbol: 1.0/n_assets for symbol in symbols}
            
            # Calculate returns
            returns_data = data[price_columns].pct_change().dropna()
            expected_returns = returns_data.mean()
            covariance_matrix = returns_data.cov()
            
            # Create constraints with method specification
            constraints = [{'type': 'method', 'value': method}]
            
            # Call optimizer with proper parameters
            result = self.optimizer.optimize(expected_returns, covariance_matrix, constraints)
            
            # Extract weights from result
            if isinstance(result, dict) and 'weights' in result:
                weights = result['weights']
            else:
                weights = result
            
            # Ensure weights is a dictionary with proper symbol names
            if isinstance(weights, pd.Series):
                # Map column names to symbol names
                symbol_weights = {}
                for col, weight in weights.items():
                    symbol = col.replace('_Close', '') if '_Close' in col else col
                    symbol_weights[symbol] = weight
                weights = symbol_weights
            elif isinstance(weights, np.ndarray):
                symbols = [col.replace('_Close', '') for col in price_columns]
                weights = dict(zip(symbols, weights))
            
            return weights
        except Exception as e:
            raise Exception(f"Failed to optimize portfolio: {str(e)}")
    
    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for portfolio returns."""
        try:
            # Simple risk metrics calculation since risk_analyzer is not available
            if returns.empty:
                return {}
            
            # Basic risk metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + returns.mean()) ** 252 - 1
            volatility = returns.std() * (252 ** 0.5)
            
            # Risk metrics
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Additional metrics
            win_rate = (returns > 0).sum() / len(returns)
            
            return {
                'Mean_Return': annualized_return,
                'Volatility': volatility,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown,
                'Win_Rate': win_rate
            }
        except Exception as e:
            raise Exception(f"Failed to calculate risk metrics: {str(e)}")
    
    def calculate_portfolio_returns(self, data: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
        """Calculate portfolio returns given weights."""
        try:
            # Extract return columns from data
            return_columns = [col for col in data.columns if 'return' in col.lower() or 'pct_change' in col.lower()]
            
            if not return_columns:
                # Calculate returns from price data
                price_columns = [col for col in data.columns if 'close' in col.lower() or 'adj close' in col.lower()]
                if price_columns:
                    returns_data = data[price_columns].pct_change().dropna()
                else:
                    raise ValueError("No suitable price or return data found")
            else:
                returns_data = data[return_columns]
            
            # Align weights with available data
            aligned_weights = self._align_weights_with_data(weights, returns_data.columns)
            
            # Calculate weighted returns
            portfolio_returns = (returns_data * aligned_weights).sum(axis=1)
            
            return portfolio_returns
        except Exception as e:
            raise Exception(f"Failed to calculate portfolio returns: {str(e)}")
    
    def rebalance_portfolio(self, current_weights: Dict[str, float], 
                          target_weights: Dict[str, float],
                          transaction_cost: float = 0.001) -> Dict[str, Any]:
        """Calculate rebalancing trades and costs."""
        rebalancing_info = {
            'trades': {},
            'total_turnover': 0.0,
            'transaction_costs': 0.0,
            'net_trades': {}
        }
        
        # Calculate trades needed
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            trade = target_weight - current_weight
            
            if abs(trade) > 1e-6:  # Only include significant trades
                rebalancing_info['trades'][symbol] = trade
                rebalancing_info['total_turnover'] += abs(trade)
        
        # Calculate transaction costs
        rebalancing_info['transaction_costs'] = rebalancing_info['total_turnover'] * transaction_cost
        
        # Net trades (buy/sell amounts)
        buy_amount = sum(trade for trade in rebalancing_info['trades'].values() if trade > 0)
        sell_amount = sum(abs(trade) for trade in rebalancing_info['trades'].values() if trade < 0)
        
        rebalancing_info['net_trades'] = {
            'buy_amount': buy_amount,
            'sell_amount': sell_amount
        }
        
        return rebalancing_info
    
    def validate_weights(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Validate portfolio weights."""
        validation_report = {
            'is_valid': True,
            'issues': [],
            'total_weight': sum(weights.values()),
            'num_positions': len([w for w in weights.values() if abs(w) > 1e-6]),
            'max_weight': max(weights.values()) if weights else 0,
            'min_weight': min(weights.values()) if weights else 0
        }
        
        issues = []
        
        # Check if weights sum to approximately 1
        total_weight = validation_report['total_weight']
        if abs(total_weight - 1.0) > 0.01:
            issues.append(f"Weights sum to {total_weight:.4f}, not 1.0")
            validation_report['is_valid'] = False
        
        # Check for negative weights (if not allowed)
        negative_weights = {k: v for k, v in weights.items() if v < 0}
        if negative_weights:
            issues.append(f"Negative weights found: {negative_weights}")
        
        # Check for excessive concentration
        max_weight = validation_report['max_weight']
        if max_weight > 0.5:
            issues.append(f"High concentration: max weight is {max_weight:.2%}")
        
        # Check for too many small positions
        small_positions = len([w for w in weights.values() if 0 < abs(w) < 0.01])
        if small_positions > len(weights) * 0.3:
            issues.append(f"Many small positions: {small_positions} positions < 1%")
        
        validation_report['issues'] = issues
        return validation_report
    
    def calculate_efficient_frontier(self, data: pd.DataFrame, 
                                   num_portfolios: int = 100) -> Dict[str, Any]:
        """Calculate efficient frontier points."""
        try:
            # Extract return data
            return_columns = [col for col in data.columns if 'return' in col.lower()]
            if not return_columns:
                price_columns = [col for col in data.columns if 'close' in col.lower()]
                if price_columns:
                    returns_data = data[price_columns].pct_change().dropna()
                else:
                    raise ValueError("No suitable return data found")
            else:
                returns_data = data[return_columns]
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            # Generate efficient frontier
            target_returns = np.linspace(expected_returns.min(), expected_returns.max(), num_portfolios)
            efficient_portfolios = []
            
            for target_return in target_returns:
                try:
                    # This is a simplified implementation
                    # In practice, you'd use optimization to find the minimum variance portfolio
                    # for each target return
                    weights = self._optimize_for_target_return(expected_returns, cov_matrix, target_return)
                    portfolio_return = np.dot(weights, expected_returns)
                    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    
                    efficient_portfolios.append({
                        'return': portfolio_return,
                        'risk': portfolio_risk,
                        'sharpe': portfolio_return / portfolio_risk if portfolio_risk > 0 else 0,
                        'weights': dict(zip(returns_data.columns, weights))
                    })
                except:
                    continue
            
            return {
                'portfolios': efficient_portfolios,
                'num_portfolios': len(efficient_portfolios)
            }
        except Exception as e:
            raise Exception(f"Failed to calculate efficient frontier: {str(e)}")
    
    def _extract_symbols_from_data(self, data: pd.DataFrame) -> List[str]:
        """Extract symbol names from data columns."""
        symbols = []
        for col in data.columns:
            # Extract symbol from column names like 'AAPL_Close', 'MSFT_Adj Close', etc.
            if '_' in col:
                symbol = col.split('_')[0]
                if symbol not in symbols:
                    symbols.append(symbol)
        
        if not symbols:
            # Fallback: use column names directly
            symbols = list(data.columns)
        
        return symbols
    
    def _align_weights_with_data(self, weights: Dict[str, float], data_columns: List[str]) -> pd.Series:
        """Align portfolio weights with available data columns."""
        aligned_weights = pd.Series(0.0, index=data_columns)
        
        for col in data_columns:
            # Try to match column with weight keys
            for symbol in weights.keys():
                if symbol in col:
                    aligned_weights[col] = weights[symbol]
                    break
        
        # Normalize weights to sum to 1
        if aligned_weights.sum() > 0:
            aligned_weights = aligned_weights / aligned_weights.sum()
        
        return aligned_weights
    
    def _optimize_for_target_return(self, expected_returns: pd.Series, 
                                  cov_matrix: pd.DataFrame, 
                                  target_return: float) -> np.ndarray:
        """Optimize portfolio for target return (simplified implementation)."""
        # This is a placeholder implementation
        # In practice, you'd use quadratic programming to solve the optimization problem
        
        n_assets = len(expected_returns)
        weights = np.ones(n_assets) / n_assets  # Equal weights as fallback
        
        return weights 
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()