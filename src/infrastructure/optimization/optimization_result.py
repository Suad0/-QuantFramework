"""
Portfolio optimization result classes.

This module defines result classes for portfolio optimization operations
including optimization results and rebalancing results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    
    weights: pd.Series
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    success: bool
    method: str
    objective_value: float
    constraints_satisfied: bool
    optimization_time: float
    iterations: int
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate optimization result."""
        if not isinstance(self.weights, pd.Series):
            raise ValueError("Weights must be a pandas Series")
        if len(self.weights) == 0:
            raise ValueError("Weights cannot be empty")
        if not self.method:
            raise ValueError("Method cannot be empty")
    
    @property
    def total_weight(self) -> float:
        """Get total portfolio weight."""
        return float(self.weights.sum())
    
    @property
    def num_assets(self) -> int:
        """Get number of assets in portfolio."""
        return len(self.weights)
    
    @property
    def active_positions(self) -> pd.Series:
        """Get assets with non-zero weights."""
        return self.weights[self.weights.abs() > 1e-6]
    
    @property
    def concentration(self) -> float:
        """Get portfolio concentration (sum of squared weights)."""
        return float(np.sum(self.weights ** 2))
    
    def get_top_holdings(self, n: int = 10) -> pd.Series:
        """Get top N holdings by absolute weight."""
        return self.weights.abs().nlargest(n)
    
    def get_sector_weights(self, sector_mapping: Dict[str, str]) -> pd.Series:
        """Get weights by sector."""
        sector_weights = {}
        for asset, weight in self.weights.items():
            sector = sector_mapping.get(asset, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        return pd.Series(sector_weights).sort_values(ascending=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'weights': self.weights.to_dict(),
            'expected_return': self.expected_return,
            'expected_risk': self.expected_risk,
            'sharpe_ratio': self.sharpe_ratio,
            'success': self.success,
            'method': self.method,
            'objective_value': self.objective_value,
            'constraints_satisfied': self.constraints_satisfied,
            'optimization_time': self.optimization_time,
            'iterations': self.iterations,
            'message': self.message,
            'total_weight': self.total_weight,
            'num_assets': self.num_assets,
            'concentration': self.concentration,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class RebalanceResult:
    """Result of portfolio rebalancing operation."""
    
    trades: pd.DataFrame  # Columns: symbol, current_weight, target_weight, trade_amount, trade_value
    total_turnover: float
    transaction_costs: float
    expected_tracking_error: float
    success: bool
    method: str
    optimization_time: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate rebalance result."""
        if not isinstance(self.trades, pd.DataFrame):
            raise ValueError("Trades must be a pandas DataFrame")
        if not self.method:
            raise ValueError("Method cannot be empty")
        
        required_columns = ['symbol', 'current_weight', 'target_weight', 'trade_amount']
        missing_columns = set(required_columns) - set(self.trades.columns)
        if missing_columns:
            raise ValueError(f"Trades DataFrame missing columns: {missing_columns}")
    
    @property
    def num_trades(self) -> int:
        """Get number of trades required."""
        return len(self.trades[self.trades['trade_amount'].abs() > 1e-6])
    
    @property
    def buy_trades(self) -> pd.DataFrame:
        """Get buy trades only."""
        return self.trades[self.trades['trade_amount'] > 1e-6]
    
    @property
    def sell_trades(self) -> pd.DataFrame:
        """Get sell trades only."""
        return self.trades[self.trades['trade_amount'] < -1e-6]
    
    @property
    def total_buy_value(self) -> float:
        """Get total value of buy trades."""
        return float(self.buy_trades['trade_amount'].sum())
    
    @property
    def total_sell_value(self) -> float:
        """Get total absolute value of sell trades."""
        return float(self.sell_trades['trade_amount'].abs().sum())
    
    def get_largest_trades(self, n: int = 10) -> pd.DataFrame:
        """Get N largest trades by absolute amount."""
        return self.trades.loc[
            self.trades['trade_amount'].abs().nlargest(n).index
        ].copy()
    
    def calculate_implementation_shortfall(self, 
                                         market_impact_model: Optional[callable] = None) -> float:
        """Calculate expected implementation shortfall."""
        if market_impact_model is None:
            # Simple linear market impact model
            def market_impact_model(trade_size: float) -> float:
                return 0.001 * abs(trade_size)  # 10 bps per unit trade size
        
        total_impact = 0.0
        for _, trade in self.trades.iterrows():
            if abs(trade['trade_amount']) > 1e-6:
                impact = market_impact_model(trade['trade_amount'])
                total_impact += impact * abs(trade['trade_amount'])
        
        return total_impact
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'trades': self.trades.to_dict('records'),
            'total_turnover': self.total_turnover,
            'transaction_costs': self.transaction_costs,
            'expected_tracking_error': self.expected_tracking_error,
            'success': self.success,
            'method': self.method,
            'optimization_time': self.optimization_time,
            'message': self.message,
            'num_trades': self.num_trades,
            'total_buy_value': self.total_buy_value,
            'total_sell_value': self.total_sell_value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class BacktestResult:
    """Result of strategy backtesting with optimization."""
    
    portfolio_values: pd.Series
    weights_history: pd.DataFrame
    trades_history: pd.DataFrame
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    optimization_results: List[OptimizationResult]
    rebalance_results: List[RebalanceResult]
    total_transaction_costs: float
    success: bool
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def total_return(self) -> float:
        """Get total portfolio return."""
        if len(self.portfolio_values) < 2:
            return 0.0
        return float((self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1)
    
    @property
    def annualized_return(self) -> float:
        """Get annualized return."""
        return self.performance_metrics.get('annualized_return', 0.0)
    
    @property
    def volatility(self) -> float:
        """Get annualized volatility."""
        return self.performance_metrics.get('volatility', 0.0)
    
    @property
    def sharpe_ratio(self) -> float:
        """Get Sharpe ratio."""
        return self.performance_metrics.get('sharpe_ratio', 0.0)
    
    @property
    def max_drawdown(self) -> float:
        """Get maximum drawdown."""
        return self.performance_metrics.get('max_drawdown', 0.0)
    
    def get_rebalance_frequency(self) -> float:
        """Get average rebalancing frequency (rebalances per year)."""
        if len(self.rebalance_results) < 2:
            return 0.0
        
        time_span = (self.portfolio_values.index[-1] - self.portfolio_values.index[0]).days / 365.25
        return len(self.rebalance_results) / time_span
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_transaction_costs': self.total_transaction_costs,
            'rebalance_frequency': self.get_rebalance_frequency(),
            'num_optimizations': len(self.optimization_results),
            'num_rebalances': len(self.rebalance_results),
            'success': self.success,
            'message': self.message,
            'performance_metrics': self.performance_metrics,
            'risk_metrics': self.risk_metrics,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }