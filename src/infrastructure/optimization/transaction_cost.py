"""
Transaction cost optimization for portfolio rebalancing.

This module implements transaction cost-aware optimization for portfolio rebalancing,
including linear and quadratic cost models, market impact, and timing optimization.
"""

from typing import Dict, Any, Optional, Callable
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp
from decimal import Decimal

from .optimization_result import RebalanceResult
from ...domain.exceptions import OptimizationError


class TransactionCostOptimizer:
    """
    Transaction cost optimizer for portfolio rebalancing.
    
    Supports various transaction cost models:
    - Linear transaction costs (proportional to trade size)
    - Quadratic market impact costs
    - Fixed costs per trade
    - Bid-ask spread costs
    - Custom cost functions
    """
    
    def __init__(self):
        """Initialize transaction cost optimizer."""
        self.default_linear_cost = 0.001  # 10 bps
        self.default_market_impact = 0.0001  # 1 bp per 1% of ADV
        self.default_fixed_cost = 5.0  # $5 per trade
    
    def optimize_rebalancing(
        self, 
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: Decimal,
        cost_params: Optional[Dict[str, Any]] = None
    ) -> RebalanceResult:
        """
        Optimize portfolio rebalancing considering transaction costs.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            cost_params: Transaction cost parameters
            
        Returns:
            RebalanceResult with optimized trades
        """
        try:
            if cost_params is None:
                cost_params = self._get_default_cost_params()
            
            # Determine optimization method
            method = cost_params.get('method', 'linear_cost')
            
            if method == 'linear_cost':
                return self._optimize_linear_cost(
                    current_weights, target_weights, portfolio_value, cost_params
                )
            elif method == 'quadratic_cost':
                return self._optimize_quadratic_cost(
                    current_weights, target_weights, portfolio_value, cost_params
                )
            elif method == 'multi_period':
                return self._optimize_multi_period(
                    current_weights, target_weights, portfolio_value, cost_params
                )
            elif method == 'custom_cost':
                return self._optimize_custom_cost(
                    current_weights, target_weights, portfolio_value, cost_params
                )
            else:
                raise OptimizationError(f"Unknown transaction cost method: {method}")
                
        except Exception as e:
            raise OptimizationError(f"Transaction cost optimization failed: {str(e)}")
    
    def _get_default_cost_params(self) -> Dict[str, Any]:
        """Get default transaction cost parameters."""
        return {
            'method': 'linear_cost',
            'linear_cost_rate': self.default_linear_cost,
            'market_impact_rate': self.default_market_impact,
            'fixed_cost_per_trade': self.default_fixed_cost,
            'min_trade_size': 0.001,  # Minimum 0.1% trade size
            'max_turnover': 1.0  # Maximum 100% turnover
        }
    
    def _optimize_linear_cost(
        self, 
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: Decimal,
        cost_params: Dict[str, Any]
    ) -> RebalanceResult:
        """
        Optimize rebalancing with linear transaction costs.
        
        Linear costs are proportional to the absolute trade size.
        """
        n_assets = len(current_weights)
        linear_cost_rate = cost_params.get('linear_cost_rate', self.default_linear_cost)
        min_trade_size = cost_params.get('min_trade_size', 0.001)
        
        # Use CVXPY for optimization
        w = cp.Variable(n_assets)  # New portfolio weights
        trade_sizes = w - current_weights.values  # Trade sizes
        
        # Objective: minimize tracking error plus transaction costs
        tracking_error = cp.sum_squares(w - target_weights.values)
        transaction_costs = linear_cost_rate * cp.sum(cp.abs(trade_sizes))
        
        # Weight tracking error vs transaction costs
        lambda_te = cost_params.get('tracking_error_weight', 1.0)
        lambda_tc = cost_params.get('transaction_cost_weight', 1.0)
        
        objective = cp.Minimize(lambda_te * tracking_error + lambda_tc * transaction_costs)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # Long-only
        ]
        
        # Minimum trade size constraint (optional)
        if min_trade_size > 0:
            # This is a simplified version - actual implementation would use binary variables
            pass
        
        # Maximum turnover constraint
        max_turnover = cost_params.get('max_turnover', 1.0)
        if max_turnover < 1.0:
            constraints.append(cp.sum(cp.abs(trade_sizes)) <= max_turnover)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise OptimizationError(f"Linear cost optimization failed with status: {problem.status}")
        
        # Extract results
        optimal_weights = pd.Series(w.value, index=current_weights.index)
        trades = optimal_weights - current_weights
        
        return self._create_rebalance_result(
            current_weights, optimal_weights, trades, portfolio_value, 
            'linear_cost', cost_params
        )
    
    def _optimize_quadratic_cost(
        self, 
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: Decimal,
        cost_params: Dict[str, Any]
    ) -> RebalanceResult:
        """
        Optimize rebalancing with quadratic market impact costs.
        
        Quadratic costs model market impact that increases with trade size.
        """
        n_assets = len(current_weights)
        market_impact_rate = cost_params.get('market_impact_rate', self.default_market_impact)
        
        # Use CVXPY for optimization
        w = cp.Variable(n_assets)  # New portfolio weights
        trade_sizes = w - current_weights.values  # Trade sizes
        
        # Objective: minimize tracking error plus quadratic transaction costs
        tracking_error = cp.sum_squares(w - target_weights.values)
        
        # Quadratic market impact costs
        market_impact_costs = market_impact_rate * cp.sum_squares(trade_sizes)
        
        # Linear transaction costs (bid-ask spread)
        linear_cost_rate = cost_params.get('linear_cost_rate', self.default_linear_cost)
        linear_costs = linear_cost_rate * cp.sum(cp.abs(trade_sizes))
        
        # Combined transaction costs
        transaction_costs = market_impact_costs + linear_costs
        
        # Weight tracking error vs transaction costs
        lambda_te = cost_params.get('tracking_error_weight', 1.0)
        lambda_tc = cost_params.get('transaction_cost_weight', 1.0)
        
        objective = cp.Minimize(lambda_te * tracking_error + lambda_tc * transaction_costs)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # Long-only
        ]
        
        # Maximum turnover constraint
        max_turnover = cost_params.get('max_turnover', 1.0)
        if max_turnover < 1.0:
            constraints.append(cp.sum(cp.abs(trade_sizes)) <= max_turnover)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise OptimizationError(f"Quadratic cost optimization failed with status: {problem.status}")
        
        # Extract results
        optimal_weights = pd.Series(w.value, index=current_weights.index)
        trades = optimal_weights - current_weights
        
        return self._create_rebalance_result(
            current_weights, optimal_weights, trades, portfolio_value, 
            'quadratic_cost', cost_params
        )
    
    def _optimize_multi_period(
        self, 
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: Decimal,
        cost_params: Dict[str, Any]
    ) -> RebalanceResult:
        """
        Optimize multi-period rebalancing to minimize total transaction costs.
        
        This method spreads trades over multiple periods to reduce market impact.
        """
        n_periods = cost_params.get('n_periods', 5)
        n_assets = len(current_weights)
        
        # Use CVXPY for multi-period optimization
        # w[t] = portfolio weights at time t
        # trades[t] = trades executed at time t
        w = cp.Variable((n_periods + 1, n_assets))
        trades = cp.Variable((n_periods, n_assets))
        
        # Initial weights constraint
        constraints = [w[0, :] == current_weights.values]
        
        # Weight evolution constraints
        for t in range(n_periods):
            constraints.append(w[t + 1, :] == w[t, :] + trades[t, :])
        
        # Final target constraint (with tolerance)
        target_tolerance = cost_params.get('target_tolerance', 0.01)
        constraints.append(cp.abs(w[n_periods, :] - target_weights.values) <= target_tolerance)
        
        # Weight constraints for each period
        for t in range(n_periods + 1):
            constraints.extend([
                cp.sum(w[t, :]) == 1,  # Weights sum to 1
                w[t, :] >= 0  # Long-only
            ])
        
        # Objective: minimize total transaction costs over all periods
        total_cost = 0
        linear_cost_rate = cost_params.get('linear_cost_rate', self.default_linear_cost)
        market_impact_rate = cost_params.get('market_impact_rate', self.default_market_impact)
        
        for t in range(n_periods):
            # Linear costs
            linear_cost = linear_cost_rate * cp.sum(cp.abs(trades[t, :]))
            
            # Quadratic market impact (decreases over time)
            impact_decay = cost_params.get('impact_decay', 0.8)
            period_impact_rate = market_impact_rate * (impact_decay ** t)
            quadratic_cost = period_impact_rate * cp.sum_squares(trades[t, :])
            
            total_cost += linear_cost + quadratic_cost
        
        objective = cp.Minimize(total_cost)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise OptimizationError(f"Multi-period optimization failed with status: {problem.status}")
        
        # Extract results (aggregate all trades)
        total_trades = np.sum(trades.value, axis=0)
        optimal_weights = current_weights + pd.Series(total_trades, index=current_weights.index)
        
        return self._create_rebalance_result(
            current_weights, optimal_weights, 
            pd.Series(total_trades, index=current_weights.index),
            portfolio_value, 'multi_period', cost_params
        )
    
    def _optimize_custom_cost(
        self, 
        current_weights: pd.Series,
        target_weights: pd.Series,
        portfolio_value: Decimal,
        cost_params: Dict[str, Any]
    ) -> RebalanceResult:
        """
        Optimize rebalancing with custom cost function.
        
        Allows for user-defined cost functions for specialized use cases.
        """
        cost_function = cost_params.get('cost_function')
        if cost_function is None:
            raise OptimizationError("Custom cost function is required for custom_cost method")
        
        def objective_function(weights):
            """
            Objective function for custom cost optimization.
            """
            trades = weights - current_weights.values
            
            # Tracking error
            tracking_error = np.sum((weights - target_weights.values) ** 2)
            
            # Custom transaction costs
            transaction_costs = cost_function(trades, current_weights.values, weights)
            
            # Weight tracking error vs transaction costs
            lambda_te = cost_params.get('tracking_error_weight', 1.0)
            lambda_tc = cost_params.get('transaction_cost_weight', 1.0)
            
            return lambda_te * tracking_error + lambda_tc * transaction_costs
        
        # Constraints
        scipy_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds (long-only)
        bounds = [(0, 1) for _ in range(len(current_weights))]
        
        # Initial guess (current weights)
        x0 = current_weights.values
        
        # Optimize
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise OptimizationError(f"Custom cost optimization failed: {result.message}")
        
        # Extract results
        optimal_weights = pd.Series(result.x, index=current_weights.index)
        trades = optimal_weights - current_weights
        
        return self._create_rebalance_result(
            current_weights, optimal_weights, trades, portfolio_value, 
            'custom_cost', cost_params
        )
    
    def _create_rebalance_result(
        self, 
        current_weights: pd.Series,
        optimal_weights: pd.Series,
        trades: pd.Series,
        portfolio_value: Decimal,
        method: str,
        cost_params: Dict[str, Any]
    ) -> RebalanceResult:
        """Create rebalance result object."""
        # Calculate trade values
        trade_values = trades * float(portfolio_value)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame({
            'symbol': current_weights.index,
            'current_weight': current_weights.values,
            'target_weight': optimal_weights.values,
            'trade_amount': trades.values,
            'trade_value': trade_values.values
        })
        
        # Calculate metrics
        total_turnover = float(np.sum(np.abs(trades)))
        transaction_costs = self._calculate_transaction_costs(trades, cost_params)
        expected_tracking_error = float(np.sqrt(np.sum((optimal_weights - current_weights) ** 2)))
        
        return RebalanceResult(
            trades=trades_df,
            total_turnover=total_turnover,
            transaction_costs=transaction_costs,
            expected_tracking_error=expected_tracking_error,
            success=True,
            method=method,
            optimization_time=0.0,
            message=f"{method} rebalancing optimization completed successfully"
        )
    
    def _calculate_transaction_costs(self, trades: pd.Series, cost_params: Dict[str, Any]) -> float:
        """Calculate total transaction costs for given trades."""
        linear_cost_rate = cost_params.get('linear_cost_rate', self.default_linear_cost)
        market_impact_rate = cost_params.get('market_impact_rate', self.default_market_impact)
        fixed_cost_per_trade = cost_params.get('fixed_cost_per_trade', self.default_fixed_cost)
        
        # Linear costs (bid-ask spread)
        linear_costs = linear_cost_rate * np.sum(np.abs(trades))
        
        # Quadratic market impact costs
        market_impact_costs = market_impact_rate * np.sum(trades ** 2)
        
        # Fixed costs (only for non-zero trades)
        num_trades = np.sum(np.abs(trades) > 1e-6)
        fixed_costs = fixed_cost_per_trade * num_trades
        
        return float(linear_costs + market_impact_costs + fixed_costs)
    
    def estimate_implementation_shortfall(
        self, 
        trades: pd.Series,
        market_data: Optional[pd.DataFrame] = None,
        cost_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Estimate implementation shortfall for given trades.
        
        Implementation shortfall measures the difference between
        the decision price and the final execution price.
        """
        if cost_params is None:
            cost_params = self._get_default_cost_params()
        
        # Simple implementation shortfall model
        # In practice, this would use real market data and more sophisticated models
        
        linear_cost_rate = cost_params.get('linear_cost_rate', self.default_linear_cost)
        market_impact_rate = cost_params.get('market_impact_rate', self.default_market_impact)
        
        # Market impact (temporary and permanent)
        temporary_impact = market_impact_rate * np.sum(np.abs(trades))
        permanent_impact = 0.5 * market_impact_rate * np.sum(np.abs(trades))  # Half of temporary
        
        # Timing risk (simplified as volatility-based)
        timing_risk = 0.001 * np.sum(np.abs(trades))  # 10 bps per unit trade
        
        # Opportunity cost
        opportunity_cost = linear_cost_rate * np.sum(np.abs(trades))
        
        total_shortfall = temporary_impact + permanent_impact + timing_risk + opportunity_cost
        
        return {
            'total_implementation_shortfall': float(total_shortfall),
            'temporary_market_impact': float(temporary_impact),
            'permanent_market_impact': float(permanent_impact),
            'timing_risk': float(timing_risk),
            'opportunity_cost': float(opportunity_cost),
            'total_turnover': float(np.sum(np.abs(trades)))
        }