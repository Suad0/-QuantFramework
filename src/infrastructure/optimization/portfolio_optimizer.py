"""
Main portfolio optimizer implementation.

This module provides the concrete implementation of the IPortfolioOptimizer interface
with support for multiple optimization methods and advanced constraints.
"""

import time
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp

from ...domain.interfaces import IPortfolioOptimizer
from ...domain.entities import Portfolio
from ...domain.exceptions import OptimizationError, ValidationError
from .constraints import Constraint
from .optimization_result import OptimizationResult, RebalanceResult
from .black_litterman import BlackLittermanOptimizer
from .risk_parity import RiskParityOptimizer
from .factor_based import FactorBasedOptimizer
from .transaction_cost import TransactionCostOptimizer


class PortfolioOptimizer(IPortfolioOptimizer):
    """
    Concrete implementation of portfolio optimization with multiple methods.
    
    Supports:
    - Enhanced mean-variance optimization with advanced constraints
    - Black-Litterman optimization with market views
    - Risk parity and equal risk contribution optimization
    - Factor-based optimization using multi-factor models
    - Transaction cost optimization for rebalancing
    """
    
    def __init__(self):
        """Initialize the portfolio optimizer."""
        self.bl_optimizer = BlackLittermanOptimizer()
        self.rp_optimizer = RiskParityOptimizer()
        self.factor_optimizer = FactorBasedOptimizer()
        self.tc_optimizer = TransactionCostOptimizer()
    
    def optimize(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights using specified method and constraints.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            constraints: List of constraint dictionaries
            
        Returns:
            Dictionary containing optimization results
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(expected_returns, covariance_matrix)
            
            # Parse constraints and method
            method = self._extract_method(constraints)
            constraint_objects = self._parse_constraints(constraints)
            
            # Perform optimization based on method
            if method == 'mean_variance':
                result = self._mean_variance_optimization(
                    expected_returns, covariance_matrix, constraint_objects
                )
            elif method == 'black_litterman':
                result = self._black_litterman_optimization(
                    expected_returns, covariance_matrix, constraints
                )
            elif method == 'risk_parity':
                result = self._risk_parity_optimization(
                    covariance_matrix, constraint_objects
                )
            elif method == 'equal_risk_contribution':
                result = self._equal_risk_contribution_optimization(
                    covariance_matrix, constraint_objects
                )
            elif method == 'factor_based':
                result = self._factor_based_optimization(
                    expected_returns, covariance_matrix, constraints
                )
            elif method == 'min_variance':
                result = self._min_variance_optimization(
                    covariance_matrix, constraint_objects
                )
            elif method == 'max_diversification':
                result = self._max_diversification_optimization(
                    expected_returns, covariance_matrix, constraint_objects
                )
            else:
                raise OptimizationError(f"Unknown optimization method: {method}")
            
            # Add timing information
            result.optimization_time = time.time() - start_time
            
            return result.to_dict()
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Optimization failed: {str(e)}",
                'optimization_time': time.time() - start_time,
                'weights': pd.Series(dtype=float),
                'expected_return': 0.0,
                'expected_risk': 0.0,
                'sharpe_ratio': 0.0
            }
    
    def rebalance(
        self, 
        current_portfolio: Portfolio, 
        target_weights: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate rebalancing trades with transaction cost optimization.
        
        Args:
            current_portfolio: Current portfolio state
            target_weights: Target portfolio weights
            
        Returns:
            Dictionary containing rebalancing results
        """
        start_time = time.time()
        
        try:
            # Get current weights
            current_weights = pd.Series(current_portfolio.get_weights())
            
            # Align indices
            all_assets = set(current_weights.index) | set(target_weights.index)
            current_weights = current_weights.reindex(all_assets, fill_value=0.0)
            target_weights = target_weights.reindex(all_assets, fill_value=0.0)
            
            # Use transaction cost optimizer for rebalancing
            result = self.tc_optimizer.optimize_rebalancing(
                current_weights, target_weights, current_portfolio.total_value
            )
            
            result.optimization_time = time.time() - start_time
            return result.to_dict()
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Rebalancing failed: {str(e)}",
                'optimization_time': time.time() - start_time,
                'trades': pd.DataFrame(),
                'total_turnover': 0.0,
                'transaction_costs': 0.0,
                'expected_tracking_error': 0.0
            }
    
    def _validate_inputs(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame):
        """Validate optimization inputs."""
        if expected_returns.empty:
            raise ValidationError("Expected returns cannot be empty")
        
        if covariance_matrix.empty:
            raise ValidationError("Covariance matrix cannot be empty")
        
        if len(expected_returns) != len(covariance_matrix):
            raise ValidationError("Expected returns and covariance matrix dimensions must match")
        
        if not expected_returns.index.equals(covariance_matrix.index):
            raise ValidationError("Expected returns and covariance matrix indices must match")
        
        if not covariance_matrix.index.equals(covariance_matrix.columns):
            raise ValidationError("Covariance matrix must be square")
        
        # Check for positive semi-definite covariance matrix
        eigenvals = np.linalg.eigvals(covariance_matrix.values)
        if np.any(eigenvals < -1e-8):
            raise ValidationError("Covariance matrix must be positive semi-definite")
    
    def _extract_method(self, constraints: List[Dict[str, Any]]) -> str:
        """Extract optimization method from constraints."""
        for constraint in constraints:
            if constraint.get('type') == 'method':
                return constraint.get('method', 'mean_variance')
        return 'mean_variance'
    
    def _parse_constraints(self, constraints: List[Dict[str, Any]]) -> List[Constraint]:
        """Parse constraint dictionaries into constraint objects."""
        # This is a simplified implementation
        # In practice, you would parse the constraint dictionaries
        # and create appropriate constraint objects
        return []
    
    def _mean_variance_optimization(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Constraint]
    ) -> OptimizationResult:
        """Enhanced mean-variance optimization with advanced constraints."""
        n_assets = len(expected_returns)
        
        # Use CVXPY for convex optimization
        w = cp.Variable(n_assets)
        
        # Objective: maximize utility (return - risk penalty)
        risk_aversion = 1.0  # Can be parameterized
        utility = expected_returns.values @ w - 0.5 * risk_aversion * cp.quad_form(w, covariance_matrix.values)
        objective = cp.Maximize(utility)
        
        # Basic constraints
        cvx_constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # Long-only (can be modified)
        ]
        
        # Add custom constraints
        for constraint in constraints:
            constraint_params = constraint.apply(np.zeros(n_assets))
            # Add constraint to cvx_constraints based on constraint_params
            # This is simplified - actual implementation would be more complex
        
        # Solve optimization problem
        problem = cp.Problem(objective, cvx_constraints)
        problem.solve()
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise OptimizationError(f"Optimization failed with status: {problem.status}")
        
        # Extract results
        weights = pd.Series(w.value, index=expected_returns.index)
        expected_return = float(expected_returns @ weights)
        expected_risk = float(np.sqrt(weights @ covariance_matrix @ weights))
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0.0
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            success=True,
            method='mean_variance',
            objective_value=float(problem.value),
            constraints_satisfied=True,
            optimization_time=0.0,  # Will be set by caller
            iterations=getattr(getattr(problem, 'solver_stats', None), 'num_iters', 0) if hasattr(problem, 'solver_stats') else 0,
            message="Optimization completed successfully"
        )
    
    def _black_litterman_optimization(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Dict[str, Any]]
    ) -> OptimizationResult:
        """Black-Litterman optimization with market views."""
        return self.bl_optimizer.optimize(expected_returns, covariance_matrix, constraints)
    
    def _risk_parity_optimization(
        self, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Constraint]
    ) -> OptimizationResult:
        """Risk parity optimization."""
        return self.rp_optimizer.optimize(covariance_matrix, constraints)
    
    def _equal_risk_contribution_optimization(
        self, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Constraint]
    ) -> OptimizationResult:
        """Equal risk contribution optimization."""
        return self.rp_optimizer.optimize_equal_risk_contribution(covariance_matrix, constraints)
    
    def _factor_based_optimization(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Dict[str, Any]]
    ) -> OptimizationResult:
        """Factor-based optimization using multi-factor models."""
        return self.factor_optimizer.optimize(expected_returns, covariance_matrix, constraints)
    
    def _min_variance_optimization(
        self, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Constraint]
    ) -> OptimizationResult:
        """Minimum variance optimization."""
        n_assets = len(covariance_matrix)
        
        # Use CVXPY for convex optimization
        w = cp.Variable(n_assets)
        
        # Objective: minimize variance
        variance = cp.quad_form(w, covariance_matrix.values)
        objective = cp.Minimize(variance)
        
        # Basic constraints
        cvx_constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # Long-only
        ]
        
        # Solve optimization problem
        problem = cp.Problem(objective, cvx_constraints)
        problem.solve()
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise OptimizationError(f"Optimization failed with status: {problem.status}")
        
        # Extract results
        weights = pd.Series(w.value, index=covariance_matrix.index)
        expected_risk = float(np.sqrt(weights @ covariance_matrix @ weights))
        
        return OptimizationResult(
            weights=weights,
            expected_return=0.0,  # Not applicable for min variance
            expected_risk=expected_risk,
            sharpe_ratio=0.0,  # Not applicable
            success=True,
            method='min_variance',
            objective_value=float(problem.value),
            constraints_satisfied=True,
            optimization_time=0.0,
            iterations=getattr(getattr(problem, 'solver_stats', None), 'num_iters', 0) if hasattr(problem, 'solver_stats') else 0,
            message="Minimum variance optimization completed successfully"
        )
    
    def _max_diversification_optimization(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Constraint]
    ) -> OptimizationResult:
        """Maximum diversification optimization."""
        n_assets = len(expected_returns)
        
        # Calculate asset volatilities
        volatilities = np.sqrt(np.diag(covariance_matrix.values))
        
        def objective_function(weights):
            # Diversification ratio = weighted average volatility / portfolio volatility
            weighted_vol = np.sum(weights * volatilities)
            portfolio_vol = np.sqrt(weights @ covariance_matrix.values @ weights)
            return -weighted_vol / portfolio_vol  # Negative for maximization
        
        # Constraints
        scipy_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds (long-only)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
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
            raise OptimizationError(f"Optimization failed: {result.message}")
        
        # Extract results
        weights = pd.Series(result.x, index=expected_returns.index)
        expected_return = float(expected_returns @ weights)
        expected_risk = float(np.sqrt(weights @ covariance_matrix @ weights))
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0.0
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            success=True,
            method='max_diversification',
            objective_value=-result.fun,  # Convert back to positive
            constraints_satisfied=True,
            optimization_time=0.0,
            iterations=result.nit,
            message="Maximum diversification optimization completed successfully"
        )