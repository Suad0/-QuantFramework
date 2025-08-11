"""
Risk parity portfolio optimization implementation.

This module implements risk parity and equal risk contribution optimization methods
that focus on equalizing risk contributions rather than maximizing return.
"""

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import linalg

from .constraints import Constraint
from .optimization_result import OptimizationResult
from ...domain.exceptions import OptimizationError


class RiskParityOptimizer:
    """
    Risk parity portfolio optimization implementation.
    
    Risk parity portfolios aim to equalize the risk contribution of each asset
    to the total portfolio risk, rather than maximizing expected return.
    """
    
    def __init__(self, method: str = 'equal_risk_contribution'):
        """
        Initialize risk parity optimizer.
        
        Args:
            method: Risk parity method ('equal_risk_contribution', 'inverse_volatility', 'equal_marginal_risk')
        """
        self.method = method
    
    def optimize(
        self, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Constraint]
    ) -> OptimizationResult:
        """
        Perform risk parity optimization.
        
        Args:
            covariance_matrix: Asset return covariance matrix
            constraints: List of constraint objects
            
        Returns:
            OptimizationResult with risk parity optimal weights
        """
        try:
            if self.method == 'equal_risk_contribution':
                return self.optimize_equal_risk_contribution(covariance_matrix, constraints)
            elif self.method == 'inverse_volatility':
                return self.optimize_inverse_volatility(covariance_matrix, constraints)
            elif self.method == 'equal_marginal_risk':
                return self.optimize_equal_marginal_risk(covariance_matrix, constraints)
            else:
                raise OptimizationError(f"Unknown risk parity method: {self.method}")
                
        except Exception as e:
            raise OptimizationError(f"Risk parity optimization failed: {str(e)}")
    
    def optimize_equal_risk_contribution(
        self, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Constraint]
    ) -> OptimizationResult:
        """
        Optimize for equal risk contribution (ERC) portfolio.
        
        In an ERC portfolio, each asset contributes equally to the total portfolio risk.
        This is achieved by minimizing the sum of squared differences between
        risk contributions and their target (1/n).
        """
        n_assets = len(covariance_matrix)
        Sigma = covariance_matrix.values
        
        def objective_function(weights):
            """
            Objective function for ERC optimization.
            Minimizes the sum of squared deviations from equal risk contribution.
            """
            # Portfolio variance
            portfolio_var = weights @ Sigma @ weights
            
            if portfolio_var <= 1e-10:
                return 1e10  # Penalty for zero variance
            
            # Marginal risk contributions
            marginal_contrib = Sigma @ weights
            
            # Risk contributions
            risk_contrib = weights * marginal_contrib / portfolio_var
            
            # Target risk contribution (equal for all assets)
            target_contrib = 1.0 / n_assets
            
            # Sum of squared deviations from target
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        scipy_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Add custom constraints
        for constraint in constraints:
            constraint_params = constraint.apply(np.zeros(n_assets))
            if 'constraints' in constraint_params:
                scipy_constraints.extend(constraint_params['constraints'])
        
        # Bounds (long-only by default)
        bounds = [(0.001, 0.999) for _ in range(n_assets)]  # Small bounds to avoid numerical issues
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            raise OptimizationError(f"ERC optimization failed: {result.message}")
        
        # Create result
        weights = pd.Series(result.x, index=covariance_matrix.index)
        return self._create_result(weights, covariance_matrix, 'equal_risk_contribution', result)
    
    def optimize_inverse_volatility(
        self, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Constraint]
    ) -> OptimizationResult:
        """
        Optimize for inverse volatility weighted portfolio.
        
        Assets are weighted inversely proportional to their volatility.
        This is a simple form of risk parity.
        """
        # Calculate individual asset volatilities
        volatilities = np.sqrt(np.diag(covariance_matrix.values))
        
        # Inverse volatility weights
        inv_vol_weights = 1.0 / volatilities
        weights_values = inv_vol_weights / np.sum(inv_vol_weights)
        
        weights = pd.Series(weights_values, index=covariance_matrix.index)
        
        # Validate constraints
        constraints_satisfied = all(
            constraint.validate(weights_values, asset_names=covariance_matrix.index.tolist())
            for constraint in constraints
        )
        
        return OptimizationResult(
            weights=weights,
            expected_return=0.0,  # Not applicable for risk parity
            expected_risk=float(np.sqrt(weights @ covariance_matrix @ weights)),
            sharpe_ratio=0.0,  # Not applicable
            success=True,
            method='inverse_volatility',
            objective_value=0.0,
            constraints_satisfied=constraints_satisfied,
            optimization_time=0.0,
            iterations=1,  # Analytical solution
            message="Inverse volatility optimization completed successfully"
        )
    
    def optimize_equal_marginal_risk(
        self, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Constraint]
    ) -> OptimizationResult:
        """
        Optimize for equal marginal risk contribution portfolio.
        
        In this approach, the marginal risk contribution of each asset is equalized.
        """
        n_assets = len(covariance_matrix)
        Sigma = covariance_matrix.values
        
        def objective_function(weights):
            """
            Objective function for equal marginal risk optimization.
            Minimizes the variance of marginal risk contributions.
            """
            # Portfolio variance
            portfolio_var = weights @ Sigma @ weights
            
            if portfolio_var <= 1e-10:
                return 1e10  # Penalty for zero variance
            
            # Marginal risk contributions
            marginal_contrib = Sigma @ weights
            
            # Normalize by portfolio volatility
            portfolio_vol = np.sqrt(portfolio_var)
            normalized_marginal = marginal_contrib / portfolio_vol
            
            # Minimize variance of marginal contributions
            return np.var(normalized_marginal)
        
        # Constraints
        scipy_constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds (long-only by default)
        bounds = [(0.001, 0.999) for _ in range(n_assets)]
        
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
            raise OptimizationError(f"Equal marginal risk optimization failed: {result.message}")
        
        # Create result
        weights = pd.Series(result.x, index=covariance_matrix.index)
        return self._create_result(weights, covariance_matrix, 'equal_marginal_risk', result)
    
    def calculate_risk_contributions(self, weights: pd.Series, covariance_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate risk contributions for a given portfolio.
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            
        Returns:
            Series of risk contributions for each asset
        """
        # Portfolio variance
        portfolio_var = weights @ covariance_matrix @ weights
        
        if portfolio_var <= 1e-10:
            return pd.Series(0.0, index=weights.index)
        
        # Marginal risk contributions
        marginal_contrib = covariance_matrix @ weights
        
        # Risk contributions
        risk_contrib = weights * marginal_contrib / portfolio_var
        
        return risk_contrib
    
    def calculate_marginal_risk_contributions(
        self, 
        weights: pd.Series, 
        covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate marginal risk contributions for a given portfolio.
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            
        Returns:
            Series of marginal risk contributions for each asset
        """
        # Portfolio volatility
        portfolio_vol = np.sqrt(weights @ covariance_matrix @ weights)
        
        if portfolio_vol <= 1e-10:
            return pd.Series(0.0, index=weights.index)
        
        # Marginal risk contributions
        marginal_contrib = (covariance_matrix @ weights) / portfolio_vol
        
        return marginal_contrib
    
    def _create_result(
        self, 
        weights: pd.Series, 
        covariance_matrix: pd.DataFrame,
        method: str, 
        optimization_result: Optional[object] = None
    ) -> OptimizationResult:
        """Create optimization result object."""
        expected_risk = float(np.sqrt(weights @ covariance_matrix @ weights))
        
        # Calculate risk contributions for metadata
        risk_contrib = self.calculate_risk_contributions(weights, covariance_matrix)
        marginal_contrib = self.calculate_marginal_risk_contributions(weights, covariance_matrix)
        
        metadata = {
            'risk_contributions': risk_contrib.to_dict(),
            'marginal_risk_contributions': marginal_contrib.to_dict(),
            'risk_contribution_std': float(risk_contrib.std()),
            'marginal_risk_contribution_std': float(marginal_contrib.std())
        }
        
        return OptimizationResult(
            weights=weights,
            expected_return=0.0,  # Not applicable for risk parity
            expected_risk=expected_risk,
            sharpe_ratio=0.0,  # Not applicable without expected returns
            success=True,
            method=method,
            objective_value=float(optimization_result.fun) if optimization_result else 0.0,
            constraints_satisfied=True,
            optimization_time=0.0,
            iterations=optimization_result.nit if optimization_result else 1,
            message=f"{method} optimization completed successfully",
            metadata=metadata
        )
    
    def analyze_risk_budget(self, weights: pd.Series, covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze the risk budget of a portfolio.
        
        Args:
            weights: Portfolio weights
            covariance_matrix: Asset covariance matrix
            
        Returns:
            Dictionary with risk budget analysis
        """
        risk_contrib = self.calculate_risk_contributions(weights, covariance_matrix)
        marginal_contrib = self.calculate_marginal_risk_contributions(weights, covariance_matrix)
        
        return {
            'total_risk': float(np.sqrt(weights @ covariance_matrix @ weights)),
            'risk_contributions': risk_contrib.to_dict(),
            'marginal_risk_contributions': marginal_contrib.to_dict(),
            'risk_concentration': float(np.sum(risk_contrib ** 2)),  # Herfindahl index
            'effective_number_of_bets': float(1.0 / np.sum(risk_contrib ** 2)),
            'max_risk_contribution': float(risk_contrib.max()),
            'min_risk_contribution': float(risk_contrib.min()),
            'risk_contribution_range': float(risk_contrib.max() - risk_contrib.min())
        }