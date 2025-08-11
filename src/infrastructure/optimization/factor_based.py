"""
Factor-based portfolio optimization implementation.

This module implements factor-based optimization using multi-factor models
such as Fama-French factors, custom risk factors, and factor tilting strategies.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp

from .optimization_result import OptimizationResult
from ...domain.exceptions import OptimizationError, ValidationError


class FactorBasedOptimizer:
    """
    Factor-based portfolio optimization implementation.
    
    Supports various factor-based optimization approaches:
    - Factor exposure optimization
    - Factor risk budgeting
    - Factor tilting
    - Multi-factor model optimization
    """
    
    def __init__(self):
        """Initialize factor-based optimizer."""
        pass
    
    def optimize(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Dict[str, Any]]
    ) -> OptimizationResult:
        """
        Perform factor-based optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Asset return covariance matrix
            constraints: List containing factor model parameters and constraints
            
        Returns:
            OptimizationResult with factor-based optimal weights
        """
        try:
            # Extract factor-based parameters
            factor_params = self._extract_factor_parameters(constraints)
            
            # Get factor model components
            factor_loadings = factor_params.get('factor_loadings')
            factor_returns = factor_params.get('factor_returns')
            factor_covariance = factor_params.get('factor_covariance')
            specific_risk = factor_params.get('specific_risk')
            
            if factor_loadings is None:
                raise ValidationError("Factor loadings matrix is required for factor-based optimization")
            
            # Determine optimization method
            method = factor_params.get('method', 'factor_exposure')
            
            if method == 'factor_exposure':
                return self._optimize_factor_exposure(
                    expected_returns, covariance_matrix, factor_params
                )
            elif method == 'factor_risk_budgeting':
                return self._optimize_factor_risk_budgeting(
                    expected_returns, covariance_matrix, factor_params
                )
            elif method == 'factor_tilting':
                return self._optimize_factor_tilting(
                    expected_returns, covariance_matrix, factor_params
                )
            elif method == 'multi_factor_model':
                return self._optimize_multi_factor_model(
                    expected_returns, factor_loadings, factor_returns, 
                    factor_covariance, specific_risk, factor_params
                )
            else:
                raise OptimizationError(f"Unknown factor-based method: {method}")
                
        except Exception as e:
            raise OptimizationError(f"Factor-based optimization failed: {str(e)}")
    
    def _extract_factor_parameters(self, constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract factor-based parameters from constraints."""
        factor_params = {}
        
        for constraint in constraints:
            if constraint.get('type') == 'factor_based':
                factor_params.update(constraint)
                break
        
        return factor_params
    
    def _optimize_factor_exposure(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        factor_params: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Optimize portfolio with specific factor exposure targets.
        
        This method optimizes the portfolio to achieve target exposures
        to specific factors while maximizing expected return or minimizing risk.
        """
        n_assets = len(expected_returns)
        factor_loadings = factor_params['factor_loadings']  # DataFrame: assets x factors
        target_exposures = factor_params.get('target_exposures', {})  # Dict: factor -> target exposure
        
        # Use CVXPY for optimization
        w = cp.Variable(n_assets)
        
        # Objective: maximize expected return minus risk penalty
        risk_aversion = factor_params.get('risk_aversion', 1.0)
        utility = expected_returns.values @ w - 0.5 * risk_aversion * cp.quad_form(w, covariance_matrix.values)
        objective = cp.Maximize(utility)
        
        # Constraints
        cvx_constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # Long-only (can be modified)
        ]
        
        # Factor exposure constraints
        for factor_name, target_exposure in target_exposures.items():
            if factor_name in factor_loadings.columns:
                factor_loading = factor_loadings[factor_name].values
                portfolio_exposure = factor_loading @ w
                
                # Allow for tolerance around target exposure
                tolerance = factor_params.get('exposure_tolerance', 0.1)
                cvx_constraints.extend([
                    portfolio_exposure >= target_exposure - tolerance,
                    portfolio_exposure <= target_exposure + tolerance
                ])
        
        # Solve optimization problem
        problem = cp.Problem(objective, cvx_constraints)
        problem.solve()
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise OptimizationError(f"Factor exposure optimization failed with status: {problem.status}")
        
        # Extract results
        weights = pd.Series(w.value, index=expected_returns.index)
        
        # Calculate factor exposures
        portfolio_exposures = self._calculate_factor_exposures(weights, factor_loadings)
        
        return self._create_result(
            weights, expected_returns, covariance_matrix, 
            'factor_exposure', problem, portfolio_exposures
        )
    
    def _optimize_factor_risk_budgeting(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        factor_params: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Optimize portfolio with factor risk budgeting.
        
        This method allocates risk budget across different factors
        rather than individual assets.
        """
        factor_loadings = factor_params['factor_loadings']
        factor_covariance = factor_params.get('factor_covariance')
        specific_risk = factor_params.get('specific_risk')
        risk_budgets = factor_params.get('risk_budgets', {})  # Dict: factor -> risk budget
        
        if factor_covariance is None or specific_risk is None:
            raise ValidationError("Factor covariance and specific risk are required for factor risk budgeting")
        
        n_assets = len(expected_returns)
        n_factors = len(factor_loadings.columns)
        
        def objective_function(weights):
            """
            Objective function for factor risk budgeting.
            Minimizes deviations from target factor risk budgets.
            """
            # Calculate factor risk contributions
            factor_risk_contrib = self._calculate_factor_risk_contributions(
                weights, factor_loadings, factor_covariance, specific_risk
            )
            
            # Calculate deviations from target risk budgets
            total_deviation = 0.0
            for factor_name, target_budget in risk_budgets.items():
                if factor_name in factor_risk_contrib:
                    actual_budget = factor_risk_contrib[factor_name]
                    total_deviation += (actual_budget - target_budget) ** 2
            
            return total_deviation
        
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
            raise OptimizationError(f"Factor risk budgeting optimization failed: {result.message}")
        
        # Extract results
        weights = pd.Series(result.x, index=expected_returns.index)
        portfolio_exposures = self._calculate_factor_exposures(weights, factor_loadings)
        
        return self._create_result(
            weights, expected_returns, covariance_matrix, 
            'factor_risk_budgeting', result, portfolio_exposures
        )
    
    def _optimize_factor_tilting(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        factor_params: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Optimize portfolio with factor tilting.
        
        This method tilts the portfolio towards or away from specific factors
        based on factor views or predictions.
        """
        factor_loadings = factor_params['factor_loadings']
        factor_views = factor_params.get('factor_views', {})  # Dict: factor -> view (positive/negative)
        benchmark_weights = factor_params.get('benchmark_weights')  # Benchmark portfolio
        
        if benchmark_weights is None:
            # Use equal weights as benchmark
            benchmark_weights = pd.Series(1.0 / len(expected_returns), index=expected_returns.index)
        
        n_assets = len(expected_returns)
        
        # Use CVXPY for optimization
        w = cp.Variable(n_assets)
        
        # Base objective: maximize expected return minus risk penalty
        risk_aversion = factor_params.get('risk_aversion', 1.0)
        base_utility = expected_returns.values @ w - 0.5 * risk_aversion * cp.quad_form(w, covariance_matrix.values)
        
        # Factor tilting terms
        tilting_strength = factor_params.get('tilting_strength', 0.1)
        tilting_term = 0.0
        
        for factor_name, view in factor_views.items():
            if factor_name in factor_loadings.columns:
                factor_loading = factor_loadings[factor_name].values
                portfolio_exposure = factor_loading @ w
                benchmark_exposure = factor_loading @ benchmark_weights.values
                
                # Tilt towards factor if view is positive, away if negative
                tilting_term += view * tilting_strength * (portfolio_exposure - benchmark_exposure)
        
        # Combined objective
        objective = cp.Maximize(base_utility + tilting_term)
        
        # Constraints
        cvx_constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # Long-only
        ]
        
        # Tracking error constraint (optional)
        max_tracking_error = factor_params.get('max_tracking_error')
        if max_tracking_error is not None:
            tracking_var = cp.quad_form(w - benchmark_weights.values, covariance_matrix.values)
            cvx_constraints.append(tracking_var <= max_tracking_error ** 2)
        
        # Solve optimization problem
        problem = cp.Problem(objective, cvx_constraints)
        problem.solve()
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise OptimizationError(f"Factor tilting optimization failed with status: {problem.status}")
        
        # Extract results
        weights = pd.Series(w.value, index=expected_returns.index)
        portfolio_exposures = self._calculate_factor_exposures(weights, factor_loadings)
        
        return self._create_result(
            weights, expected_returns, covariance_matrix, 
            'factor_tilting', problem, portfolio_exposures
        )
    
    def _optimize_multi_factor_model(
        self, 
        expected_returns: pd.Series,
        factor_loadings: pd.DataFrame,
        factor_returns: pd.Series,
        factor_covariance: pd.DataFrame,
        specific_risk: pd.Series,
        factor_params: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Optimize portfolio using a multi-factor model.
        
        This method uses the factor model to decompose risk and return,
        then optimizes based on factor exposures and specific risks.
        """
        n_assets = len(expected_returns)
        
        # Use CVXPY for optimization
        w = cp.Variable(n_assets)
        
        # Factor-based expected returns
        factor_expected_returns = factor_loadings.values @ factor_returns.values
        
        # Factor-based risk model
        factor_risk = cp.quad_form(factor_loadings.values.T @ w, factor_covariance.values)
        specific_risk_term = cp.sum(cp.multiply(cp.square(w), specific_risk.values))
        total_risk = factor_risk + specific_risk_term
        
        # Objective: maximize expected return minus risk penalty
        risk_aversion = factor_params.get('risk_aversion', 1.0)
        utility = factor_expected_returns @ w - 0.5 * risk_aversion * total_risk
        objective = cp.Maximize(utility)
        
        # Constraints
        cvx_constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # Long-only
        ]
        
        # Factor exposure constraints
        factor_limits = factor_params.get('factor_limits', {})
        for factor_name, (min_exp, max_exp) in factor_limits.items():
            if factor_name in factor_loadings.columns:
                factor_loading = factor_loadings[factor_name].values
                portfolio_exposure = factor_loading @ w
                cvx_constraints.extend([
                    portfolio_exposure >= min_exp,
                    portfolio_exposure <= max_exp
                ])
        
        # Solve optimization problem
        problem = cp.Problem(objective, cvx_constraints)
        problem.solve()
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise OptimizationError(f"Multi-factor model optimization failed with status: {problem.status}")
        
        # Extract results
        weights = pd.Series(w.value, index=expected_returns.index)
        portfolio_exposures = self._calculate_factor_exposures(weights, factor_loadings)
        
        # Calculate factor-based risk decomposition
        risk_decomposition = self._calculate_factor_risk_decomposition(
            weights, factor_loadings, factor_covariance, specific_risk
        )
        
        return self._create_result(
            weights, expected_returns, None, 
            'multi_factor_model', problem, portfolio_exposures, risk_decomposition
        )
    
    def _calculate_factor_exposures(
        self, 
        weights: pd.Series, 
        factor_loadings: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate portfolio factor exposures."""
        exposures = {}
        for factor_name in factor_loadings.columns:
            exposure = float(factor_loadings[factor_name] @ weights)
            exposures[factor_name] = exposure
        return exposures
    
    def _calculate_factor_risk_contributions(
        self, 
        weights: np.ndarray,
        factor_loadings: pd.DataFrame,
        factor_covariance: pd.DataFrame,
        specific_risk: pd.Series
    ) -> Dict[str, float]:
        """Calculate factor risk contributions."""
        # Portfolio factor exposures
        factor_exposures = factor_loadings.T @ weights
        
        # Factor risk contributions
        factor_risk_contrib = {}
        total_risk_var = 0.0
        
        # Factor risk variance
        factor_risk_var = factor_exposures @ factor_covariance.values @ factor_exposures
        total_risk_var += factor_risk_var
        
        # Specific risk variance
        specific_risk_var = np.sum(weights**2 * specific_risk.values**2)
        total_risk_var += specific_risk_var
        
        # Calculate contributions
        for i, factor_name in enumerate(factor_loadings.columns):
            factor_contrib = (factor_exposures[i]**2 * factor_covariance.iloc[i, i]) / total_risk_var
            factor_risk_contrib[factor_name] = float(factor_contrib)
        
        # Specific risk contribution
        factor_risk_contrib['specific_risk'] = float(specific_risk_var / total_risk_var)
        
        return factor_risk_contrib
    
    def _calculate_factor_risk_decomposition(
        self, 
        weights: pd.Series,
        factor_loadings: pd.DataFrame,
        factor_covariance: pd.DataFrame,
        specific_risk: pd.Series
    ) -> Dict[str, Any]:
        """Calculate detailed factor risk decomposition."""
        # Portfolio factor exposures
        factor_exposures = factor_loadings.T @ weights
        
        # Factor risk variance
        factor_risk_var = factor_exposures @ factor_covariance @ factor_exposures
        
        # Specific risk variance
        specific_risk_var = np.sum(weights**2 * specific_risk**2)
        
        # Total risk
        total_risk_var = factor_risk_var + specific_risk_var
        total_risk_vol = np.sqrt(total_risk_var)
        
        return {
            'total_risk': float(total_risk_vol),
            'factor_risk': float(np.sqrt(factor_risk_var)),
            'specific_risk': float(np.sqrt(specific_risk_var)),
            'factor_risk_contribution': float(factor_risk_var / total_risk_var),
            'specific_risk_contribution': float(specific_risk_var / total_risk_var),
            'factor_exposures': factor_exposures.to_dict()
        }
    
    def _create_result(
        self, 
        weights: pd.Series, 
        expected_returns: pd.Series,
        covariance_matrix: Optional[pd.DataFrame], 
        method: str,
        optimization_result: object,
        portfolio_exposures: Dict[str, float],
        risk_decomposition: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """Create optimization result object."""
        expected_return = float(expected_returns @ weights)
        
        if covariance_matrix is not None:
            expected_risk = float(np.sqrt(weights @ covariance_matrix @ weights))
        else:
            expected_risk = risk_decomposition.get('total_risk', 0.0) if risk_decomposition else 0.0
        
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0.0
        
        metadata = {
            'factor_exposures': portfolio_exposures
        }
        
        if risk_decomposition:
            metadata['risk_decomposition'] = risk_decomposition
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            success=True,
            method=method,
            objective_value=float(optimization_result.value) if hasattr(optimization_result, 'value') else float(optimization_result.fun),
            constraints_satisfied=True,
            optimization_time=0.0,
            iterations=getattr(optimization_result, 'nit', 0),
            message=f"{method} optimization completed successfully",
            metadata=metadata
        )