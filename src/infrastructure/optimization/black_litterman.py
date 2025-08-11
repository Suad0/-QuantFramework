"""
Black-Litterman portfolio optimization implementation.

This module implements the Black-Litterman model for portfolio optimization,
which combines market equilibrium with investor views to generate expected returns.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy import linalg

from .optimization_result import OptimizationResult
from ...domain.exceptions import OptimizationError, ValidationError


class BlackLittermanOptimizer:
    """
    Black-Litterman portfolio optimization implementation.
    
    The Black-Litterman model combines:
    1. Market equilibrium (implied returns from market cap weights)
    2. Investor views (subjective beliefs about expected returns)
    3. Uncertainty in both equilibrium and views
    """
    
    def __init__(self, risk_aversion: float = 3.0, tau: float = 0.025):
        """
        Initialize Black-Litterman optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter (typically 2-5)
            tau: Scaling factor for uncertainty of prior (typically 0.01-0.05)
        """
        self.risk_aversion = risk_aversion
        self.tau = tau
    
    def optimize(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        constraints: List[Dict[str, Any]]
    ) -> OptimizationResult:
        """
        Perform Black-Litterman optimization.
        
        Args:
            expected_returns: Market equilibrium returns (can be implied from market caps)
            covariance_matrix: Asset return covariance matrix
            constraints: List containing market views and other parameters
            
        Returns:
            OptimizationResult with Black-Litterman optimal weights
        """
        try:
            # Extract Black-Litterman specific parameters
            bl_params = self._extract_bl_parameters(constraints)
            
            # Get market capitalization weights (equilibrium portfolio)
            market_caps = bl_params.get('market_caps')
            if market_caps is None:
                # If no market caps provided, use equal weights as proxy
                market_caps = pd.Series(1.0, index=expected_returns.index)
            
            w_market = market_caps / market_caps.sum()
            
            # Calculate implied equilibrium returns
            pi = self._calculate_implied_returns(w_market, covariance_matrix)
            
            # Get investor views
            views = bl_params.get('views', [])
            if not views:
                # No views provided, return market portfolio
                return self._create_result(w_market, pi, covariance_matrix, 'black_litterman_no_views')
            
            # Process views
            P, Q, Omega = self._process_views(views, expected_returns.index)
            
            # Calculate Black-Litterman expected returns
            bl_returns = self._calculate_bl_returns(pi, P, Q, Omega, covariance_matrix)
            
            # Calculate Black-Litterman covariance matrix
            bl_covariance = self._calculate_bl_covariance(P, Omega, covariance_matrix)
            
            # Optimize portfolio using Black-Litterman inputs
            optimal_weights = self._optimize_portfolio(bl_returns, bl_covariance)
            
            return self._create_result(optimal_weights, bl_returns, bl_covariance, 'black_litterman')
            
        except Exception as e:
            raise OptimizationError(f"Black-Litterman optimization failed: {str(e)}")
    
    def _extract_bl_parameters(self, constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract Black-Litterman specific parameters from constraints."""
        bl_params = {}
        
        for constraint in constraints:
            if constraint.get('type') == 'black_litterman':
                bl_params.update(constraint)
                break
        
        return bl_params
    
    def _calculate_implied_returns(self, market_weights: pd.Series, covariance_matrix: pd.DataFrame) -> pd.Series:
        """
        Calculate implied equilibrium returns from market portfolio.
        
        Formula: π = λ * Σ * w_market
        where λ is risk aversion, Σ is covariance matrix, w_market is market portfolio
        """
        implied_returns = self.risk_aversion * (covariance_matrix @ market_weights)
        return implied_returns
    
    def _process_views(self, views: List[Dict[str, Any]], asset_index: pd.Index) -> tuple:
        """
        Process investor views into matrices P, Q, and Omega.
        
        Args:
            views: List of view dictionaries
            asset_index: Index of assets
            
        Returns:
            Tuple of (P, Q, Omega) matrices
        """
        n_assets = len(asset_index)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))  # Picking matrix
        Q = np.zeros(n_views)  # View returns
        Omega = np.zeros((n_views, n_views))  # View uncertainty matrix
        
        for i, view in enumerate(views):
            view_type = view.get('type', 'absolute')
            
            if view_type == 'absolute':
                # Absolute view: Asset A will return X%
                asset = view['asset']
                if asset not in asset_index:
                    raise ValidationError(f"Asset {asset} not found in portfolio")
                
                asset_idx = asset_index.get_loc(asset)
                P[i, asset_idx] = 1.0
                Q[i] = view['expected_return']
                Omega[i, i] = view.get('confidence', 0.01)  # Default 1% uncertainty
                
            elif view_type == 'relative':
                # Relative view: Asset A will outperform Asset B by X%
                asset_a = view['asset_a']
                asset_b = view['asset_b']
                
                if asset_a not in asset_index or asset_b not in asset_index:
                    raise ValidationError(f"Assets {asset_a} or {asset_b} not found in portfolio")
                
                idx_a = asset_index.get_loc(asset_a)
                idx_b = asset_index.get_loc(asset_b)
                
                P[i, idx_a] = 1.0
                P[i, idx_b] = -1.0
                Q[i] = view['expected_outperformance']
                Omega[i, i] = view.get('confidence', 0.01)
                
            elif view_type == 'portfolio':
                # Portfolio view: A portfolio of assets will return X%
                portfolio_weights = view['weights']
                for asset, weight in portfolio_weights.items():
                    if asset not in asset_index:
                        raise ValidationError(f"Asset {asset} not found in portfolio")
                    
                    asset_idx = asset_index.get_loc(asset)
                    P[i, asset_idx] = weight
                
                Q[i] = view['expected_return']
                Omega[i, i] = view.get('confidence', 0.01)
        
        return P, Q, Omega
    
    def _calculate_bl_returns(
        self, 
        pi: pd.Series, 
        P: np.ndarray, 
        Q: np.ndarray, 
        Omega: np.ndarray,
        covariance_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate Black-Litterman expected returns.
        
        Formula: μ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * [(τΣ)^(-1)π + P'Ω^(-1)Q]
        """
        Sigma = covariance_matrix.values
        tau_Sigma_inv = linalg.inv(self.tau * Sigma)
        Omega_inv = linalg.inv(Omega)
        
        # Calculate the precision matrices
        prior_precision = tau_Sigma_inv
        view_precision = P.T @ Omega_inv @ P
        
        # Combined precision matrix
        combined_precision = prior_precision + view_precision
        combined_precision_inv = linalg.inv(combined_precision)
        
        # Calculate Black-Litterman returns
        prior_term = prior_precision @ pi.values
        view_term = P.T @ Omega_inv @ Q
        
        bl_returns_values = combined_precision_inv @ (prior_term + view_term)
        bl_returns = pd.Series(bl_returns_values, index=pi.index)
        
        return bl_returns
    
    def _calculate_bl_covariance(
        self, 
        P: np.ndarray, 
        Omega: np.ndarray,
        covariance_matrix: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Black-Litterman covariance matrix.
        
        Formula: Σ_BL = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1)
        """
        Sigma = covariance_matrix.values
        tau_Sigma_inv = linalg.inv(self.tau * Sigma)
        Omega_inv = linalg.inv(Omega)
        
        # Combined precision matrix
        combined_precision = tau_Sigma_inv + P.T @ Omega_inv @ P
        bl_covariance_values = linalg.inv(combined_precision)
        
        bl_covariance = pd.DataFrame(
            bl_covariance_values,
            index=covariance_matrix.index,
            columns=covariance_matrix.columns
        )
        
        return bl_covariance
    
    def _optimize_portfolio(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame) -> pd.Series:
        """
        Optimize portfolio using Black-Litterman expected returns and covariance.
        
        Uses mean-variance optimization with the Black-Litterman inputs.
        """
        # Mean-variance optimization: w = (1/λ) * Σ^(-1) * μ
        Sigma_inv = linalg.inv(covariance_matrix.values)
        optimal_weights_values = (1 / self.risk_aversion) * (Sigma_inv @ expected_returns.values)
        
        # Normalize weights to sum to 1
        optimal_weights_values = optimal_weights_values / np.sum(optimal_weights_values)
        
        # Ensure non-negative weights (long-only constraint)
        optimal_weights_values = np.maximum(optimal_weights_values, 0)
        optimal_weights_values = optimal_weights_values / np.sum(optimal_weights_values)
        
        optimal_weights = pd.Series(optimal_weights_values, index=expected_returns.index)
        return optimal_weights
    
    def _create_result(
        self, 
        weights: pd.Series, 
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame, 
        method: str
    ) -> OptimizationResult:
        """Create optimization result object."""
        expected_return = float(expected_returns @ weights)
        expected_risk = float(np.sqrt(weights @ covariance_matrix @ weights))
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0.0
        
        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            success=True,
            method=method,
            objective_value=expected_return - 0.5 * self.risk_aversion * expected_risk**2,
            constraints_satisfied=True,
            optimization_time=0.0,
            iterations=1,  # Analytical solution
            message="Black-Litterman optimization completed successfully"
        )
    
    def create_absolute_view(self, asset: str, expected_return: float, confidence: float = 0.01) -> Dict[str, Any]:
        """
        Create an absolute view for an asset.
        
        Args:
            asset: Asset symbol
            expected_return: Expected return for the asset
            confidence: Confidence in the view (lower = more confident)
            
        Returns:
            View dictionary
        """
        return {
            'type': 'absolute',
            'asset': asset,
            'expected_return': expected_return,
            'confidence': confidence
        }
    
    def create_relative_view(
        self, 
        asset_a: str, 
        asset_b: str, 
        expected_outperformance: float,
        confidence: float = 0.01
    ) -> Dict[str, Any]:
        """
        Create a relative view between two assets.
        
        Args:
            asset_a: First asset symbol
            asset_b: Second asset symbol
            expected_outperformance: Expected outperformance of A over B
            confidence: Confidence in the view
            
        Returns:
            View dictionary
        """
        return {
            'type': 'relative',
            'asset_a': asset_a,
            'asset_b': asset_b,
            'expected_outperformance': expected_outperformance,
            'confidence': confidence
        }
    
    def create_portfolio_view(
        self, 
        weights: Dict[str, float], 
        expected_return: float,
        confidence: float = 0.01
    ) -> Dict[str, Any]:
        """
        Create a portfolio view for a combination of assets.
        
        Args:
            weights: Dictionary of asset weights in the view portfolio
            expected_return: Expected return of the view portfolio
            confidence: Confidence in the view
            
        Returns:
            View dictionary
        """
        return {
            'type': 'portfolio',
            'weights': weights,
            'expected_return': expected_return,
            'confidence': confidence
        }