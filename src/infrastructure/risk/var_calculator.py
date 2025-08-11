"""
Value at Risk (VaR) calculator with multiple methodologies.

This module implements various VaR calculation methods including:
- Parametric VaR (assuming normal distribution)
- Historical VaR (using historical simulation)
- Monte Carlo VaR (using Monte Carlo simulation)
- Expected Shortfall (CVaR) calculations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings

from ...domain.entities import Portfolio
from ...domain.value_objects import RiskMetrics
from ...domain.exceptions import ValidationError


class VaRMethod(Enum):
    """VaR calculation methods."""
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"


@dataclass
class VaRResult:
    """Result of VaR calculation."""
    var_95: float
    var_99: float
    cvar_95: float  # Expected Shortfall at 95%
    cvar_99: float  # Expected Shortfall at 99%
    method: VaRMethod
    confidence_intervals: Dict[str, Tuple[float, float]]
    calculation_date: datetime
    lookback_days: int
    portfolio_value: float


class VaRCalculator:
    """
    Comprehensive VaR calculator supporting multiple methodologies.
    
    This class provides various methods for calculating Value at Risk and
    Expected Shortfall (Conditional VaR) for portfolios.
    """
    
    def __init__(self, lookback_days: int = 252):
        """
        Initialize VaR calculator.
        
        Args:
            lookback_days: Number of historical days to use for calculations
        """
        self.lookback_days = lookback_days
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if self.lookback_days < 30:
            raise ValidationError("Lookback days must be at least 30")
        if self.lookback_days > 2520:  # 10 years
            warnings.warn("Very long lookback period may not reflect current market conditions")
    
    def calculate_var(
        self,
        returns: pd.Series,
        method: VaRMethod = VaRMethod.HISTORICAL,
        confidence_levels: List[float] = [0.95, 0.99],
        portfolio_value: float = 1.0,
        **kwargs
    ) -> VaRResult:
        """
        Calculate VaR using specified method.
        
        Args:
            returns: Historical returns series
            method: VaR calculation method
            confidence_levels: Confidence levels for VaR calculation
            portfolio_value: Current portfolio value
            **kwargs: Additional method-specific parameters
            
        Returns:
            VaRResult containing VaR and CVaR values
        """
        self._validate_returns(returns)
        
        if method == VaRMethod.PARAMETRIC:
            return self._calculate_parametric_var(
                returns, confidence_levels, portfolio_value, **kwargs
            )
        elif method == VaRMethod.HISTORICAL:
            return self._calculate_historical_var(
                returns, confidence_levels, portfolio_value, **kwargs
            )
        elif method == VaRMethod.MONTE_CARLO:
            return self._calculate_monte_carlo_var(
                returns, confidence_levels, portfolio_value, **kwargs
            )
        else:
            raise ValidationError(f"Unsupported VaR method: {method}")
    
    def _validate_returns(self, returns: pd.Series) -> None:
        """Validate returns data."""
        if returns.empty:
            raise ValidationError("Returns series cannot be empty")
        if len(returns) < 30:
            raise ValidationError("Need at least 30 observations for VaR calculation")
        if returns.isnull().any():
            warnings.warn("Returns contain NaN values, they will be dropped")
    
    def _calculate_parametric_var(
        self,
        returns: pd.Series,
        confidence_levels: List[float],
        portfolio_value: float,
        distribution: str = 'normal',
        **kwargs
    ) -> VaRResult:
        """
        Calculate parametric VaR assuming a specific distribution.
        
        Args:
            returns: Historical returns
            confidence_levels: Confidence levels
            portfolio_value: Portfolio value
            distribution: Distribution assumption ('normal', 't', 'skewed_t')
        """
        clean_returns = returns.dropna()
        
        if distribution == 'normal':
            return self._normal_var(clean_returns, confidence_levels, portfolio_value)
        elif distribution == 't':
            return self._t_distribution_var(clean_returns, confidence_levels, portfolio_value)
        elif distribution == 'skewed_t':
            return self._skewed_t_var(clean_returns, confidence_levels, portfolio_value)
        else:
            raise ValidationError(f"Unsupported distribution: {distribution}")
    
    def _normal_var(
        self,
        returns: pd.Series,
        confidence_levels: List[float],
        portfolio_value: float
    ) -> VaRResult:
        """Calculate VaR assuming normal distribution."""
        mean = returns.mean()
        std = returns.std()
        
        var_values = {}
        cvar_values = {}
        confidence_intervals = {}
        
        for conf in confidence_levels:
            # VaR calculation
            z_score = stats.norm.ppf(1 - conf)
            var = -(mean + z_score * std) * portfolio_value
            var_values[f'var_{int(conf*100)}'] = var
            
            # CVaR calculation (Expected Shortfall)
            cvar = -(mean - std * stats.norm.pdf(z_score) / (1 - conf)) * portfolio_value
            cvar_values[f'cvar_{int(conf*100)}'] = cvar
            
            # Confidence intervals using bootstrap
            ci = self._bootstrap_confidence_interval(returns, conf, portfolio_value)
            confidence_intervals[f'var_{int(conf*100)}'] = ci
        
        return VaRResult(
            var_95=var_values.get('var_95', 0),
            var_99=var_values.get('var_99', 0),
            cvar_95=cvar_values.get('cvar_95', 0),
            cvar_99=cvar_values.get('cvar_99', 0),
            method=VaRMethod.PARAMETRIC,
            confidence_intervals=confidence_intervals,
            calculation_date=datetime.now(),
            lookback_days=len(returns),
            portfolio_value=portfolio_value
        )
    
    def _t_distribution_var(
        self,
        returns: pd.Series,
        confidence_levels: List[float],
        portfolio_value: float
    ) -> VaRResult:
        """Calculate VaR using Student's t-distribution."""
        # Fit t-distribution parameters
        params = stats.t.fit(returns)
        df, loc, scale = params
        
        var_values = {}
        cvar_values = {}
        confidence_intervals = {}
        
        for conf in confidence_levels:
            # VaR calculation
            t_score = stats.t.ppf(1 - conf, df, loc, scale)
            var = -t_score * portfolio_value
            var_values[f'var_{int(conf*100)}'] = var
            
            # CVaR calculation for t-distribution
            cvar = self._t_distribution_cvar(df, loc, scale, conf, portfolio_value)
            cvar_values[f'cvar_{int(conf*100)}'] = cvar
            
            # Confidence intervals
            ci = self._bootstrap_confidence_interval(returns, conf, portfolio_value)
            confidence_intervals[f'var_{int(conf*100)}'] = ci
        
        return VaRResult(
            var_95=var_values.get('var_95', 0),
            var_99=var_values.get('var_99', 0),
            cvar_95=cvar_values.get('cvar_95', 0),
            cvar_99=cvar_values.get('cvar_99', 0),
            method=VaRMethod.PARAMETRIC,
            confidence_intervals=confidence_intervals,
            calculation_date=datetime.now(),
            lookback_days=len(returns),
            portfolio_value=portfolio_value
        )
    
    def _skewed_t_var(
        self,
        returns: pd.Series,
        confidence_levels: List[float],
        portfolio_value: float
    ) -> VaRResult:
        """Calculate VaR using skewed t-distribution."""
        # For simplicity, use t-distribution as approximation
        # In practice, you would use a proper skewed t-distribution library
        return self._t_distribution_var(returns, confidence_levels, portfolio_value)
    
    def _t_distribution_cvar(
        self,
        df: float,
        loc: float,
        scale: float,
        confidence: float,
        portfolio_value: float
    ) -> float:
        """Calculate CVaR for t-distribution."""
        var_quantile = stats.t.ppf(1 - confidence, df, loc, scale)
        
        # Numerical integration for CVaR
        def integrand(x):
            return x * stats.t.pdf(x, df, loc, scale)
        
        from scipy.integrate import quad
        integral, _ = quad(integrand, -np.inf, var_quantile)
        cvar = -integral / (1 - confidence) * portfolio_value
        
        return cvar
    
    def _calculate_historical_var(
        self,
        returns: pd.Series,
        confidence_levels: List[float],
        portfolio_value: float,
        **kwargs
    ) -> VaRResult:
        """Calculate historical VaR using empirical distribution."""
        clean_returns = returns.dropna().tail(self.lookback_days)
        
        var_values = {}
        cvar_values = {}
        confidence_intervals = {}
        
        for conf in confidence_levels:
            # VaR calculation
            var_quantile = np.percentile(clean_returns, (1 - conf) * 100)
            var = -var_quantile * portfolio_value
            var_values[f'var_{int(conf*100)}'] = var
            
            # CVaR calculation
            tail_returns = clean_returns[clean_returns <= var_quantile]
            cvar = -tail_returns.mean() * portfolio_value if len(tail_returns) > 0 else var
            cvar_values[f'cvar_{int(conf*100)}'] = cvar
            
            # Confidence intervals using bootstrap
            ci = self._bootstrap_confidence_interval(clean_returns, conf, portfolio_value)
            confidence_intervals[f'var_{int(conf*100)}'] = ci
        
        return VaRResult(
            var_95=var_values.get('var_95', 0),
            var_99=var_values.get('var_99', 0),
            cvar_95=cvar_values.get('cvar_95', 0),
            cvar_99=cvar_values.get('cvar_99', 0),
            method=VaRMethod.HISTORICAL,
            confidence_intervals=confidence_intervals,
            calculation_date=datetime.now(),
            lookback_days=len(clean_returns),
            portfolio_value=portfolio_value
        )
    
    def _calculate_monte_carlo_var(
        self,
        returns: pd.Series,
        confidence_levels: List[float],
        portfolio_value: float,
        n_simulations: int = 10000,
        **kwargs
    ) -> VaRResult:
        """Calculate Monte Carlo VaR using simulated returns."""
        clean_returns = returns.dropna()
        
        # Fit distribution parameters
        mean = clean_returns.mean()
        std = clean_returns.std()
        
        # Generate Monte Carlo simulations
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean, std, n_simulations)
        
        var_values = {}
        cvar_values = {}
        confidence_intervals = {}
        
        for conf in confidence_levels:
            # VaR calculation
            var_quantile = np.percentile(simulated_returns, (1 - conf) * 100)
            var = -var_quantile * portfolio_value
            var_values[f'var_{int(conf*100)}'] = var
            
            # CVaR calculation
            tail_returns = simulated_returns[simulated_returns <= var_quantile]
            cvar = -tail_returns.mean() * portfolio_value if len(tail_returns) > 0 else var
            cvar_values[f'cvar_{int(conf*100)}'] = cvar
            
            # Confidence intervals from simulation
            ci = self._monte_carlo_confidence_interval(
                simulated_returns, conf, portfolio_value
            )
            confidence_intervals[f'var_{int(conf*100)}'] = ci
        
        return VaRResult(
            var_95=var_values.get('var_95', 0),
            var_99=var_values.get('var_99', 0),
            cvar_95=cvar_values.get('cvar_95', 0),
            cvar_99=cvar_values.get('cvar_99', 0),
            method=VaRMethod.MONTE_CARLO,
            confidence_intervals=confidence_intervals,
            calculation_date=datetime.now(),
            lookback_days=len(clean_returns),
            portfolio_value=portfolio_value
        )
    
    def _bootstrap_confidence_interval(
        self,
        returns: pd.Series,
        confidence: float,
        portfolio_value: float,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrap method."""
        bootstrap_vars = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = returns.sample(n=len(returns), replace=True)
            var_quantile = np.percentile(bootstrap_sample, (1 - confidence) * 100)
            var = -var_quantile * portfolio_value
            bootstrap_vars.append(var)
        
        lower_ci = np.percentile(bootstrap_vars, 2.5)
        upper_ci = np.percentile(bootstrap_vars, 97.5)
        
        return (lower_ci, upper_ci)
    
    def _monte_carlo_confidence_interval(
        self,
        simulated_returns: np.ndarray,
        confidence: float,
        portfolio_value: float
    ) -> Tuple[float, float]:
        """Calculate confidence interval from Monte Carlo simulation."""
        # Split simulations into batches and calculate VaR for each
        batch_size = 1000
        batch_vars = []
        
        for i in range(0, len(simulated_returns), batch_size):
            batch = simulated_returns[i:i+batch_size]
            if len(batch) >= batch_size:
                var_quantile = np.percentile(batch, (1 - confidence) * 100)
                var = -var_quantile * portfolio_value
                batch_vars.append(var)
        
        if len(batch_vars) > 1:
            lower_ci = np.percentile(batch_vars, 2.5)
            upper_ci = np.percentile(batch_vars, 97.5)
        else:
            # Fallback to single estimate
            var_quantile = np.percentile(simulated_returns, (1 - confidence) * 100)
            var = -var_quantile * portfolio_value
            lower_ci = upper_ci = var
        
        return (lower_ci, upper_ci)
    
    def calculate_component_var(
        self,
        returns_matrix: pd.DataFrame,
        weights: pd.Series,
        method: VaRMethod = VaRMethod.HISTORICAL,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate component VaR for each asset in the portfolio.
        
        Args:
            returns_matrix: DataFrame with asset returns
            weights: Portfolio weights
            method: VaR calculation method
            confidence: Confidence level
            
        Returns:
            Dictionary with component VaR for each asset
        """
        portfolio_returns = (returns_matrix * weights).sum(axis=1)
        portfolio_var_result = self.calculate_var(
            portfolio_returns, method, [confidence]
        )
        portfolio_var = portfolio_var_result.var_95 if confidence == 0.95 else portfolio_var_result.var_99
        
        component_vars = {}
        
        for asset in returns_matrix.columns:
            # Calculate marginal VaR using finite difference
            epsilon = 0.001
            
            # Perturb weight slightly
            perturbed_weights = weights.copy()
            perturbed_weights[asset] += epsilon
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize
            
            perturbed_returns = (returns_matrix * perturbed_weights).sum(axis=1)
            perturbed_var_result = self.calculate_var(
                perturbed_returns, method, [confidence]
            )
            perturbed_var = perturbed_var_result.var_95 if confidence == 0.95 else perturbed_var_result.var_99
            
            # Marginal VaR
            marginal_var = (perturbed_var - portfolio_var) / epsilon
            
            # Component VaR
            component_vars[asset] = marginal_var * weights[asset]
        
        return component_vars
    
    def backtest_var(
        self,
        returns: pd.Series,
        var_estimates: pd.Series,
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Backtest VaR model performance.
        
        Args:
            returns: Actual returns
            var_estimates: VaR estimates
            confidence: Confidence level
            
        Returns:
            Dictionary with backtesting statistics
        """
        # Align series
        aligned_data = pd.concat([returns, var_estimates], axis=1, join='inner')
        aligned_data.columns = ['returns', 'var']
        
        # Calculate violations
        violations = aligned_data['returns'] < -aligned_data['var']
        violation_rate = violations.mean()
        expected_rate = 1 - confidence
        
        # Kupiec test for unconditional coverage
        n = len(violations)
        x = violations.sum()
        
        if x > 0 and x < n:
            lr_uc = -2 * np.log(
                (expected_rate ** x) * ((1 - expected_rate) ** (n - x)) /
                ((x / n) ** x) * ((1 - x / n) ** (n - x))
            )
        else:
            lr_uc = np.inf
        
        # Independence test (simplified)
        violation_clusters = self._count_violation_clusters(violations)
        
        return {
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'kupiec_test_statistic': lr_uc,
            'kupiec_p_value': 1 - stats.chi2.cdf(lr_uc, 1) if lr_uc != np.inf else 0,
            'violation_clusters': violation_clusters,
            'average_violation_size': aligned_data[violations]['returns'].mean() if violations.any() else 0
        }
    
    def _count_violation_clusters(self, violations: pd.Series) -> int:
        """Count clusters of consecutive violations."""
        clusters = 0
        in_cluster = False
        
        for violation in violations:
            if violation and not in_cluster:
                clusters += 1
                in_cluster = True
            elif not violation:
                in_cluster = False
        
        return clusters