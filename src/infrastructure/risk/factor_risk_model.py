"""
Factor risk model for exposure analysis and attribution.

This module provides comprehensive factor risk modeling capabilities including:
- Multi-factor risk model construction
- Factor exposure analysis
- Risk attribution and decomposition
- Factor return forecasting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import warnings

from ...domain.entities import Portfolio, Position
from ...domain.exceptions import ValidationError


class FactorType(Enum):
    """Types of risk factors."""
    MARKET = "market"
    SIZE = "size"
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    SECTOR = "sector"
    COUNTRY = "country"
    CURRENCY = "currency"
    STYLE = "style"
    MACRO = "macro"


@dataclass
class FactorExposure:
    """Factor exposure for a single asset or portfolio."""
    asset_id: str
    exposures: Dict[str, float]
    specific_risk: float
    total_risk: float
    timestamp: datetime
    
    def get_exposure(self, factor_name: str) -> float:
        """Get exposure to a specific factor."""
        return self.exposures.get(factor_name, 0.0)
    
    def get_factor_contribution(self, factor_name: str, factor_return: float) -> float:
        """Calculate factor contribution to return."""
        return self.get_exposure(factor_name) * factor_return


@dataclass
class FactorReturn:
    """Factor return data."""
    factor_name: str
    returns: pd.Series
    volatility: float
    sharpe_ratio: float
    description: str
    factor_type: FactorType


@dataclass
class RiskAttribution:
    """Risk attribution results."""
    portfolio_id: str
    total_risk: float
    factor_contributions: Dict[str, float]
    specific_risk: float
    factor_correlations: pd.DataFrame
    timestamp: datetime
    
    def get_factor_contribution_percent(self, factor_name: str) -> float:
        """Get factor contribution as percentage of total risk."""
        if self.total_risk == 0:
            return 0.0
        return (self.factor_contributions.get(factor_name, 0.0) / self.total_risk) * 100


class FactorRiskModel:
    """
    Multi-factor risk model for portfolio risk analysis.
    
    This class implements a comprehensive factor risk model that can be used
    for risk attribution, exposure analysis, and risk forecasting.
    """
    
    def __init__(
        self,
        lookback_days: int = 252,
        min_observations: int = 60,
        regularization_alpha: float = 0.1
    ):
        """
        Initialize factor risk model.
        
        Args:
            lookback_days: Number of days for factor estimation
            min_observations: Minimum observations required for factor estimation
            regularization_alpha: Regularization parameter for factor regression
        """
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.regularization_alpha = regularization_alpha
        
        # Model components
        self.factors: Dict[str, FactorReturn] = {}
        self.factor_loadings: Dict[str, pd.Series] = {}
        self.factor_covariance: Optional[pd.DataFrame] = None
        self.specific_risks: Dict[str, float] = {}
        
        # Model metadata
        self.last_update: Optional[datetime] = None
        self.is_fitted: bool = False
        
        # Initialize common factors
        self._initialize_common_factors()
    
    def _initialize_common_factors(self) -> None:
        """Initialize common risk factors."""
        # This would typically load factor data from external sources
        # For demonstration, we'll create synthetic factor data
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=self.lookback_days),
            end=datetime.now(),
            freq='D'
        )
        
        np.random.seed(42)
        
        # Market factor (similar to market index)
        market_returns = pd.Series(
            np.random.normal(0.0005, 0.012, len(dates)),
            index=dates,
            name='market'
        )
        
        # Size factor (small minus big)
        size_returns = pd.Series(
            np.random.normal(0.0002, 0.008, len(dates)),
            index=dates,
            name='size'
        )
        
        # Value factor (high minus low book-to-market)
        value_returns = pd.Series(
            np.random.normal(0.0001, 0.006, len(dates)),
            index=dates,
            name='value'
        )
        
        # Momentum factor
        momentum_returns = pd.Series(
            np.random.normal(0.0003, 0.010, len(dates)),
            index=dates,
            name='momentum'
        )
        
        # Quality factor
        quality_returns = pd.Series(
            np.random.normal(0.0001, 0.005, len(dates)),
            index=dates,
            name='quality'
        )
        
        # Volatility factor
        volatility_returns = pd.Series(
            np.random.normal(-0.0002, 0.015, len(dates)),
            index=dates,
            name='volatility'
        )
        
        # Add factors to model
        self.factors = {
            'market': FactorReturn(
                factor_name='market',
                returns=market_returns,
                volatility=market_returns.std() * np.sqrt(252),
                sharpe_ratio=market_returns.mean() / market_returns.std() * np.sqrt(252),
                description='Market factor representing broad market movements',
                factor_type=FactorType.MARKET
            ),
            'size': FactorReturn(
                factor_name='size',
                returns=size_returns,
                volatility=size_returns.std() * np.sqrt(252),
                sharpe_ratio=size_returns.mean() / size_returns.std() * np.sqrt(252),
                description='Size factor (small cap vs large cap)',
                factor_type=FactorType.SIZE
            ),
            'value': FactorReturn(
                factor_name='value',
                returns=value_returns,
                volatility=value_returns.std() * np.sqrt(252),
                sharpe_ratio=value_returns.mean() / value_returns.std() * np.sqrt(252),
                description='Value factor (value vs growth)',
                factor_type=FactorType.VALUE
            ),
            'momentum': FactorReturn(
                factor_name='momentum',
                returns=momentum_returns,
                volatility=momentum_returns.std() * np.sqrt(252),
                sharpe_ratio=momentum_returns.mean() / momentum_returns.std() * np.sqrt(252),
                description='Momentum factor (winners vs losers)',
                factor_type=FactorType.MOMENTUM
            ),
            'quality': FactorReturn(
                factor_name='quality',
                returns=quality_returns,
                volatility=quality_returns.std() * np.sqrt(252),
                sharpe_ratio=quality_returns.mean() / quality_returns.std() * np.sqrt(252),
                description='Quality factor (high quality vs low quality)',
                factor_type=FactorType.QUALITY
            ),
            'volatility': FactorReturn(
                factor_name='volatility',
                returns=volatility_returns,
                volatility=volatility_returns.std() * np.sqrt(252),
                sharpe_ratio=volatility_returns.mean() / volatility_returns.std() * np.sqrt(252),
                description='Volatility factor (low vol vs high vol)',
                factor_type=FactorType.VOLATILITY
            )
        }
    
    def add_custom_factor(
        self,
        factor_name: str,
        factor_returns: pd.Series,
        description: str,
        factor_type: FactorType = FactorType.STYLE
    ) -> None:
        """Add a custom factor to the model."""
        if len(factor_returns) < self.min_observations:
            raise ValidationError(f"Factor {factor_name} has insufficient observations")
        
        factor_return = FactorReturn(
            factor_name=factor_name,
            returns=factor_returns,
            volatility=factor_returns.std() * np.sqrt(252),
            sharpe_ratio=factor_returns.mean() / factor_returns.std() * np.sqrt(252),
            description=description,
            factor_type=factor_type
        )
        
        self.factors[factor_name] = factor_return
        self.is_fitted = False  # Need to refit model
    
    def fit_model(
        self,
        asset_returns: pd.DataFrame,
        sector_data: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Fit the factor risk model to asset returns.
        
        Args:
            asset_returns: DataFrame with asset returns (assets as columns)
            sector_data: Optional mapping of assets to sectors
        """
        if asset_returns.empty:
            raise ValidationError("Asset returns cannot be empty")
        
        # Align factor and asset returns
        factor_returns_df = self._get_factor_returns_dataframe()
        aligned_data = pd.concat([asset_returns, factor_returns_df], axis=1, join='inner')
        
        if len(aligned_data) < self.min_observations:
            raise ValidationError("Insufficient aligned observations for model fitting")
        
        # Separate asset and factor returns
        asset_cols = asset_returns.columns
        factor_cols = factor_returns_df.columns
        
        asset_data = aligned_data[asset_cols]
        factor_data = aligned_data[factor_cols]
        
        # Add sector factors if provided
        if sector_data:
            sector_factors = self._create_sector_factors(asset_data, sector_data)
            factor_data = pd.concat([factor_data, sector_factors], axis=1)
        
        # Fit factor loadings for each asset
        self.factor_loadings = {}
        self.specific_risks = {}
        
        for asset in asset_cols:
            loadings, specific_risk = self._fit_asset_factor_model(
                asset_data[asset], factor_data
            )
            self.factor_loadings[asset] = loadings
            self.specific_risks[asset] = specific_risk
        
        # Calculate factor covariance matrix
        self.factor_covariance = factor_data.cov() * 252  # Annualized
        
        self.is_fitted = True
        self.last_update = datetime.now()
    
    def _get_factor_returns_dataframe(self) -> pd.DataFrame:
        """Get factor returns as a DataFrame."""
        factor_data = {}
        for factor_name, factor in self.factors.items():
            factor_data[factor_name] = factor.returns
        
        return pd.DataFrame(factor_data)
    
    def _create_sector_factors(
        self,
        asset_returns: pd.DataFrame,
        sector_data: Dict[str, str]
    ) -> pd.DataFrame:
        """Create sector factors from asset returns."""
        sectors = set(sector_data.values())
        sector_factors = pd.DataFrame(index=asset_returns.index)
        
        for sector in sectors:
            sector_assets = [asset for asset, sec in sector_data.items() if sec == sector]
            sector_assets = [asset for asset in sector_assets if asset in asset_returns.columns]
            
            if sector_assets:
                # Equal-weighted sector return
                sector_return = asset_returns[sector_assets].mean(axis=1)
                sector_factors[f'sector_{sector.lower()}'] = sector_return
        
        return sector_factors
    
    def _fit_asset_factor_model(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> Tuple[pd.Series, float]:
        """Fit factor model for a single asset."""
        # Remove NaN values
        aligned_data = pd.concat([asset_returns, factor_returns], axis=1).dropna()
        
        if len(aligned_data) < self.min_observations:
            warnings.warn(f"Insufficient data for asset {asset_returns.name}")
            return pd.Series(index=factor_returns.columns, dtype=float).fillna(0), 0.1
        
        y = aligned_data.iloc[:, 0]  # Asset returns
        X = aligned_data.iloc[:, 1:]  # Factor returns
        
        # Use Ridge regression for regularization
        model = Ridge(alpha=self.regularization_alpha, fit_intercept=True)
        model.fit(X, y)
        
        # Calculate factor loadings
        loadings = pd.Series(model.coef_, index=X.columns)
        
        # Calculate specific risk (residual volatility)
        predictions = model.predict(X)
        residuals = y - predictions
        specific_risk = residuals.std() * np.sqrt(252)  # Annualized
        
        return loadings, specific_risk
    
    def calculate_portfolio_exposures(
        self,
        portfolio_weights: pd.Series
    ) -> FactorExposure:
        """Calculate portfolio factor exposures."""
        if not self.is_fitted:
            raise ValidationError("Model must be fitted before calculating exposures")
        
        # Align weights with available assets
        available_assets = set(self.factor_loadings.keys())
        portfolio_assets = set(portfolio_weights.index)
        common_assets = available_assets.intersection(portfolio_assets)
        
        if not common_assets:
            raise ValidationError("No common assets between portfolio and model")
        
        # Calculate weighted factor exposures
        portfolio_exposures = {}
        
        for factor_name in self.factors.keys():
            exposure = 0.0
            for asset in common_assets:
                weight = portfolio_weights[asset]
                asset_exposure = self.factor_loadings[asset].get(factor_name, 0.0)
                exposure += weight * asset_exposure
            portfolio_exposures[factor_name] = exposure
        
        # Calculate portfolio specific risk
        specific_risk_squared = 0.0
        for asset in common_assets:
            weight = portfolio_weights[asset]
            asset_specific_risk = self.specific_risks.get(asset, 0.1)
            specific_risk_squared += (weight ** 2) * (asset_specific_risk ** 2)
        
        specific_risk = np.sqrt(specific_risk_squared)
        
        # Calculate total portfolio risk
        factor_risk = self._calculate_factor_risk(portfolio_exposures)
        total_risk = np.sqrt(factor_risk ** 2 + specific_risk ** 2)
        
        return FactorExposure(
            asset_id='portfolio',
            exposures=portfolio_exposures,
            specific_risk=specific_risk,
            total_risk=total_risk,
            timestamp=datetime.now()
        )
    
    def _calculate_factor_risk(self, exposures: Dict[str, float]) -> float:
        """Calculate factor risk contribution."""
        if self.factor_covariance is None:
            return 0.0
        
        # Convert exposures to vector
        factor_names = list(self.factor_covariance.index)
        exposure_vector = np.array([exposures.get(name, 0.0) for name in factor_names])
        
        # Calculate factor risk: sqrt(w' * Cov * w)
        factor_variance = np.dot(exposure_vector, np.dot(self.factor_covariance.values, exposure_vector))
        return np.sqrt(max(0, factor_variance))
    
    def calculate_risk_attribution(
        self,
        portfolio_weights: pd.Series
    ) -> RiskAttribution:
        """Calculate detailed risk attribution."""
        portfolio_exposure = self.calculate_portfolio_exposures(portfolio_weights)
        
        # Calculate marginal risk contributions
        factor_contributions = {}
        
        for factor_name, exposure in portfolio_exposure.exposures.items():
            if factor_name in self.factor_covariance.index:
                # Marginal contribution to risk
                factor_variance = self.factor_covariance.loc[factor_name, factor_name]
                marginal_risk = exposure * factor_variance / portfolio_exposure.total_risk
                factor_contributions[factor_name] = marginal_risk
        
        return RiskAttribution(
            portfolio_id='portfolio',
            total_risk=portfolio_exposure.total_risk,
            factor_contributions=factor_contributions,
            specific_risk=portfolio_exposure.specific_risk,
            factor_correlations=self.factor_covariance.corr(),
            timestamp=datetime.now()
        )
    
    def forecast_factor_returns(
        self,
        horizon_days: int = 30,
        method: str = 'historical'
    ) -> Dict[str, float]:
        """Forecast factor returns."""
        forecasts = {}
        
        for factor_name, factor in self.factors.items():
            if method == 'historical':
                # Simple historical average
                forecast = factor.returns.tail(self.lookback_days).mean() * horizon_days
            elif method == 'ewma':
                # Exponentially weighted moving average
                weights = np.exp(-np.arange(len(factor.returns)) * 0.01)
                weights = weights[::-1] / weights.sum()
                forecast = np.average(factor.returns.tail(len(weights)), weights=weights) * horizon_days
            else:
                forecast = 0.0
            
            forecasts[factor_name] = forecast
        
        return forecasts
    
    def stress_test_factors(
        self,
        portfolio_weights: pd.Series,
        factor_shocks: Dict[str, float]
    ) -> Dict[str, float]:
        """Stress test portfolio with factor shocks."""
        portfolio_exposure = self.calculate_portfolio_exposures(portfolio_weights)
        
        results = {}
        
        for factor_name, shock in factor_shocks.items():
            if factor_name in portfolio_exposure.exposures:
                exposure = portfolio_exposure.exposures[factor_name]
                impact = exposure * shock
                results[factor_name] = impact
        
        results['total_impact'] = sum(results.values())
        
        return results
    
    def get_factor_summary(self) -> pd.DataFrame:
        """Get summary statistics for all factors."""
        data = []
        
        for factor_name, factor in self.factors.items():
            data.append({
                'factor_name': factor_name,
                'factor_type': factor.factor_type.value,
                'mean_return': factor.returns.mean() * 252,  # Annualized
                'volatility': factor.volatility,
                'sharpe_ratio': factor.sharpe_ratio,
                'description': factor.description
            })
        
        return pd.DataFrame(data)
    
    def get_asset_exposures(self, asset_id: str) -> Optional[pd.Series]:
        """Get factor exposures for a specific asset."""
        return self.factor_loadings.get(asset_id)
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostic information."""
        if not self.is_fitted:
            return {'status': 'not_fitted'}
        
        # Calculate R-squared for each asset
        r_squared_values = []
        for asset_id in self.factor_loadings.keys():
            # This would typically be calculated during fitting
            # For now, use a placeholder
            r_squared_values.append(0.75)  # Typical R-squared for factor models
        
        avg_r_squared = np.mean(r_squared_values) if r_squared_values else 0
        
        # Factor correlation analysis
        factor_corr = self.factor_covariance.corr() if self.factor_covariance is not None else pd.DataFrame()
        max_correlation = factor_corr.abs().values[np.triu_indices_from(factor_corr.values, k=1)].max() if not factor_corr.empty else 0
        
        return {
            'status': 'fitted',
            'last_update': self.last_update,
            'num_factors': len(self.factors),
            'num_assets': len(self.factor_loadings),
            'avg_r_squared': avg_r_squared,
            'max_factor_correlation': max_correlation,
            'lookback_days': self.lookback_days,
            'regularization_alpha': self.regularization_alpha
        }
    
    def export_model(self, filepath: str) -> None:
        """Export model to file."""
        import pickle
        
        model_data = {
            'factors': self.factors,
            'factor_loadings': self.factor_loadings,
            'factor_covariance': self.factor_covariance,
            'specific_risks': self.specific_risks,
            'last_update': self.last_update,
            'is_fitted': self.is_fitted,
            'lookback_days': self.lookback_days,
            'min_observations': self.min_observations,
            'regularization_alpha': self.regularization_alpha
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.factors = model_data['factors']
        self.factor_loadings = model_data['factor_loadings']
        self.factor_covariance = model_data['factor_covariance']
        self.specific_risks = model_data['specific_risks']
        self.last_update = model_data['last_update']
        self.is_fitted = model_data['is_fitted']
        self.lookback_days = model_data['lookback_days']
        self.min_observations = model_data['min_observations']
        self.regularization_alpha = model_data['regularization_alpha']