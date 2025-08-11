"""
Performance Attribution Analysis for quantitative strategies.

Provides detailed attribution analysis including factor-based attribution,
sector attribution, and style attribution.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ...domain.exceptions import ValidationError


@dataclass
class AttributionResult:
    """Results from performance attribution analysis."""
    
    # Factor attribution
    factor_returns: Dict[str, float]
    factor_exposures: Dict[str, float]
    factor_contributions: Dict[str, float]
    
    # Style attribution
    style_returns: Dict[str, float]
    style_exposures: Dict[str, float]
    
    # Sector attribution
    sector_returns: Dict[str, float]
    sector_weights: Dict[str, float]
    sector_contributions: Dict[str, float]
    
    # Selection and allocation effects
    selection_effect: float
    allocation_effect: float
    interaction_effect: float
    
    # Residual analysis
    alpha: float
    residual_return: float
    r_squared: float
    
    # Statistical measures
    tracking_error: float
    information_ratio: float
    
    timestamp: datetime


@dataclass
class BrinsonAttributionResult:
    """Results from Brinson attribution analysis."""
    
    # Portfolio and benchmark weights/returns
    portfolio_weights: Dict[str, float]
    benchmark_weights: Dict[str, float]
    portfolio_returns: Dict[str, float]
    benchmark_returns: Dict[str, float]
    
    # Attribution effects
    allocation_effect: Dict[str, float]
    selection_effect: Dict[str, float]
    interaction_effect: Dict[str, float]
    
    # Total effects
    total_allocation: float
    total_selection: float
    total_interaction: float
    total_active_return: float
    
    timestamp: datetime


class PerformanceAttributionAnalyzer:
    """Analyzes performance attribution using multiple methodologies."""
    
    def __init__(self):
        self.factor_models = {}
        self.benchmark_data = {}
    
    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None
    ) -> AttributionResult:
        """
        Perform factor-based attribution analysis.
        
        Args:
            portfolio_returns: Portfolio return series
            factor_returns: DataFrame with factor returns (columns = factors)
            benchmark_returns: Optional benchmark return series
            
        Returns:
            AttributionResult with factor attribution details
        """
        if len(portfolio_returns) == 0:
            raise ValidationError("Portfolio returns cannot be empty")
        
        if factor_returns.empty:
            raise ValidationError("Factor returns cannot be empty")
        
        # Align data
        aligned_data = pd.concat([
            portfolio_returns.rename('portfolio'),
            factor_returns,
            benchmark_returns.rename('benchmark') if benchmark_returns is not None else pd.Series(dtype=float)
        ], axis=1).dropna()
        
        if len(aligned_data) < 10:
            raise ValidationError("Insufficient data for attribution analysis")
        
        # Prepare regression data
        y = aligned_data['portfolio'].values
        X = aligned_data[factor_returns.columns].values
        
        # Fit factor model
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate factor exposures (betas)
        factor_exposures = dict(zip(factor_returns.columns, model.coef_))
        
        # Calculate factor contributions
        factor_contributions = {}
        for i, factor in enumerate(factor_returns.columns):
            factor_mean_return = aligned_data[factor].mean()
            factor_contributions[factor] = factor_exposures[factor] * factor_mean_return
        
        # Calculate alpha and residuals
        alpha = model.intercept_
        predicted_returns = model.predict(X)
        residuals = y - predicted_returns
        residual_return = np.mean(residuals)
        
        # R-squared
        r_squared = model.score(X, y)
        
        # Calculate tracking error and information ratio
        if benchmark_returns is not None:
            excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
            tracking_error = excess_returns.std() * np.sqrt(252)  # Annualized
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            tracking_error = 0.0
            information_ratio = 0.0
        
        # Factor returns (average over period)
        factor_returns_dict = {
            factor: aligned_data[factor].mean() 
            for factor in factor_returns.columns
        }
        
        return AttributionResult(
            factor_returns=factor_returns_dict,
            factor_exposures=factor_exposures,
            factor_contributions=factor_contributions,
            style_returns={},  # Will be populated by style attribution
            style_exposures={},
            sector_returns={},  # Will be populated by sector attribution
            sector_weights={},
            sector_contributions={},
            selection_effect=0.0,  # Will be calculated separately
            allocation_effect=0.0,
            interaction_effect=0.0,
            alpha=alpha,
            residual_return=residual_return,
            r_squared=r_squared,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            timestamp=datetime.now()
        )
    
    def brinson_attribution(
        self,
        portfolio_weights: Dict[str, float],
        benchmark_weights: Dict[str, float],
        portfolio_returns: Dict[str, float],
        benchmark_returns: Dict[str, float]
    ) -> BrinsonAttributionResult:
        """
        Perform Brinson attribution analysis.
        
        Args:
            portfolio_weights: Portfolio weights by sector/asset
            benchmark_weights: Benchmark weights by sector/asset
            portfolio_returns: Portfolio returns by sector/asset
            benchmark_returns: Benchmark returns by sector/asset
            
        Returns:
            BrinsonAttributionResult with attribution effects
        """
        # Ensure all sectors are present in all dictionaries
        all_sectors = set(portfolio_weights.keys()) | set(benchmark_weights.keys()) | \
                     set(portfolio_returns.keys()) | set(benchmark_returns.keys())
        
        # Fill missing values with zeros
        for sector in all_sectors:
            portfolio_weights.setdefault(sector, 0.0)
            benchmark_weights.setdefault(sector, 0.0)
            portfolio_returns.setdefault(sector, 0.0)
            benchmark_returns.setdefault(sector, 0.0)
        
        allocation_effect = {}
        selection_effect = {}
        interaction_effect = {}
        
        for sector in all_sectors:
            wp = portfolio_weights[sector]
            wb = benchmark_weights[sector]
            rp = portfolio_returns[sector]
            rb = benchmark_returns[sector]
            
            # Brinson attribution formulas
            allocation_effect[sector] = (wp - wb) * rb
            selection_effect[sector] = wb * (rp - rb)
            interaction_effect[sector] = (wp - wb) * (rp - rb)
        
        # Calculate totals
        total_allocation = sum(allocation_effect.values())
        total_selection = sum(selection_effect.values())
        total_interaction = sum(interaction_effect.values())
        total_active_return = total_allocation + total_selection + total_interaction
        
        return BrinsonAttributionResult(
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect,
            total_allocation=total_allocation,
            total_selection=total_selection,
            total_interaction=total_interaction,
            total_active_return=total_active_return,
            timestamp=datetime.now()
        )
    
    def style_attribution(
        self,
        portfolio_returns: pd.Series,
        style_factors: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Perform style-based attribution analysis.
        
        Args:
            portfolio_returns: Portfolio return series
            style_factors: DataFrame with style factor returns (value, growth, momentum, etc.)
            
        Returns:
            Dictionary with style exposures and contributions
        """
        # Align data
        aligned_data = pd.concat([
            portfolio_returns.rename('portfolio'),
            style_factors
        ], axis=1).dropna()
        
        if len(aligned_data) < 10:
            raise ValidationError("Insufficient data for style attribution")
        
        # Fit style model
        y = aligned_data['portfolio'].values
        X = aligned_data[style_factors.columns].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate style exposures and contributions
        style_exposures = dict(zip(style_factors.columns, model.coef_))
        style_contributions = {}
        
        for i, style in enumerate(style_factors.columns):
            style_mean_return = aligned_data[style].mean()
            style_contributions[style] = style_exposures[style] * style_mean_return
        
        return {
            'exposures': style_exposures,
            'contributions': style_contributions,
            'alpha': model.intercept_,
            'r_squared': model.score(X, y)
        }
    
    def rolling_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int = 252,
        benchmark_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Perform rolling attribution analysis.
        
        Args:
            portfolio_returns: Portfolio return series
            factor_returns: DataFrame with factor returns
            window: Rolling window size
            benchmark_returns: Optional benchmark returns
            
        Returns:
            DataFrame with rolling attribution metrics
        """
        if len(portfolio_returns) < window * 2:
            raise ValidationError(f"Insufficient data for rolling attribution (need at least {window * 2} periods)")
        
        # Align data
        aligned_data = pd.concat([
            portfolio_returns.rename('portfolio'),
            factor_returns,
            benchmark_returns.rename('benchmark') if benchmark_returns is not None else pd.Series(dtype=float)
        ], axis=1).dropna()
        
        results = []
        
        for i in range(window, len(aligned_data)):
            window_data = aligned_data.iloc[i-window:i]
            
            # Fit factor model for this window
            y = window_data['portfolio'].values
            X = window_data[factor_returns.columns].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate metrics for this window
            result = {
                'date': aligned_data.index[i],
                'alpha': model.intercept_,
                'r_squared': model.score(X, y)
            }
            
            # Add factor exposures
            for j, factor in enumerate(factor_returns.columns):
                result[f'{factor}_exposure'] = model.coef_[j]
                result[f'{factor}_contribution'] = model.coef_[j] * window_data[factor].mean()
            
            # Add tracking error and information ratio if benchmark available
            if benchmark_returns is not None:
                excess_returns = window_data['portfolio'] - window_data['benchmark']
                result['tracking_error'] = excess_returns.std() * np.sqrt(252)
                result['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            
            results.append(result)
        
        return pd.DataFrame(results).set_index('date')
    
    def sector_attribution(
        self,
        portfolio_holdings: Dict[str, Dict[str, float]],  # {sector: {symbol: weight}}
        sector_returns: Dict[str, float],
        benchmark_sector_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Perform sector-based attribution analysis.
        
        Args:
            portfolio_holdings: Portfolio holdings by sector and symbol
            sector_returns: Returns by sector
            benchmark_sector_weights: Benchmark weights by sector
            
        Returns:
            Dictionary with sector attribution results
        """
        # Calculate portfolio sector weights
        portfolio_sector_weights = {}
        for sector, holdings in portfolio_holdings.items():
            portfolio_sector_weights[sector] = sum(holdings.values())
        
        # Ensure all sectors are represented
        all_sectors = set(portfolio_sector_weights.keys()) | set(benchmark_sector_weights.keys())
        
        for sector in all_sectors:
            portfolio_sector_weights.setdefault(sector, 0.0)
            benchmark_sector_weights.setdefault(sector, 0.0)
            sector_returns.setdefault(sector, 0.0)
        
        # Calculate attribution effects
        allocation_effects = {}
        selection_effects = {}
        
        for sector in all_sectors:
            wp = portfolio_sector_weights[sector]
            wb = benchmark_sector_weights[sector]
            rs = sector_returns[sector]
            
            # Allocation effect: (portfolio weight - benchmark weight) * sector return
            allocation_effects[sector] = (wp - wb) * rs
            
            # Selection effect would require individual stock returns within sectors
            # For now, we'll set it to zero or calculate based on available data
            selection_effects[sector] = 0.0
        
        return {
            'portfolio_weights': portfolio_sector_weights,
            'benchmark_weights': benchmark_sector_weights,
            'sector_returns': sector_returns,
            'allocation_effects': allocation_effects,
            'selection_effects': selection_effects,
            'total_allocation': sum(allocation_effects.values()),
            'total_selection': sum(selection_effects.values())
        }
    
    def generate_attribution_report(self, attribution: AttributionResult) -> str:
        """Generate a comprehensive attribution report."""
        
        report = []
        report.append("=== PERFORMANCE ATTRIBUTION REPORT ===")
        report.append(f"Analysis Date: {attribution.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Factor attribution
        if attribution.factor_returns:
            report.append("=== FACTOR ATTRIBUTION ===")
            report.append(f"{'Factor':<15} {'Return':<10} {'Exposure':<10} {'Contribution':<12}")
            report.append("-" * 50)
            
            for factor in attribution.factor_returns.keys():
                factor_return = attribution.factor_returns[factor]
                exposure = attribution.factor_exposures.get(factor, 0.0)
                contribution = attribution.factor_contributions.get(factor, 0.0)
                
                report.append(f"{factor:<15} {factor_return:>9.2%} {exposure:>9.2f} {contribution:>11.2%}")
            
            report.append("")
        
        # Alpha and residual analysis
        report.append("=== ALPHA ANALYSIS ===")
        report.append(f"Alpha:              {attribution.alpha:>10.2%}")
        report.append(f"Residual Return:    {attribution.residual_return:>10.2%}")
        report.append(f"R-Squared:          {attribution.r_squared:>10.2%}")
        report.append(f"Tracking Error:     {attribution.tracking_error:>10.2%}")
        report.append(f"Information Ratio:  {attribution.information_ratio:>10.2f}")
        report.append("")
        
        # Allocation and selection effects
        if attribution.allocation_effect != 0 or attribution.selection_effect != 0:
            report.append("=== ALLOCATION & SELECTION EFFECTS ===")
            report.append(f"Allocation Effect:  {attribution.allocation_effect:>10.2%}")
            report.append(f"Selection Effect:   {attribution.selection_effect:>10.2%}")
            report.append(f"Interaction Effect: {attribution.interaction_effect:>10.2%}")
            report.append("")
        
        return "\n".join(report)