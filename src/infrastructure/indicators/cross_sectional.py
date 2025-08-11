"""
Cross-Sectional Technical Indicators

This module contains cross-sectional features that compare securities
against each other or against market benchmarks.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.application.services.feature_engine import BaseIndicator


class RelativeStrength(BaseIndicator):
    """Relative Strength vs Benchmark"""
    
    def __init__(self):
        super().__init__("RelativeStrength", "Relative strength vs benchmark")
        self.set_parameter('benchmark_column', 'benchmark')
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        benchmark_col = kwargs.get('benchmark_column', self._parameters.get('benchmark_column', 'benchmark'))
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        if benchmark_col not in data.columns:
            raise ValueError(f"Benchmark column '{benchmark_col}' not found in data")
        
        close = data['close']
        benchmark = data[benchmark_col]
        
        # Calculate relative performance
        asset_return = close.pct_change(period)
        benchmark_return = benchmark.pct_change(period)
        
        relative_strength = asset_return - benchmark_return
        
        return relative_strength


class RelativeStrengthIndex(BaseIndicator):
    """Relative Strength Index vs Universe"""
    
    def __init__(self):
        super().__init__("RelativeStrengthIndex", "Relative strength index vs universe")
        self.set_parameter('period', 20)
        self.set_parameter('universe_columns', [])  # List of column names
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        universe_cols = kwargs.get('universe_columns', self._parameters.get('universe_columns', []))
        
        if not universe_cols:
            # If no universe specified, use all numeric columns except close
            universe_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                           if col != 'close']
        
        close = data['close']
        
        # Calculate returns for asset and universe
        asset_returns = close.pct_change(period)
        
        rs_values = pd.Series(index=data.index, dtype=float)
        
        for i in range(period, len(data)):
            # Get universe returns for this period
            universe_returns = []
            for col in universe_cols:
                if col in data.columns:
                    universe_return = data[col].iloc[i] / data[col].iloc[i-period] - 1
                    universe_returns.append(universe_return)
            
            if universe_returns:
                # Calculate percentile rank
                asset_ret = asset_returns.iloc[i]
                if not np.isnan(asset_ret):
                    rank = sum(1 for ret in universe_returns if ret < asset_ret)
                    rs_values.iloc[i] = rank / len(universe_returns) * 100
        
        return rs_values


class SectorMomentum(BaseIndicator):
    """Sector Momentum"""
    
    def __init__(self):
        super().__init__("SectorMomentum", "Sector momentum indicator")
        self.set_parameter('sector_column', 'sector')
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        sector_col = kwargs.get('sector_column', self._parameters.get('sector_column', 'sector'))
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        if sector_col not in data.columns:
            raise ValueError(f"Sector column '{sector_col}' not found in data")
        
        close = data['close']
        sectors = data[sector_col]
        
        sector_momentum = pd.Series(index=data.index, dtype=float)
        
        # Calculate momentum for each sector
        unique_sectors = sectors.dropna().unique()
        sector_returns = {}
        
        for sector in unique_sectors:
            sector_mask = sectors == sector
            if sector_mask.sum() > 0:
                # Calculate average sector return
                sector_prices = close[sector_mask]
                if len(sector_prices) > period:
                    sector_return = sector_prices.pct_change(period).mean()
                    sector_returns[sector] = sector_return
        
        # Assign sector momentum to each observation
        for i in range(len(data)):
            sector = sectors.iloc[i]
            if pd.notna(sector) and sector in sector_returns:
                sector_momentum.iloc[i] = sector_returns[sector].iloc[i] if hasattr(sector_returns[sector], 'iloc') else sector_returns[sector]
        
        return sector_momentum


class CrossSectionalRank(BaseIndicator):
    """Cross-Sectional Rank"""
    
    def __init__(self):
        super().__init__("CrossSectionalRank", "Cross-sectional percentile rank")
        self.set_parameter('universe_columns', [])
        self.set_parameter('ascending', True)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        universe_cols = kwargs.get('universe_columns', self._parameters.get('universe_columns', []))
        ascending = kwargs.get('ascending', self._parameters.get('ascending', True))
        
        if not universe_cols:
            # Use all numeric columns
            universe_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        close = data['close']
        rank_series = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(data)):
            # Get cross-sectional values
            cross_section = []
            for col in universe_cols:
                if col in data.columns and not pd.isna(data[col].iloc[i]):
                    cross_section.append(data[col].iloc[i])
            
            if cross_section and not pd.isna(close.iloc[i]):
                # Calculate percentile rank
                if ascending:
                    rank = sum(1 for val in cross_section if val < close.iloc[i])
                else:
                    rank = sum(1 for val in cross_section if val > close.iloc[i])
                
                rank_series.iloc[i] = rank / len(cross_section) * 100
        
        return rank_series


class RelativeVolatility(BaseIndicator):
    """Relative Volatility"""
    
    def __init__(self):
        super().__init__("RelativeVolatility", "Relative volatility vs benchmark")
        self.set_parameter('benchmark_column', 'benchmark')
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        benchmark_col = kwargs.get('benchmark_column', self._parameters.get('benchmark_column', 'benchmark'))
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        if benchmark_col not in data.columns:
            raise ValueError(f"Benchmark column '{benchmark_col}' not found in data")
        
        close = data['close']
        benchmark = data[benchmark_col]
        
        # Calculate volatilities
        asset_vol = close.pct_change().rolling(window=period).std()
        benchmark_vol = benchmark.pct_change().rolling(window=period).std()
        
        relative_vol = asset_vol / benchmark_vol
        
        return relative_vol


class BetaStability(BaseIndicator):
    """Beta Stability"""
    
    def __init__(self):
        super().__init__("BetaStability", "Beta stability measure")
        self.set_parameter('benchmark_column', 'benchmark')
        self.set_parameter('period', 60)
        self.set_parameter('sub_period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        benchmark_col = kwargs.get('benchmark_column', self._parameters.get('benchmark_column', 'benchmark'))
        period = kwargs.get('period', self._parameters.get('period', 60))
        sub_period = kwargs.get('sub_period', self._parameters.get('sub_period', 20))
        
        if benchmark_col not in data.columns:
            raise ValueError(f"Benchmark column '{benchmark_col}' not found in data")
        
        close = data['close']
        benchmark = data[benchmark_col]
        
        # Calculate returns
        asset_returns = close.pct_change()
        benchmark_returns = benchmark.pct_change()
        
        beta_stability = pd.Series(index=data.index, dtype=float)
        
        for i in range(period, len(data)):
            # Calculate multiple sub-period betas
            sub_betas = []
            for j in range(0, period - sub_period + 1, sub_period // 2):
                start_idx = i - period + j
                end_idx = start_idx + sub_period
                
                if end_idx <= i:
                    asset_ret_sub = asset_returns.iloc[start_idx:end_idx]
                    bench_ret_sub = benchmark_returns.iloc[start_idx:end_idx]
                    
                    # Calculate beta for sub-period
                    covariance = asset_ret_sub.cov(bench_ret_sub)
                    variance = bench_ret_sub.var()
                    
                    if variance > 0:
                        beta = covariance / variance
                        sub_betas.append(beta)
            
            # Calculate stability as inverse of beta standard deviation
            if len(sub_betas) > 1:
                beta_std = np.std(sub_betas)
                beta_stability.iloc[i] = 1 / (1 + beta_std) if beta_std > 0 else 1
        
        return beta_stability


class InformationRatio(BaseIndicator):
    """Information Ratio"""
    
    def __init__(self):
        super().__init__("InformationRatio", "Information ratio vs benchmark")
        self.set_parameter('benchmark_column', 'benchmark')
        self.set_parameter('period', 60)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        benchmark_col = kwargs.get('benchmark_column', self._parameters.get('benchmark_column', 'benchmark'))
        period = kwargs.get('period', self._parameters.get('period', 60))
        
        if benchmark_col not in data.columns:
            raise ValueError(f"Benchmark column '{benchmark_col}' not found in data")
        
        close = data['close']
        benchmark = data[benchmark_col]
        
        # Calculate returns
        asset_returns = close.pct_change()
        benchmark_returns = benchmark.pct_change()
        
        # Active returns
        active_returns = asset_returns - benchmark_returns
        
        # Information ratio
        active_return_mean = active_returns.rolling(window=period).mean()
        tracking_error = active_returns.rolling(window=period).std()
        
        information_ratio = active_return_mean / tracking_error
        
        return information_ratio


class CorrelationStability(BaseIndicator):
    """Correlation Stability"""
    
    def __init__(self):
        super().__init__("CorrelationStability", "Correlation stability with benchmark")
        self.set_parameter('benchmark_column', 'benchmark')
        self.set_parameter('period', 60)
        self.set_parameter('sub_period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        benchmark_col = kwargs.get('benchmark_column', self._parameters.get('benchmark_column', 'benchmark'))
        period = kwargs.get('period', self._parameters.get('period', 60))
        sub_period = kwargs.get('sub_period', self._parameters.get('sub_period', 20))
        
        if benchmark_col not in data.columns:
            raise ValueError(f"Benchmark column '{benchmark_col}' not found in data")
        
        close = data['close']
        benchmark = data[benchmark_col]
        
        # Calculate returns
        asset_returns = close.pct_change()
        benchmark_returns = benchmark.pct_change()
        
        corr_stability = pd.Series(index=data.index, dtype=float)
        
        for i in range(period, len(data)):
            # Calculate multiple sub-period correlations
            sub_corrs = []
            for j in range(0, period - sub_period + 1, sub_period // 2):
                start_idx = i - period + j
                end_idx = start_idx + sub_period
                
                if end_idx <= i:
                    asset_ret_sub = asset_returns.iloc[start_idx:end_idx]
                    bench_ret_sub = benchmark_returns.iloc[start_idx:end_idx]
                    
                    # Calculate correlation for sub-period
                    correlation = asset_ret_sub.corr(bench_ret_sub)
                    if not np.isnan(correlation):
                        sub_corrs.append(correlation)
            
            # Calculate stability as inverse of correlation standard deviation
            if len(sub_corrs) > 1:
                corr_std = np.std(sub_corrs)
                corr_stability.iloc[i] = 1 / (1 + corr_std) if corr_std > 0 else 1
        
        return corr_stability