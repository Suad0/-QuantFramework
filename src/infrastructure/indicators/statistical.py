"""
Statistical Technical Indicators

This module contains statistical measures and mathematical indicators
for technical analysis and quantitative research.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.application.services.feature_engine import BaseIndicator

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SMA(BaseIndicator):
    """Simple Moving Average"""
    
    def __init__(self):
        super().__init__("SMA", "Simple Moving Average")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        return close.rolling(window=period).mean()


class EMA(BaseIndicator):
    """Exponential Moving Average"""
    
    def __init__(self):
        super().__init__("EMA", "Exponential Moving Average")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        return close.ewm(span=period).mean()


class WMA(BaseIndicator):
    """Weighted Moving Average"""
    
    def __init__(self):
        super().__init__("WMA", "Weighted Moving Average")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        weights = np.arange(1, period + 1)
        
        def weighted_mean(x):
            return np.average(x, weights=weights)
        
        return close.rolling(window=period).apply(weighted_mean, raw=True)


class TEMA(BaseIndicator):
    """Triple Exponential Moving Average"""
    
    def __init__(self):
        super().__init__("TEMA", "Triple Exponential Moving Average")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        
        tema = 3 * ema1 - 3 * ema2 + ema3
        return tema


class DEMA(BaseIndicator):
    """Double Exponential Moving Average"""
    
    def __init__(self):
        super().__init__("DEMA", "Double Exponential Moving Average")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        
        dema = 2 * ema1 - ema2
        return dema


class KAMA(BaseIndicator):
    """Kaufman's Adaptive Moving Average"""
    
    def __init__(self):
        super().__init__("KAMA", "Kaufman's Adaptive Moving Average")
        self.set_parameter('period', 10)
        self.set_parameter('fast_sc', 2)
        self.set_parameter('slow_sc', 30)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 10))
        fast_sc = kwargs.get('fast_sc', self._parameters.get('fast_sc', 2))
        slow_sc = kwargs.get('slow_sc', self._parameters.get('slow_sc', 30))
        
        close = data['close']
        
        # Direction and volatility
        direction = np.abs(close - close.shift(period))
        volatility = np.abs(close - close.shift(1)).rolling(window=period).sum()
        
        # Efficiency ratio
        er = direction / volatility
        
        # Smoothing constants
        fastest_sc = 2.0 / (fast_sc + 1)
        slowest_sc = 2.0 / (slow_sc + 1)
        sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
        
        # KAMA calculation
        kama = pd.Series(index=close.index, dtype=float)
        kama.iloc[period-1] = close.iloc[period-1]
        
        for i in range(period, len(close)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        
        return kama


class T3(BaseIndicator):
    """T3 Moving Average"""
    
    def __init__(self):
        super().__init__("T3", "T3 Moving Average")
        self.set_parameter('period', 20)
        self.set_parameter('volume_factor', 0.7)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        vf = kwargs.get('volume_factor', self._parameters.get('volume_factor', 0.7))
        
        close = data['close']
        
        # T3 coefficients
        c1 = -vf**3
        c2 = 3*vf**2 + 3*vf**3
        c3 = -6*vf**2 - 3*vf - 3*vf**3
        c4 = 1 + 3*vf + vf**3 + 3*vf**2
        
        # Multiple EMAs
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        ema4 = ema3.ewm(span=period).mean()
        ema5 = ema4.ewm(span=period).mean()
        ema6 = ema5.ewm(span=period).mean()
        
        t3 = c1*ema6 + c2*ema5 + c3*ema4 + c4*ema3
        return t3


class LinearRegression(BaseIndicator):
    """Linear Regression"""
    
    def __init__(self):
        super().__init__("LinearRegression", "Linear Regression trend line")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        
        def linear_reg_manual(y):
            """Manual linear regression calculation"""
            if len(y) < 2:
                return pd.Series([np.nan, np.nan, np.nan, np.nan])
            
            x = np.arange(len(y))
            n = len(y)
            
            # Calculate slope and intercept
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Calculate standard error
            std_err = np.sqrt(ss_res / (n - 2)) if n > 2 else np.nan
            
            return pd.Series([slope, intercept, r_squared, std_err])
        
        if HAS_SCIPY:
            def linear_reg(y):
                x = np.arange(len(y))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                return pd.Series([slope, intercept, r_value**2, std_err])
            
            lr_results = close.rolling(window=period).apply(
                lambda x: linear_reg(x).iloc[0], raw=False
            )
            
            # Calculate regression line values
            reg_line = pd.Series(index=close.index, dtype=float)
            for i in range(period-1, len(close)):
                window_data = close.iloc[i-period+1:i+1]
                x = np.arange(len(window_data))
                slope, intercept, _, _, _ = stats.linregress(x, window_data)
                reg_line.iloc[i] = slope * (period - 1) + intercept
        else:
            lr_results = close.rolling(window=period).apply(
                lambda x: linear_reg_manual(x).iloc[0], raw=False
            )
            
            # Calculate regression line values
            reg_line = pd.Series(index=close.index, dtype=float)
            for i in range(period-1, len(close)):
                window_data = close.iloc[i-period+1:i+1]
                lr_result = linear_reg_manual(window_data)
                slope, intercept = lr_result.iloc[0], lr_result.iloc[1]
                reg_line.iloc[i] = slope * (period - 1) + intercept
        
        return pd.DataFrame({
            'LinearRegression': reg_line,
            'Slope': lr_results
        })


class Correlation(BaseIndicator):
    """Rolling Correlation"""
    
    def __init__(self):
        super().__init__("Correlation", "Rolling correlation between two series")
        self.set_parameter('period', 20)
        self.set_parameter('reference_column', 'volume')
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        ref_col = kwargs.get('reference_column', self._parameters.get('reference_column', 'volume'))
        
        if ref_col not in data.columns:
            raise ValueError(f"Reference column '{ref_col}' not found in data")
        
        close = data['close']
        reference = data[ref_col]
        
        return close.rolling(window=period).corr(reference)


class Covariance(BaseIndicator):
    """Rolling Covariance"""
    
    def __init__(self):
        super().__init__("Covariance", "Rolling covariance between two series")
        self.set_parameter('period', 20)
        self.set_parameter('reference_column', 'volume')
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        ref_col = kwargs.get('reference_column', self._parameters.get('reference_column', 'volume'))
        
        if ref_col not in data.columns:
            raise ValueError(f"Reference column '{ref_col}' not found in data")
        
        close = data['close']
        reference = data[ref_col]
        
        return close.rolling(window=period).cov(reference)


class Beta(BaseIndicator):
    """Rolling Beta"""
    
    def __init__(self):
        super().__init__("Beta", "Rolling beta coefficient")
        self.set_parameter('period', 60)
        self.set_parameter('market_column', 'market_return')
    
    def get_required_columns(self) -> List[str]:
        return ['close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 60))
        market_col = kwargs.get('market_column', self._parameters.get('market_column', 'market_return'))
        
        if market_col not in data.columns:
            raise ValueError(f"Market column '{market_col}' not found in data")
        
        # Calculate returns
        asset_returns = data['close'].pct_change()
        market_returns = data[market_col]
        
        # Rolling beta calculation
        covariance = asset_returns.rolling(window=period).cov(market_returns)
        market_variance = market_returns.rolling(window=period).var()
        
        beta = covariance / market_variance
        return beta


class RSquared(BaseIndicator):
    """Rolling R-Squared"""
    
    def __init__(self):
        super().__init__("RSquared", "Rolling R-squared coefficient")
        self.set_parameter('period', 60)
        self.set_parameter('market_column', 'market_return')
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 60))
        market_col = kwargs.get('market_column', self._parameters.get('market_column', 'market_return'))
        
        if market_col not in data.columns:
            raise ValueError(f"Market column '{market_col}' not found in data")
        
        # Calculate returns
        asset_returns = data['close'].pct_change()
        market_returns = data[market_col]
        
        # Rolling R-squared
        correlation = asset_returns.rolling(window=period).corr(market_returns)
        r_squared = correlation ** 2
        
        return r_squared


class ZScore(BaseIndicator):
    """Z-Score (Standard Score)"""
    
    def __init__(self):
        super().__init__("ZScore", "Z-Score standardization")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        rolling_mean = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()
        
        z_score = (close - rolling_mean) / rolling_std
        return z_score


class PercentRank(BaseIndicator):
    """Percent Rank"""
    
    def __init__(self):
        super().__init__("PercentRank", "Percentile rank within rolling window")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        
        def percentile_rank_manual(x):
            """Manual percentile rank calculation"""
            if len(x) < 2:
                return np.nan
            current_value = x[-1]
            historical_values = x[:-1]
            rank = np.sum(historical_values < current_value)
            return rank / len(historical_values)
        
        if HAS_SCIPY:
            def percentile_rank(x):
                return stats.percentileofscore(x[:-1], x[-1]) / 100.0
            percent_rank = close.rolling(window=period).apply(percentile_rank, raw=True)
        else:
            percent_rank = close.rolling(window=period).apply(percentile_rank_manual, raw=True)
        
        return percent_rank


class Skewness(BaseIndicator):
    """Rolling Skewness"""
    
    def __init__(self):
        super().__init__("Skewness", "Rolling skewness of returns")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        returns = data['close'].pct_change()
        skewness = returns.rolling(window=period).skew()
        
        return skewness


class Kurtosis(BaseIndicator):
    """Rolling Kurtosis"""
    
    def __init__(self):
        super().__init__("Kurtosis", "Rolling kurtosis of returns")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        returns = data['close'].pct_change()
        kurtosis = returns.rolling(window=period).kurt()
        
        return kurtosis


class Entropy(BaseIndicator):
    """Rolling Shannon Entropy"""
    
    def __init__(self):
        super().__init__("Entropy", "Rolling Shannon entropy")
        self.set_parameter('period', 20)
        self.set_parameter('bins', 10)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        bins = kwargs.get('bins', self._parameters.get('bins', 10))
        
        returns = data['close'].pct_change()
        
        def shannon_entropy(x):
            if len(x) < 2:
                return np.nan
            hist, _ = np.histogram(x, bins=bins)
            hist = hist[hist > 0]  # Remove zero bins
            if len(hist) == 0:
                return 0
            prob = hist / hist.sum()
            return -np.sum(prob * np.log2(prob))
        
        entropy = returns.rolling(window=period).apply(shannon_entropy, raw=True)
        return entropy


class HurstExponent(BaseIndicator):
    """Hurst Exponent"""
    
    def __init__(self):
        super().__init__("HurstExponent", "Hurst exponent for trend analysis")
        self.set_parameter('period', 100)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 100))
        
        close = data['close']
        
        def hurst_exponent(ts):
            if len(ts) < 10:
                return np.nan
            
            # Calculate the range of the cumulative sum of deviations
            ts = np.array(ts)
            n = len(ts)
            
            # Mean-centered cumulative sum
            mean_ts = ts - np.mean(ts)
            cumsum_ts = np.cumsum(mean_ts)
            
            # Range
            R = np.max(cumsum_ts) - np.min(cumsum_ts)
            
            # Standard deviation
            S = np.std(ts)
            
            if S == 0:
                return np.nan
            
            # Hurst exponent approximation
            return np.log(R/S) / np.log(n)
        
        hurst = close.rolling(window=period).apply(hurst_exponent, raw=True)
        return hurst