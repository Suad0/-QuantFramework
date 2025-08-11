"""
Momentum Technical Indicators

This module contains momentum-based technical indicators that measure
the rate of change in price movements.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.application.services.feature_engine import BaseIndicator


class RSI(BaseIndicator):
    """Relative Strength Index"""
    
    def __init__(self):
        super().__init__("RSI", "Relative Strength Index momentum oscillator")
        self.set_parameter('period', 14)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 14))
        
        close = data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence"""
    
    def __init__(self):
        super().__init__("MACD", "Moving Average Convergence Divergence")
        self.set_parameter('fast_period', 12)
        self.set_parameter('slow_period', 26)
        self.set_parameter('signal_period', 9)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        fast_period = kwargs.get('fast_period', self._parameters.get('fast_period', 12))
        slow_period = kwargs.get('slow_period', self._parameters.get('slow_period', 26))
        signal_period = kwargs.get('signal_period', self._parameters.get('signal_period', 9))
        
        close = data['close']
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })


class Stochastic(BaseIndicator):
    """Stochastic Oscillator"""
    
    def __init__(self):
        super().__init__("Stochastic", "Stochastic momentum oscillator")
        self.set_parameter('k_period', 14)
        self.set_parameter('d_period', 3)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        k_period = kwargs.get('k_period', self._parameters.get('k_period', 14))
        d_period = kwargs.get('d_period', self._parameters.get('d_period', 3))
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'K': k_percent,
            'D': d_percent
        })


class Williams_R(BaseIndicator):
    """Williams %R"""
    
    def __init__(self):
        super().__init__("Williams_R", "Williams %R momentum oscillator")
        self.set_parameter('period', 14)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 14))
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        
        return williams_r


class ROC(BaseIndicator):
    """Rate of Change"""
    
    def __init__(self):
        super().__init__("ROC", "Rate of Change momentum indicator")
        self.set_parameter('period', 12)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 12))
        
        close = data['close']
        roc = ((close - close.shift(period)) / close.shift(period)) * 100
        
        return roc


class MFI(BaseIndicator):
    """Money Flow Index"""
    
    def __init__(self):
        super().__init__("MFI", "Money Flow Index volume-weighted momentum")
        self.set_parameter('period', 14)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 14))
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return mfi


class TSI(BaseIndicator):
    """True Strength Index"""
    
    def __init__(self):
        super().__init__("TSI", "True Strength Index double-smoothed momentum")
        self.set_parameter('first_smooth', 25)
        self.set_parameter('second_smooth', 13)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        first_smooth = kwargs.get('first_smooth', self._parameters.get('first_smooth', 25))
        second_smooth = kwargs.get('second_smooth', self._parameters.get('second_smooth', 13))
        
        close = data['close']
        momentum = close.diff()
        abs_momentum = momentum.abs()
        
        # Double smoothing
        momentum_smooth1 = momentum.ewm(span=first_smooth).mean()
        momentum_smooth2 = momentum_smooth1.ewm(span=second_smooth).mean()
        
        abs_momentum_smooth1 = abs_momentum.ewm(span=first_smooth).mean()
        abs_momentum_smooth2 = abs_momentum_smooth1.ewm(span=second_smooth).mean()
        
        tsi = 100 * (momentum_smooth2 / abs_momentum_smooth2)
        
        return tsi


class UO(BaseIndicator):
    """Ultimate Oscillator"""
    
    def __init__(self):
        super().__init__("UO", "Ultimate Oscillator multi-timeframe momentum")
        self.set_parameter('period1', 7)
        self.set_parameter('period2', 14)
        self.set_parameter('period3', 28)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period1 = kwargs.get('period1', self._parameters.get('period1', 7))
        period2 = kwargs.get('period2', self._parameters.get('period2', 14))
        period3 = kwargs.get('period3', self._parameters.get('period3', 28))
        
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)
        
        true_low = np.minimum(low, prev_close)
        buying_pressure = close - true_low
        true_range = np.maximum(high, prev_close) - true_low
        
        bp1 = buying_pressure.rolling(window=period1).sum()
        tr1 = true_range.rolling(window=period1).sum()
        
        bp2 = buying_pressure.rolling(window=period2).sum()
        tr2 = true_range.rolling(window=period2).sum()
        
        bp3 = buying_pressure.rolling(window=period3).sum()
        tr3 = true_range.rolling(window=period3).sum()
        
        uo = 100 * ((4 * bp1/tr1) + (2 * bp2/tr2) + (bp3/tr3)) / 7
        
        return uo


class CCI(BaseIndicator):
    """Commodity Channel Index"""
    
    def __init__(self):
        super().__init__("CCI", "Commodity Channel Index momentum oscillator")
        self.set_parameter('period', 20)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci


class DPO(BaseIndicator):
    """Detrended Price Oscillator"""
    
    def __init__(self):
        super().__init__("DPO", "Detrended Price Oscillator")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        sma = close.rolling(window=period).mean()
        shift_period = period // 2 + 1
        
        dpo = close - sma.shift(shift_period)
        
        return dpo


class CMO(BaseIndicator):
    """Chande Momentum Oscillator"""
    
    def __init__(self):
        super().__init__("CMO", "Chande Momentum Oscillator")
        self.set_parameter('period', 14)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 14))
        
        close = data['close']
        momentum = close.diff()
        
        positive_sum = momentum.where(momentum > 0, 0).rolling(window=period).sum()
        negative_sum = momentum.where(momentum < 0, 0).abs().rolling(window=period).sum()
        
        cmo = 100 * (positive_sum - negative_sum) / (positive_sum + negative_sum)
        
        return cmo


class TRIX(BaseIndicator):
    """TRIX - Triple Exponential Average"""
    
    def __init__(self):
        super().__init__("TRIX", "TRIX triple exponential momentum")
        self.set_parameter('period', 14)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 14))
        
        close = data['close']
        
        # Triple exponential smoothing
        ema1 = close.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()
        ema3 = ema2.ewm(span=period).mean()
        
        # Rate of change of triple EMA
        trix = ema3.pct_change() * 10000  # Scale for readability
        
        return trix


class KST(BaseIndicator):
    """Know Sure Thing"""
    
    def __init__(self):
        super().__init__("KST", "Know Sure Thing momentum oscillator")
        self.set_parameter('roc1_period', 10)
        self.set_parameter('roc2_period', 15)
        self.set_parameter('roc3_period', 20)
        self.set_parameter('roc4_period', 30)
        self.set_parameter('sma1_period', 10)
        self.set_parameter('sma2_period', 10)
        self.set_parameter('sma3_period', 10)
        self.set_parameter('sma4_period', 15)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        
        # Get parameters
        roc1_period = kwargs.get('roc1_period', self._parameters.get('roc1_period', 10))
        roc2_period = kwargs.get('roc2_period', self._parameters.get('roc2_period', 15))
        roc3_period = kwargs.get('roc3_period', self._parameters.get('roc3_period', 20))
        roc4_period = kwargs.get('roc4_period', self._parameters.get('roc4_period', 30))
        
        sma1_period = kwargs.get('sma1_period', self._parameters.get('sma1_period', 10))
        sma2_period = kwargs.get('sma2_period', self._parameters.get('sma2_period', 10))
        sma3_period = kwargs.get('sma3_period', self._parameters.get('sma3_period', 10))
        sma4_period = kwargs.get('sma4_period', self._parameters.get('sma4_period', 15))
        
        close = data['close']
        
        # Calculate ROCs
        roc1 = ((close - close.shift(roc1_period)) / close.shift(roc1_period)) * 100
        roc2 = ((close - close.shift(roc2_period)) / close.shift(roc2_period)) * 100
        roc3 = ((close - close.shift(roc3_period)) / close.shift(roc3_period)) * 100
        roc4 = ((close - close.shift(roc4_period)) / close.shift(roc4_period)) * 100
        
        # Smooth ROCs
        roc1_sma = roc1.rolling(window=sma1_period).mean()
        roc2_sma = roc2.rolling(window=sma2_period).mean()
        roc3_sma = roc3.rolling(window=sma3_period).mean()
        roc4_sma = roc4.rolling(window=sma4_period).mean()
        
        # Calculate KST
        kst = (roc1_sma * 1) + (roc2_sma * 2) + (roc3_sma * 3) + (roc4_sma * 4)
        
        return kst