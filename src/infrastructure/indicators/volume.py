"""
Volume Technical Indicators

This module contains volume-based technical indicators that analyze
trading volume patterns and price-volume relationships.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.application.services.feature_engine import BaseIndicator


class OBV(BaseIndicator):
    """On-Balance Volume"""
    
    def __init__(self):
        super().__init__("OBV", "On-Balance Volume accumulation indicator")
    
    def get_required_columns(self) -> List[str]:
        return ['close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        
        close = data['close']
        volume = data['volume']
        
        price_change = close.diff()
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0)).cumsum()
        
        return pd.Series(obv, index=data.index)


class AD(BaseIndicator):
    """Accumulation/Distribution Line"""
    
    def __init__(self):
        super().__init__("AD", "Accumulation/Distribution Line")
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Handle division by zero
        ad = (clv * volume).cumsum()
        
        return ad


class ADOSC(BaseIndicator):
    """Accumulation/Distribution Oscillator"""
    
    def __init__(self):
        super().__init__("ADOSC", "Accumulation/Distribution Oscillator")
        self.set_parameter('fast_period', 3)
        self.set_parameter('slow_period', 10)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        fast_period = kwargs.get('fast_period', self._parameters.get('fast_period', 3))
        slow_period = kwargs.get('slow_period', self._parameters.get('slow_period', 10))
        
        # First calculate AD line
        ad_indicator = AD()
        ad_line = ad_indicator.calculate(data)
        
        # Calculate oscillator
        fast_ema = ad_line.ewm(span=fast_period).mean()
        slow_ema = ad_line.ewm(span=slow_period).mean()
        adosc = fast_ema - slow_ema
        
        return adosc


class PVT(BaseIndicator):
    """Price Volume Trend"""
    
    def __init__(self):
        super().__init__("PVT", "Price Volume Trend indicator")
    
    def get_required_columns(self) -> List[str]:
        return ['close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        
        close = data['close']
        volume = data['volume']
        
        price_change_pct = close.pct_change()
        pvt = (price_change_pct * volume).cumsum()
        
        return pvt


class NVI(BaseIndicator):
    """Negative Volume Index"""
    
    def __init__(self):
        super().__init__("NVI", "Negative Volume Index")
        self.set_parameter('initial_value', 1000)
    
    def get_required_columns(self) -> List[str]:
        return ['close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        initial_value = kwargs.get('initial_value', self._parameters.get('initial_value', 1000))
        
        close = data['close']
        volume = data['volume']
        
        nvi = pd.Series(index=data.index, dtype=float)
        nvi.iloc[0] = initial_value
        
        for i in range(1, len(data)):
            if volume.iloc[i] < volume.iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] * (close.iloc[i] / close.iloc[i-1])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        return nvi


class PVI(BaseIndicator):
    """Positive Volume Index"""
    
    def __init__(self):
        super().__init__("PVI", "Positive Volume Index")
        self.set_parameter('initial_value', 1000)
    
    def get_required_columns(self) -> List[str]:
        return ['close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        initial_value = kwargs.get('initial_value', self._parameters.get('initial_value', 1000))
        
        close = data['close']
        volume = data['volume']
        
        pvi = pd.Series(index=data.index, dtype=float)
        pvi.iloc[0] = initial_value
        
        for i in range(1, len(data)):
            if volume.iloc[i] > volume.iloc[i-1]:
                pvi.iloc[i] = pvi.iloc[i-1] * (close.iloc[i] / close.iloc[i-1])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        
        return pvi


class EMV(BaseIndicator):
    """Ease of Movement"""
    
    def __init__(self):
        super().__init__("EMV", "Ease of Movement indicator")
        self.set_parameter('period', 14)
        self.set_parameter('scale', 10000)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 14))
        scale = kwargs.get('scale', self._parameters.get('scale', 10000))
        
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Distance moved
        distance = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        
        # Box height (high - low)
        box_height = high - low
        
        # Scale factor
        scale_factor = volume / (scale * box_height)
        scale_factor = scale_factor.replace([np.inf, -np.inf], 0)
        
        # Raw EMV
        raw_emv = distance / scale_factor
        raw_emv = raw_emv.replace([np.inf, -np.inf], 0).fillna(0)
        
        # Smoothed EMV
        emv = raw_emv.rolling(window=period).mean()
        
        return emv


class FI(BaseIndicator):
    """Force Index"""
    
    def __init__(self):
        super().__init__("FI", "Force Index")
        self.set_parameter('period', 13)
    
    def get_required_columns(self) -> List[str]:
        return ['close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 13))
        
        close = data['close']
        volume = data['volume']
        
        # Raw Force Index
        raw_fi = (close - close.shift(1)) * volume
        
        # Smoothed Force Index
        if period == 1:
            return raw_fi
        else:
            return raw_fi.ewm(span=period).mean()


class VWMACD(BaseIndicator):
    """Volume Weighted MACD"""
    
    def __init__(self):
        super().__init__("VWMACD", "Volume Weighted MACD")
        self.set_parameter('fast_period', 12)
        self.set_parameter('slow_period', 26)
        self.set_parameter('signal_period', 9)
    
    def get_required_columns(self) -> List[str]:
        return ['close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        fast_period = kwargs.get('fast_period', self._parameters.get('fast_period', 12))
        slow_period = kwargs.get('slow_period', self._parameters.get('slow_period', 26))
        signal_period = kwargs.get('signal_period', self._parameters.get('signal_period', 9))
        
        close = data['close']
        volume = data['volume']
        
        # Volume weighted price
        vwp = close * volume
        
        # Volume weighted EMAs
        vw_fast = vwp.ewm(span=fast_period).mean() / volume.ewm(span=fast_period).mean()
        vw_slow = vwp.ewm(span=slow_period).mean() / volume.ewm(span=slow_period).mean()
        
        # VWMACD line
        vwmacd_line = vw_fast - vw_slow
        signal_line = vwmacd_line.ewm(span=signal_period).mean()
        histogram = vwmacd_line - signal_line
        
        return pd.DataFrame({
            'VWMACD': vwmacd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })


class VolumePriceConfirmation(BaseIndicator):
    """Volume Price Confirmation Index"""
    
    def __init__(self):
        super().__init__("VolumePriceConfirmation", "Volume Price Confirmation Index")
        self.set_parameter('period', 20)
    
    def get_required_columns(self) -> List[str]:
        return ['close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        volume = data['volume']
        
        # Price change and volume change
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        
        # Correlation between price and volume changes
        vpc = price_change.rolling(window=period).corr(volume_change)
        
        return vpc


class VolumeWeightedRSI(BaseIndicator):
    """Volume Weighted RSI"""
    
    def __init__(self):
        super().__init__("VolumeWeightedRSI", "Volume Weighted RSI")
        self.set_parameter('period', 14)
    
    def get_required_columns(self) -> List[str]:
        return ['close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 14))
        
        close = data['close']
        volume = data['volume']
        
        # Volume weighted price changes
        price_change = close.diff()
        vw_gain = (price_change * volume).where(price_change > 0, 0)
        vw_loss = (-price_change * volume).where(price_change < 0, 0)
        
        # Volume weighted averages
        avg_vw_gain = vw_gain.rolling(window=period).sum() / volume.rolling(window=period).sum()
        avg_vw_loss = vw_loss.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        # Volume weighted RS and RSI
        vw_rs = avg_vw_gain / avg_vw_loss
        vw_rsi = 100 - (100 / (1 + vw_rs))
        
        return vw_rsi