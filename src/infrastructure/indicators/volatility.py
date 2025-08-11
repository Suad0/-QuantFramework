"""
Volatility Technical Indicators

This module contains volatility-based technical indicators that measure
price volatility and market uncertainty.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.application.services.feature_engine import BaseIndicator


class BollingerBands(BaseIndicator):
    """Bollinger Bands"""
    
    def __init__(self):
        super().__init__("BollingerBands", "Bollinger Bands volatility indicator")
        self.set_parameter('period', 20)
        self.set_parameter('std_dev', 2)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        std_dev = kwargs.get('std_dev', self._parameters.get('std_dev', 2))
        
        close = data['close']
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'Upper': upper_band,
            'Middle': sma,
            'Lower': lower_band,
            'Width': (upper_band - lower_band) / sma,
            'Position': (close - lower_band) / (upper_band - lower_band)
        })


class ATR(BaseIndicator):
    """Average True Range"""
    
    def __init__(self):
        super().__init__("ATR", "Average True Range volatility measure")
        self.set_parameter('period', 14)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 14))
        
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.rolling(window=period).mean()
        
        return atr


class Keltner(BaseIndicator):
    """Keltner Channels"""
    
    def __init__(self):
        super().__init__("Keltner", "Keltner Channels volatility bands")
        self.set_parameter('period', 20)
        self.set_parameter('multiplier', 2)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        multiplier = kwargs.get('multiplier', self._parameters.get('multiplier', 2))
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Typical price and ATR
        typical_price = (high + low + close) / 3
        ema = typical_price.ewm(span=period).mean()
        
        # Calculate ATR
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.ewm(span=period).mean()
        
        upper_band = ema + (multiplier * atr)
        lower_band = ema - (multiplier * atr)
        
        return pd.DataFrame({
            'Upper': upper_band,
            'Middle': ema,
            'Lower': lower_band
        })


class DonchianChannel(BaseIndicator):
    """Donchian Channels"""
    
    def __init__(self):
        super().__init__("DonchianChannel", "Donchian Channels breakout indicator")
        self.set_parameter('period', 20)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        high = data['high']
        low = data['low']
        
        upper_band = high.rolling(window=period).max()
        lower_band = low.rolling(window=period).min()
        middle_band = (upper_band + lower_band) / 2
        
        return pd.DataFrame({
            'Upper': upper_band,
            'Middle': middle_band,
            'Lower': lower_band
        })


class VWAP(BaseIndicator):
    """Volume Weighted Average Price"""
    
    def __init__(self):
        super().__init__("VWAP", "Volume Weighted Average Price")
        self.set_parameter('period', None)  # None means cumulative
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period'))
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        typical_price = (high + low + close) / 3
        pv = typical_price * volume
        
        if period is None:
            # Cumulative VWAP
            vwap = pv.cumsum() / volume.cumsum()
        else:
            # Rolling VWAP
            vwap = pv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return vwap


class StandardDeviation(BaseIndicator):
    """Standard Deviation"""
    
    def __init__(self):
        super().__init__("StandardDeviation", "Price standard deviation")
        self.set_parameter('period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        close = data['close']
        std_dev = close.rolling(window=period).std()
        
        return std_dev


class HistoricalVolatility(BaseIndicator):
    """Historical Volatility"""
    
    def __init__(self):
        super().__init__("HistoricalVolatility", "Annualized historical volatility")
        self.set_parameter('period', 30)
        self.set_parameter('trading_days', 252)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 30))
        trading_days = kwargs.get('trading_days', self._parameters.get('trading_days', 252))
        
        close = data['close']
        returns = np.log(close / close.shift(1))
        volatility = returns.rolling(window=period).std() * np.sqrt(trading_days)
        
        return volatility


class GarmanKlass(BaseIndicator):
    """Garman-Klass Volatility Estimator"""
    
    def __init__(self):
        super().__init__("GarmanKlass", "Garman-Klass volatility estimator")
        self.set_parameter('period', 30)
        self.set_parameter('trading_days', 252)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'open', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 30))
        trading_days = kwargs.get('trading_days', self._parameters.get('trading_days', 252))
        
        high = data['high']
        low = data['low']
        open_price = data['open']
        close = data['close']
        
        # Garman-Klass estimator
        gk = (np.log(high/close) * np.log(high/open_price) + 
              np.log(low/close) * np.log(low/open_price))
        
        volatility = np.sqrt(gk.rolling(window=period).mean() * trading_days)
        
        return volatility


class Parkinson(BaseIndicator):
    """Parkinson Volatility Estimator"""
    
    def __init__(self):
        super().__init__("Parkinson", "Parkinson volatility estimator")
        self.set_parameter('period', 30)
        self.set_parameter('trading_days', 252)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 30))
        trading_days = kwargs.get('trading_days', self._parameters.get('trading_days', 252))
        
        high = data['high']
        low = data['low']
        
        # Parkinson estimator
        pk = (1 / (4 * np.log(2))) * (np.log(high/low))**2
        volatility = np.sqrt(pk.rolling(window=period).mean() * trading_days)
        
        return volatility


class RogersSatchell(BaseIndicator):
    """Rogers-Satchell Volatility Estimator"""
    
    def __init__(self):
        super().__init__("RogersSatchell", "Rogers-Satchell volatility estimator")
        self.set_parameter('period', 30)
        self.set_parameter('trading_days', 252)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'open', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 30))
        trading_days = kwargs.get('trading_days', self._parameters.get('trading_days', 252))
        
        high = data['high']
        low = data['low']
        open_price = data['open']
        close = data['close']
        
        # Rogers-Satchell estimator
        rs = (np.log(high/close) * np.log(high/open_price) + 
              np.log(low/close) * np.log(low/open_price))
        
        volatility = np.sqrt(rs.rolling(window=period).mean() * trading_days)
        
        return volatility