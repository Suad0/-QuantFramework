"""
Oscillator Technical Indicators

This module contains oscillator indicators that fluctuate between
fixed upper and lower bounds to identify overbought/oversold conditions.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.application.services.feature_engine import BaseIndicator


class StochasticRSI(BaseIndicator):
    """Stochastic RSI"""
    
    def __init__(self):
        super().__init__("StochasticRSI", "Stochastic RSI oscillator")
        self.set_parameter('rsi_period', 14)
        self.set_parameter('stoch_period', 14)
        self.set_parameter('k_period', 3)
        self.set_parameter('d_period', 3)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        rsi_period = kwargs.get('rsi_period', self._parameters.get('rsi_period', 14))
        stoch_period = kwargs.get('stoch_period', self._parameters.get('stoch_period', 14))
        k_period = kwargs.get('k_period', self._parameters.get('k_period', 3))
        d_period = kwargs.get('d_period', self._parameters.get('d_period', 3))
        
        close = data['close']
        
        # Calculate RSI first
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic of RSI
        rsi_low = rsi.rolling(window=stoch_period).min()
        rsi_high = rsi.rolling(window=stoch_period).max()
        
        stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
        
        # Smooth with moving averages
        k_percent = stoch_rsi.rolling(window=k_period).mean()
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'StochRSI': stoch_rsi,
            'K': k_percent,
            'D': d_percent
        })


class MACD_Histogram(BaseIndicator):
    """MACD Histogram"""
    
    def __init__(self):
        super().__init__("MACD_Histogram", "MACD Histogram oscillator")
        self.set_parameter('fast_period', 12)
        self.set_parameter('slow_period', 26)
        self.set_parameter('signal_period', 9)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        fast_period = kwargs.get('fast_period', self._parameters.get('fast_period', 12))
        slow_period = kwargs.get('slow_period', self._parameters.get('slow_period', 26))
        signal_period = kwargs.get('signal_period', self._parameters.get('signal_period', 9))
        
        close = data['close']
        
        # MACD calculation
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return histogram


class PPO(BaseIndicator):
    """Percentage Price Oscillator"""
    
    def __init__(self):
        super().__init__("PPO", "Percentage Price Oscillator")
        self.set_parameter('fast_period', 12)
        self.set_parameter('slow_period', 26)
        self.set_parameter('signal_period', 9)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        fast_period = kwargs.get('fast_period', self._parameters.get('fast_period', 12))
        slow_period = kwargs.get('slow_period', self._parameters.get('slow_period', 26))
        signal_period = kwargs.get('signal_period', self._parameters.get('signal_period', 9))
        
        close = data['close']
        
        # PPO calculation
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        
        ppo_line = ((ema_fast - ema_slow) / ema_slow) * 100
        signal_line = ppo_line.ewm(span=signal_period).mean()
        histogram = ppo_line - signal_line
        
        return pd.DataFrame({
            'PPO': ppo_line,
            'Signal': signal_line,
            'Histogram': histogram
        })


class PriceOscillator(BaseIndicator):
    """Price Oscillator"""
    
    def __init__(self):
        super().__init__("PriceOscillator", "Price Oscillator")
        self.set_parameter('fast_period', 10)
        self.set_parameter('slow_period', 20)
        self.set_parameter('ma_type', 'SMA')  # SMA or EMA
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        fast_period = kwargs.get('fast_period', self._parameters.get('fast_period', 10))
        slow_period = kwargs.get('slow_period', self._parameters.get('slow_period', 20))
        ma_type = kwargs.get('ma_type', self._parameters.get('ma_type', 'SMA'))
        
        close = data['close']
        
        if ma_type.upper() == 'SMA':
            fast_ma = close.rolling(window=fast_period).mean()
            slow_ma = close.rolling(window=slow_period).mean()
        else:  # EMA
            fast_ma = close.ewm(span=fast_period).mean()
            slow_ma = close.ewm(span=slow_period).mean()
        
        price_oscillator = fast_ma - slow_ma
        
        return price_oscillator


class VolumeOscillator(BaseIndicator):
    """Volume Oscillator"""
    
    def __init__(self):
        super().__init__("VolumeOscillator", "Volume Oscillator")
        self.set_parameter('fast_period', 5)
        self.set_parameter('slow_period', 10)
    
    def get_required_columns(self) -> List[str]:
        return ['volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        fast_period = kwargs.get('fast_period', self._parameters.get('fast_period', 5))
        slow_period = kwargs.get('slow_period', self._parameters.get('slow_period', 10))
        
        volume = data['volume']
        
        fast_ma = volume.rolling(window=fast_period).mean()
        slow_ma = volume.rolling(window=slow_period).mean()
        
        volume_oscillator = ((fast_ma - slow_ma) / slow_ma) * 100
        
        return volume_oscillator


class DerivativeOscillator(BaseIndicator):
    """Derivative Oscillator"""
    
    def __init__(self):
        super().__init__("DerivativeOscillator", "Derivative Oscillator")
        self.set_parameter('rsi_period', 14)
        self.set_parameter('ema1_period', 5)
        self.set_parameter('ema2_period', 3)
        self.set_parameter('sma_period', 9)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        rsi_period = kwargs.get('rsi_period', self._parameters.get('rsi_period', 14))
        ema1_period = kwargs.get('ema1_period', self._parameters.get('ema1_period', 5))
        ema2_period = kwargs.get('ema2_period', self._parameters.get('ema2_period', 3))
        sma_period = kwargs.get('sma_period', self._parameters.get('sma_period', 9))
        
        close = data['close']
        
        # Calculate RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Double smooth RSI
        ema1 = rsi.ewm(span=ema1_period).mean()
        ema2 = ema1.ewm(span=ema2_period).mean()
        
        # Signal line
        signal = ema2.rolling(window=sma_period).mean()
        
        # Derivative oscillator
        derivative_osc = ema2 - signal
        
        return derivative_osc


class RVGI(BaseIndicator):
    """Relative Vigor Index"""
    
    def __init__(self):
        super().__init__("RVGI", "Relative Vigor Index")
        self.set_parameter('period', 10)
    
    def get_required_columns(self) -> List[str]:
        return ['open', 'high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 10))
        
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Numerator: Close - Open
        numerator = close - open_price
        
        # Denominator: High - Low
        denominator = high - low
        
        # Apply 4-period weighted moving average
        def weighted_ma_4(series):
            return (series + 2*series.shift(1) + 2*series.shift(2) + series.shift(3)) / 6
        
        num_wma = weighted_ma_4(numerator)
        den_wma = weighted_ma_4(denominator)
        
        # RVGI calculation
        rvgi = num_wma.rolling(window=period).sum() / den_wma.rolling(window=period).sum()
        
        # Signal line (4-period weighted MA of RVGI)
        signal = weighted_ma_4(rvgi)
        
        return pd.DataFrame({
            'RVGI': rvgi,
            'Signal': signal
        })


class KeltnerOscillator(BaseIndicator):
    """Keltner Oscillator"""
    
    def __init__(self):
        super().__init__("KeltnerOscillator", "Keltner Oscillator")
        self.set_parameter('period', 20)
        self.set_parameter('multiplier', 2)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        multiplier = kwargs.get('multiplier', self._parameters.get('multiplier', 2))
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Typical price and EMA
        typical_price = (high + low + close) / 3
        ema = typical_price.ewm(span=period).mean()
        
        # ATR calculation
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.ewm(span=period).mean()
        
        # Keltner bands
        upper_band = ema + (multiplier * atr)
        lower_band = ema - (multiplier * atr)
        
        # Oscillator (position within bands)
        keltner_osc = (close - ema) / (upper_band - lower_band) * 100
        
        return keltner_osc


class ChaikinOscillator(BaseIndicator):
    """Chaikin Oscillator"""
    
    def __init__(self):
        super().__init__("ChaikinOscillator", "Chaikin Oscillator")
        self.set_parameter('fast_period', 3)
        self.set_parameter('slow_period', 10)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        fast_period = kwargs.get('fast_period', self._parameters.get('fast_period', 3))
        slow_period = kwargs.get('slow_period', self._parameters.get('slow_period', 10))
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Accumulation/Distribution Line
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad_line = (clv * volume).cumsum()
        
        # Chaikin Oscillator
        fast_ema = ad_line.ewm(span=fast_period).mean()
        slow_ema = ad_line.ewm(span=slow_period).mean()
        
        chaikin_osc = fast_ema - slow_ema
        
        return chaikin_osc


class AroonOscillator(BaseIndicator):
    """Aroon Oscillator"""
    
    def __init__(self):
        super().__init__("AroonOscillator", "Aroon Oscillator")
        self.set_parameter('period', 25)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 25))
        
        high = data['high']
        low = data['low']
        
        # Calculate Aroon Up and Down
        aroon_up = pd.Series(index=data.index, dtype=float)
        aroon_down = pd.Series(index=data.index, dtype=float)
        
        for i in range(period, len(data)):
            window_high = high.iloc[i-period+1:i+1]
            window_low = low.iloc[i-period+1:i+1]
            
            periods_since_high = period - 1 - window_high.argmax()
            periods_since_low = period - 1 - window_low.argmin()
            
            aroon_up.iloc[i] = ((period - periods_since_high) / period) * 100
            aroon_down.iloc[i] = ((period - periods_since_low) / period) * 100
        
        # Aroon Oscillator
        aroon_oscillator = aroon_up - aroon_down
        
        return aroon_oscillator


class PolarizedFractalEfficiency(BaseIndicator):
    """Polarized Fractal Efficiency"""
    
    def __init__(self):
        super().__init__("PolarizedFractalEfficiency", "Polarized Fractal Efficiency")
        self.set_parameter('period', 10)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 10))
        
        close = data['close']
        
        def pfe_calc(prices):
            if len(prices) < 2:
                return np.nan
            
            # Linear distance
            linear_distance = abs(prices[-1] - prices[0])
            
            # Euclidean distance (sum of price changes)
            euclidean_distance = np.sum(np.abs(np.diff(prices)))
            
            if euclidean_distance == 0:
                return 0
            
            # PFE calculation
            pfe = linear_distance / euclidean_distance
            
            # Add sign based on direction
            if prices[-1] > prices[0]:
                return pfe * 100
            else:
                return -pfe * 100
        
        pfe = close.rolling(window=period).apply(pfe_calc, raw=True)
        
        return pfe