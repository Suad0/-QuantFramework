"""
Trend Technical Indicators

This module contains trend-following indicators that identify
market direction and trend strength.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.application.services.feature_engine import BaseIndicator


class ADX(BaseIndicator):
    """Average Directional Index"""
    
    def __init__(self):
        super().__init__("ADX", "Average Directional Index trend strength")
        self.set_parameter('period', 14)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 14))
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        tr_smooth = tr.ewm(alpha=1/period).mean()
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period).mean()
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period).mean()
        
        return pd.DataFrame({
            'ADX': adx,
            'Plus_DI': plus_di,
            'Minus_DI': minus_di
        })


class Aroon(BaseIndicator):
    """Aroon Indicator"""
    
    def __init__(self):
        super().__init__("Aroon", "Aroon trend indicator")
        self.set_parameter('period', 25)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 25))
        
        high = data['high']
        low = data['low']
        
        # Periods since highest high and lowest low
        aroon_up = pd.Series(index=data.index, dtype=float)
        aroon_down = pd.Series(index=data.index, dtype=float)
        
        for i in range(period, len(data)):
            window_high = high.iloc[i-period+1:i+1]
            window_low = low.iloc[i-period+1:i+1]
            
            periods_since_high = period - 1 - window_high.argmax()
            periods_since_low = period - 1 - window_low.argmin()
            
            aroon_up.iloc[i] = ((period - periods_since_high) / period) * 100
            aroon_down.iloc[i] = ((period - periods_since_low) / period) * 100
        
        aroon_oscillator = aroon_up - aroon_down
        
        return pd.DataFrame({
            'Aroon_Up': aroon_up,
            'Aroon_Down': aroon_down,
            'Aroon_Oscillator': aroon_oscillator
        })


class PSAR(BaseIndicator):
    """Parabolic SAR"""
    
    def __init__(self):
        super().__init__("PSAR", "Parabolic Stop and Reverse")
        self.set_parameter('af_start', 0.02)
        self.set_parameter('af_increment', 0.02)
        self.set_parameter('af_maximum', 0.20)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        af_start = kwargs.get('af_start', self._parameters.get('af_start', 0.02))
        af_increment = kwargs.get('af_increment', self._parameters.get('af_increment', 0.02))
        af_maximum = kwargs.get('af_maximum', self._parameters.get('af_maximum', 0.20))
        
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        psar = np.zeros(len(data))
        trend = np.zeros(len(data))  # 1 for uptrend, -1 for downtrend
        af = np.zeros(len(data))
        ep = np.zeros(len(data))  # Extreme point
        
        # Initialize
        psar[0] = low[0]
        trend[0] = 1
        af[0] = af_start
        ep[0] = high[0]
        
        for i in range(1, len(data)):
            # Calculate PSAR
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            
            # Check for trend reversal
            if trend[i-1] == 1:  # Uptrend
                if low[i] <= psar[i]:
                    # Trend reversal to downtrend
                    trend[i] = -1
                    psar[i] = ep[i-1]  # Set PSAR to previous EP
                    af[i] = af_start
                    ep[i] = low[i]
                else:
                    # Continue uptrend
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_increment, af_maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # Ensure PSAR doesn't exceed previous two lows
                    psar[i] = min(psar[i], low[i-1])
                    if i > 1:
                        psar[i] = min(psar[i], low[i-2])
            
            else:  # Downtrend
                if high[i] >= psar[i]:
                    # Trend reversal to uptrend
                    trend[i] = 1
                    psar[i] = ep[i-1]  # Set PSAR to previous EP
                    af[i] = af_start
                    ep[i] = high[i]
                else:
                    # Continue downtrend
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_increment, af_maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # Ensure PSAR doesn't exceed previous two highs
                    psar[i] = max(psar[i], high[i-1])
                    if i > 1:
                        psar[i] = max(psar[i], high[i-2])
        
        return pd.Series(psar, index=data.index)


class SuperTrend(BaseIndicator):
    """SuperTrend Indicator"""
    
    def __init__(self):
        super().__init__("SuperTrend", "SuperTrend indicator")
        self.set_parameter('period', 10)
        self.set_parameter('multiplier', 3.0)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 10))
        multiplier = kwargs.get('multiplier', self._parameters.get('multiplier', 3.0))
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate ATR
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = tr.rolling(window=period).mean()
        
        # Basic bands
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Final bands
        final_upper = pd.Series(index=data.index, dtype=float)
        final_lower = pd.Series(index=data.index, dtype=float)
        supertrend = pd.Series(index=data.index, dtype=float)
        trend = pd.Series(index=data.index, dtype=int)
        
        for i in range(len(data)):
            if i == 0:
                final_upper.iloc[i] = upper_band.iloc[i]
                final_lower.iloc[i] = lower_band.iloc[i]
                supertrend.iloc[i] = upper_band.iloc[i]
                trend.iloc[i] = 1
            else:
                # Final upper band
                if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
                    final_upper.iloc[i] = upper_band.iloc[i]
                else:
                    final_upper.iloc[i] = final_upper.iloc[i-1]
                
                # Final lower band
                if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
                    final_lower.iloc[i] = lower_band.iloc[i]
                else:
                    final_lower.iloc[i] = final_lower.iloc[i-1]
                
                # SuperTrend
                if supertrend.iloc[i-1] == final_upper.iloc[i-1] and close.iloc[i] <= final_upper.iloc[i]:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    trend.iloc[i] = -1
                elif supertrend.iloc[i-1] == final_upper.iloc[i-1] and close.iloc[i] > final_upper.iloc[i]:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    trend.iloc[i] = 1
                elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and close.iloc[i] >= final_lower.iloc[i]:
                    supertrend.iloc[i] = final_lower.iloc[i]
                    trend.iloc[i] = 1
                elif supertrend.iloc[i-1] == final_lower.iloc[i-1] and close.iloc[i] < final_lower.iloc[i]:
                    supertrend.iloc[i] = final_upper.iloc[i]
                    trend.iloc[i] = -1
                else:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    trend.iloc[i] = trend.iloc[i-1]
        
        return pd.DataFrame({
            'SuperTrend': supertrend,
            'Trend': trend,
            'Upper_Band': final_upper,
            'Lower_Band': final_lower
        })


class Ichimoku(BaseIndicator):
    """Ichimoku Cloud"""
    
    def __init__(self):
        super().__init__("Ichimoku", "Ichimoku Kinko Hyo")
        self.set_parameter('tenkan_period', 9)
        self.set_parameter('kijun_period', 26)
        self.set_parameter('senkou_b_period', 52)
        self.set_parameter('displacement', 26)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        tenkan_period = kwargs.get('tenkan_period', self._parameters.get('tenkan_period', 9))
        kijun_period = kwargs.get('kijun_period', self._parameters.get('kijun_period', 26))
        senkou_b_period = kwargs.get('senkou_b_period', self._parameters.get('senkou_b_period', 52))
        displacement = kwargs.get('displacement', self._parameters.get('displacement', 26))
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        senkou_high = high.rolling(window=senkou_b_period).max()
        senkou_low = low.rolling(window=senkou_b_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-displacement)
        
        return pd.DataFrame({
            'Tenkan_Sen': tenkan_sen,
            'Kijun_Sen': kijun_sen,
            'Senkou_Span_A': senkou_span_a,
            'Senkou_Span_B': senkou_span_b,
            'Chikou_Span': chikou_span
        })


class VortexIndicator(BaseIndicator):
    """Vortex Indicator"""
    
    def __init__(self):
        super().__init__("VortexIndicator", "Vortex Indicator")
        self.set_parameter('period', 14)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low', 'close']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 14))
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Vortex Movements
        vm_plus = np.abs(high - low.shift(1))
        vm_minus = np.abs(low - high.shift(1))
        
        # Vortex Indicators
        vi_plus = vm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum()
        vi_minus = vm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()
        
        return pd.DataFrame({
            'VI_Plus': vi_plus,
            'VI_Minus': vi_minus,
            'VI_Diff': vi_plus - vi_minus
        })


class MassIndex(BaseIndicator):
    """Mass Index"""
    
    def __init__(self):
        super().__init__("MassIndex", "Mass Index reversal indicator")
        self.set_parameter('period', 25)
        self.set_parameter('ema_period', 9)
    
    def get_required_columns(self) -> List[str]:
        return ['high', 'low']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 25))
        ema_period = kwargs.get('ema_period', self._parameters.get('ema_period', 9))
        
        high = data['high']
        low = data['low']
        
        # High-Low range
        hl_range = high - low
        
        # Single and double smoothed EMAs
        ema1 = hl_range.ewm(span=ema_period).mean()
        ema2 = ema1.ewm(span=ema_period).mean()
        
        # Mass Index
        mass_index = (ema1 / ema2).rolling(window=period).sum()
        
        return mass_index


class TrendIntensity(BaseIndicator):
    """Trend Intensity Index"""
    
    def __init__(self):
        super().__init__("TrendIntensity", "Trend Intensity Index")
        self.set_parameter('period', 30)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 30))
        
        close = data['close']
        
        # Simple moving average
        sma = close.rolling(window=period).mean()
        
        # Count periods where close is above/below SMA
        above_sma = (close > sma).rolling(window=period).sum()
        
        # Trend intensity as percentage
        trend_intensity = above_sma / period * 100
        
        return trend_intensity