"""
Example custom indicator plugin

This demonstrates how to create custom indicators that can be loaded
into the feature engine using the plugin architecture.
"""
import numpy as np
import pandas as pd
from src.application.services.feature_engine import BaseIndicator


class CustomMomentumIndicator(BaseIndicator):
    """Custom momentum indicator example"""
    
    def __init__(self):
        super().__init__("CustomMomentum", "Custom momentum indicator with decay")
        self.set_parameter('period', 20)
        self.set_parameter('decay_factor', 0.9)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        decay_factor = kwargs.get('decay_factor', self._parameters.get('decay_factor', 0.9))
        
        close = data['close']
        returns = close.pct_change()
        
        # Apply exponential decay to returns
        weights = np.array([decay_factor ** i for i in range(period)])
        weights = weights / weights.sum()  # Normalize
        
        def weighted_momentum(x):
            if len(x) < period:
                return np.nan
            return np.sum(x[-period:] * weights)
        
        momentum = returns.rolling(window=period).apply(weighted_momentum, raw=True)
        return momentum


class AdvancedVolatilityIndicator(BaseIndicator):
    """Advanced volatility indicator with multiple timeframes"""
    
    def __init__(self):
        super().__init__("AdvancedVolatility", "Multi-timeframe volatility indicator")
        self.set_parameter('short_period', 10)
        self.set_parameter('long_period', 30)
        self.set_parameter('threshold', 0.02)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        short_period = kwargs.get('short_period', self._parameters.get('short_period', 10))
        long_period = kwargs.get('long_period', self._parameters.get('long_period', 30))
        threshold = kwargs.get('threshold', self._parameters.get('threshold', 0.02))
        
        close = data['close']
        returns = close.pct_change()
        
        # Short-term volatility
        short_vol = returns.rolling(window=short_period).std()
        
        # Long-term volatility
        long_vol = returns.rolling(window=long_period).std()
        
        # Volatility ratio
        vol_ratio = short_vol / long_vol
        
        # Volatility regime (high/low based on threshold)
        vol_regime = (short_vol > threshold).astype(int)
        
        return pd.DataFrame({
            'Short_Vol': short_vol,
            'Long_Vol': long_vol,
            'Vol_Ratio': vol_ratio,
            'Vol_Regime': vol_regime
        })


class MarketMicrostructureIndicator(BaseIndicator):
    """Market microstructure indicator using OHLCV data"""
    
    def __init__(self):
        super().__init__("MarketMicrostructure", "Market microstructure analysis")
        self.set_parameter('period', 20)
    
    def get_required_columns(self):
        return ['open', 'high', 'low', 'close', 'volume']
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self.validate_data(data)
        period = kwargs.get('period', self._parameters.get('period', 20))
        
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Price impact (how much price moves per unit volume)
        price_change = np.abs(close - open_price)
        price_impact = price_change / volume
        price_impact_ma = price_impact.rolling(window=period).mean()
        
        # Intraday range as percentage of close
        intraday_range = (high - low) / close
        range_ma = intraday_range.rolling(window=period).mean()
        
        # Volume-weighted price change
        vw_price_change = (close - open_price) * volume
        vw_price_change_ma = vw_price_change.rolling(window=period).mean()
        
        # Market efficiency (how close close is to VWAP-like measure)
        typical_price = (high + low + close) / 3
        vwap_approx = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        efficiency = 1 - np.abs(close - vwap_approx) / close
        
        return pd.DataFrame({
            'Price_Impact': price_impact_ma,
            'Intraday_Range': range_ma,
            'VW_Price_Change': vw_price_change_ma,
            'Market_Efficiency': efficiency
        })


# Example of a composite indicator that uses multiple base indicators
class CompositeSignalIndicator(BaseIndicator):
    """Composite signal combining multiple indicators"""
    
    def __init__(self):
        super().__init__("CompositeSignal", "Composite signal from multiple indicators")
        self.set_parameter('rsi_period', 14)
        self.set_parameter('macd_fast', 12)
        self.set_parameter('macd_slow', 26)
        self.set_parameter('bb_period', 20)
    
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        self.validate_data(data)
        rsi_period = kwargs.get('rsi_period', self._parameters.get('rsi_period', 14))
        macd_fast = kwargs.get('macd_fast', self._parameters.get('macd_fast', 12))
        macd_slow = kwargs.get('macd_slow', self._parameters.get('macd_slow', 26))
        bb_period = kwargs.get('bb_period', self._parameters.get('bb_period', 20))
        
        close = data['close']
        
        # Calculate RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema_fast = close.ewm(span=macd_fast).mean()
        ema_slow = close.ewm(span=macd_slow).mean()
        macd = ema_fast - ema_slow
        
        # Calculate Bollinger Band position
        sma = close.rolling(window=bb_period).mean()
        std = close.rolling(window=bb_period).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        bb_position = (close - lower_band) / (upper_band - lower_band)
        
        # Composite signal (normalized and weighted)
        rsi_signal = (rsi - 50) / 50  # Normalize RSI to [-1, 1]
        macd_signal = np.tanh(macd / close.rolling(20).std())  # Normalize MACD
        bb_signal = (bb_position - 0.5) * 2  # Normalize BB position to [-1, 1]
        
        # Weighted composite (equal weights for simplicity)
        composite = (rsi_signal + macd_signal + bb_signal) / 3
        
        return composite