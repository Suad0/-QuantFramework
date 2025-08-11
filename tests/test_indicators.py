"""
Test suite for technical indicators library
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.application.services.feature_engine import FeatureEngine, IndicatorRegistry
from src.infrastructure.indicators import *


class TestIndicatorLibrary:
    """Test comprehensive indicator library"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        n = len(dates)
        
        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n),
            'benchmark': prices * (1 + np.random.normal(0, 0.001, n)),
            'sector': np.random.choice(['Tech', 'Finance', 'Healthcare'], n)
        })
        
        # Ensure high >= low and realistic OHLC relationships
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        return data.set_index('date')
    
    def test_indicator_registry_initialization(self):
        """Test that indicator registry loads all built-in indicators"""
        registry = IndicatorRegistry()
        indicators = registry.list_indicators()
        
        # Should have loaded many indicators
        assert len(indicators) > 50
        
        # Check for key indicators from each category
        expected_indicators = [
            'RSI', 'MACD', 'BollingerBands', 'ATR', 'OBV', 'SMA', 'EMA',
            'ADX', 'Aroon', 'StochasticRSI', 'RelativeStrength'
        ]
        
        for indicator in expected_indicators:
            assert indicator in indicators, f"Missing indicator: {indicator}"
    
    def test_momentum_indicators(self, sample_data):
        """Test momentum indicators"""
        registry = IndicatorRegistry()
        
        # Test RSI
        rsi = registry.get_indicator('RSI', period=14)
        rsi_values = rsi.calculate(sample_data)
        assert isinstance(rsi_values, pd.Series)
        assert 0 <= rsi_values.max() <= 100
        assert 0 <= rsi_values.min() <= 100
        
        # Test MACD
        macd = registry.get_indicator('MACD')
        macd_values = macd.calculate(sample_data)
        assert isinstance(macd_values, pd.DataFrame)
        assert 'MACD' in macd_values.columns
        assert 'Signal' in macd_values.columns
        assert 'Histogram' in macd_values.columns
    
    def test_volatility_indicators(self, sample_data):
        """Test volatility indicators"""
        registry = IndicatorRegistry()
        
        # Test Bollinger Bands
        bb = registry.get_indicator('BollingerBands', period=20, std_dev=2)
        bb_values = bb.calculate(sample_data)
        assert isinstance(bb_values, pd.DataFrame)
        assert 'Upper' in bb_values.columns
        assert 'Lower' in bb_values.columns
        assert 'Middle' in bb_values.columns
        
        # Test ATR
        atr = registry.get_indicator('ATR', period=14)
        atr_values = atr.calculate(sample_data)
        assert isinstance(atr_values, pd.Series)
        assert (atr_values >= 0).all()
    
    def test_volume_indicators(self, sample_data):
        """Test volume indicators"""
        registry = IndicatorRegistry()
        
        # Test OBV
        obv = registry.get_indicator('OBV')
        obv_values = obv.calculate(sample_data)
        assert isinstance(obv_values, pd.Series)
        
        # Test A/D Line
        ad = registry.get_indicator('AD')
        ad_values = ad.calculate(sample_data)
        assert isinstance(ad_values, pd.Series)
    
    def test_statistical_indicators(self, sample_data):
        """Test statistical indicators"""
        registry = IndicatorRegistry()
        
        # Test SMA
        sma = registry.get_indicator('SMA', period=20)
        sma_values = sma.calculate(sample_data)
        assert isinstance(sma_values, pd.Series)
        
        # Test EMA
        ema = registry.get_indicator('EMA', period=20)
        ema_values = ema.calculate(sample_data)
        assert isinstance(ema_values, pd.Series)
        
        # Test Z-Score
        zscore = registry.get_indicator('ZScore', period=20)
        zscore_values = zscore.calculate(sample_data)
        assert isinstance(zscore_values, pd.Series)
    
    def test_trend_indicators(self, sample_data):
        """Test trend indicators"""
        registry = IndicatorRegistry()
        
        # Test ADX
        adx = registry.get_indicator('ADX', period=14)
        adx_values = adx.calculate(sample_data)
        assert isinstance(adx_values, pd.DataFrame)
        assert 'ADX' in adx_values.columns
        
        # Test Aroon
        aroon = registry.get_indicator('Aroon', period=25)
        aroon_values = aroon.calculate(sample_data)
        assert isinstance(aroon_values, pd.DataFrame)
        assert 'Aroon_Up' in aroon_values.columns
        assert 'Aroon_Down' in aroon_values.columns
    
    def test_oscillators(self, sample_data):
        """Test oscillator indicators"""
        registry = IndicatorRegistry()
        
        # Test Stochastic RSI
        stoch_rsi = registry.get_indicator('StochasticRSI')
        stoch_rsi_values = stoch_rsi.calculate(sample_data)
        assert isinstance(stoch_rsi_values, pd.DataFrame)
        assert 'StochRSI' in stoch_rsi_values.columns
        
        # Test PPO
        ppo = registry.get_indicator('PPO')
        ppo_values = ppo.calculate(sample_data)
        assert isinstance(ppo_values, pd.DataFrame)
        assert 'PPO' in ppo_values.columns
    
    def test_cross_sectional_indicators(self, sample_data):
        """Test cross-sectional indicators"""
        registry = IndicatorRegistry()
        
        # Test Relative Strength
        rel_strength = registry.get_indicator('RelativeStrength', benchmark_column='benchmark')
        rs_values = rel_strength.calculate(sample_data)
        assert isinstance(rs_values, pd.Series)
        
        # Test Sector Momentum
        sector_mom = registry.get_indicator('SectorMomentum', sector_column='sector')
        sm_values = sector_mom.calculate(sample_data)
        assert isinstance(sm_values, pd.Series)
    
    def test_feature_engine_integration(self, sample_data):
        """Test feature engine with multiple indicators"""
        engine = FeatureEngine()
        
        feature_config = {
            'indicators': {
                'RSI': {'period': 14},
                'MACD': {'fast_period': 12, 'slow_period': 26},
                'BollingerBands': {'period': 20, 'std_dev': 2},
                'SMA': {'period': 50},
                'ATR': {'period': 14}
            },
            'cross_sectional': {
                'relative_strength': True,
                'sector_column': 'sector'
            }
        }
        
        result = engine.compute_features(sample_data, feature_config)
        
        # Should have original columns plus new features
        assert len(result.columns) > len(sample_data.columns)
        
        # Check for specific feature columns
        feature_columns = [col for col in result.columns if col not in sample_data.columns]
        assert len(feature_columns) > 5
        
        # Validate feature computation
        validation_report = engine.validate_features(result)
        assert validation_report.statistics['total_features'] > 0
    
    def test_vectorized_operations(self, sample_data):
        """Test that indicators use vectorized operations efficiently"""
        registry = IndicatorRegistry()
        
        # Test with large dataset
        large_data = pd.concat([sample_data] * 10, ignore_index=True)
        
        # Time the calculation (should be fast due to vectorization)
        import time
        start_time = time.time()
        
        rsi = registry.get_indicator('RSI', period=14)
        rsi_values = rsi.calculate(large_data)
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        # Should complete quickly (less than 1 second for this size)
        assert calculation_time < 1.0
        assert len(rsi_values) == len(large_data)
    
    def test_parameter_validation(self, sample_data):
        """Test parameter validation in indicators"""
        registry = IndicatorRegistry()
        
        # Test with invalid parameters
        with pytest.raises(Exception):
            rsi = registry.get_indicator('RSI', period=0)  # Invalid period
            rsi.calculate(sample_data)
        
        # Test with missing required columns
        incomplete_data = sample_data[['close']].copy()
        
        with pytest.raises(Exception):
            atr = registry.get_indicator('ATR')  # Requires high, low, close
            atr.calculate(incomplete_data)
    
    def test_custom_indicator_registration(self, sample_data):
        """Test custom indicator registration"""
        from src.application.services.feature_engine import BaseIndicator
        
        class CustomIndicator(BaseIndicator):
            def __init__(self):
                super().__init__("CustomTest", "Custom test indicator")
                self.set_parameter('multiplier', 2.0)
            
            def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
                self.validate_data(data)
                multiplier = kwargs.get('multiplier', self._parameters.get('multiplier', 2.0))
                return data['close'] * multiplier
        
        engine = FeatureEngine()
        custom_indicator = CustomIndicator()
        engine.register_indicator("CustomTest", custom_indicator)
        
        # Test that custom indicator works
        feature_config = {
            'indicators': {
                'CustomTest': {'multiplier': 3.0}
            }
        }
        
        result = engine.compute_features(sample_data, feature_config)
        custom_column = [col for col in result.columns if 'CustomTest' in col][0]
        
        # Verify custom calculation
        expected_values = sample_data['close'] * 3.0
        pd.testing.assert_series_equal(
            result[custom_column], 
            expected_values, 
            check_names=False
        )
    
    def test_indicator_count(self):
        """Test that we have 50+ indicators as required"""
        registry = IndicatorRegistry()
        indicators = registry.list_indicators()
        
        print(f"Total indicators loaded: {len(indicators)}")
        print("Indicators:", sorted(indicators))
        
        # Requirement: 50+ indicators
        assert len(indicators) >= 50, f"Expected 50+ indicators, got {len(indicators)}"
    
    def test_performance_optimization(self, sample_data):
        """Test performance optimization features"""
        engine = FeatureEngine()
        
        # Test parallel processing capability
        feature_config = {
            'indicators': {
                'RSI': {'period': 14},
                'MACD': {'fast_period': 12, 'slow_period': 26},
                'BollingerBands': {'period': 20},
                'SMA': {'period': 20},
                'EMA': {'period': 20}
            },
            'parallel': True  # Enable parallel processing
        }
        
        result = engine.compute_features(sample_data, feature_config)
        
        # Should complete successfully with parallel processing
        assert len(result.columns) > len(sample_data.columns)
        
        # Test sequential vs parallel (both should give same results)
        feature_config['parallel'] = False
        result_sequential = engine.compute_features(sample_data, feature_config)
        
        # Results should be identical (allowing for small floating point differences)
        for col in result.columns:
            if col in result_sequential.columns:
                if result[col].dtype in ['float64', 'float32']:
                    pd.testing.assert_series_equal(
                        result[col], 
                        result_sequential[col], 
                        check_names=False,
                        rtol=1e-10
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])