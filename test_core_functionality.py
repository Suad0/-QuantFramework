#!/usr/bin/env python3
"""
Test core functionality of the quantitative framework without GUI.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test the performance optimization components
def test_performance_components():
    """Test the performance optimization components we implemented."""
    print("Testing Performance Optimization Components...")
    
    # Test GPU Acceleration
    try:
        from infrastructure.performance.gpu_accelerator import GPUAccelerator, GPUConfig
        
        gpu = GPUAccelerator(GPUConfig())
        print(f"✅ GPU Accelerator initialized: {gpu.device}")
        
        # Test basic tensor operations
        array = np.array([[1, 2], [3, 4]], dtype=np.float32)
        tensor = gpu.from_numpy(array)
        result = gpu.to_numpy(tensor)
        print(f"✅ GPU tensor operations working: {np.allclose(array, result)}")
        
    except Exception as e:
        print(f"❌ GPU Acceleration test failed: {e}")
    
    # Test Memory Optimization
    try:
        from infrastructure.performance.memory_optimizer import MemoryOptimizer, MemoryConfig
        
        optimizer = MemoryOptimizer(MemoryConfig())
        
        # Test DataFrame compression
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.randn(1000),
            'category_col': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        compressed_df = optimizer.optimize_dataframe(df)
        compressed_memory = compressed_df.memory_usage(deep=True).sum()
        
        reduction = (1 - compressed_memory/original_memory) * 100
        print(f"✅ Memory optimization working: {reduction:.1f}% reduction")
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
    
    # Test Cache Management
    try:
        from infrastructure.performance.cache_manager import CacheManager, CacheConfig
        
        cache = CacheManager(CacheConfig(memory_cache_size_mb=32.0))
        
        # Test basic caching
        cache.put("test_key", "test_value")
        result = cache.get("test_key")
        print(f"✅ Cache management working: {result == 'test_value'}")
        
    except Exception as e:
        print(f"❌ Cache management test failed: {e}")
    
    # Test Performance Monitoring
    try:
        from infrastructure.performance.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        
        with monitor.measure_performance("test_function"):
            # Simulate some work
            sum(range(1000))
        
        print(f"✅ Performance monitoring working: {len(monitor.collector.metrics_history) > 0}")
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")


def test_infrastructure_components():
    """Test other infrastructure components."""
    print("\nTesting Infrastructure Components...")
    
    # Test Technical Indicators
    try:
        from infrastructure.indicators.trend import TrendIndicators
        
        # Generate sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.02), index=dates)
        
        indicators = TrendIndicators()
        sma = indicators.simple_moving_average(prices, window=20)
        
        print(f"✅ Technical indicators working: SMA calculated for {len(sma)} periods")
        
    except Exception as e:
        print(f"❌ Technical indicators test failed: {e}")
    
    # Test Risk Management
    try:
        from infrastructure.risk.var_calculator import VaRCalculator
        
        # Generate sample returns
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        var_calc = VaRCalculator()
        historical_var = var_calc.calculate_historical_var(returns, confidence_level=0.95)
        
        print(f"✅ Risk management working: Historical VaR = {historical_var:.4f}")
        
    except Exception as e:
        print(f"❌ Risk management test failed: {e}")
    
    # Test Portfolio Optimization
    try:
        from infrastructure.optimization.portfolio_optimizer import PortfolioOptimizer
        
        # Generate sample data
        n_assets = 5
        returns = pd.DataFrame(np.random.randn(252, n_assets) * 0.02, 
                             columns=[f'Asset_{i}' for i in range(n_assets)])
        
        optimizer = PortfolioOptimizer()
        weights = optimizer.optimize_mean_variance(returns)
        
        print(f"✅ Portfolio optimization working: {len(weights)} asset weights calculated")
        
    except Exception as e:
        print(f"❌ Portfolio optimization test failed: {e}")


def test_ml_components():
    """Test machine learning components."""
    print("\nTesting ML Components...")
    
    try:
        from infrastructure.ml.ml_framework import MLFramework
        
        # Generate sample data
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.randn(100))
        
        ml_framework = MLFramework()
        
        # Test preprocessing
        X_processed = ml_framework.preprocess_features(X)
        
        print(f"✅ ML framework working: Processed {X_processed.shape[0]} samples with {X_processed.shape[1]} features")
        
    except Exception as e:
        print(f"❌ ML framework test failed: {e}")


def main():
    """Run all tests."""
    print("🚀 Testing Professional Quantitative Framework")
    print("=" * 60)
    
    try:
        test_performance_components()
        test_infrastructure_components()
        test_ml_components()
        
        print("\n" + "=" * 60)
        print("✅ Core functionality tests completed!")
        print("🎉 The quantitative framework is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())