#!/usr/bin/env python3
"""
Comprehensive test suite for parallel processing and optimization features.

Tests all components of the performance optimization infrastructure including
parallel processing, GPU acceleration, memory optimization, caching, and monitoring.
"""

import unittest
import sys
import os
import tempfile
import shutil
import time
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from infrastructure.performance import (
    ParallelProcessor, GPUAccelerator, MemoryOptimizer, 
    CacheManager, PerformanceMonitor
)
from infrastructure.performance.parallel_processor import ProcessingConfig, BatchProcessor
from infrastructure.performance.gpu_accelerator import GPUConfig
from infrastructure.performance.memory_optimizer import MemoryConfig, LazyDataFrame
from infrastructure.performance.cache_manager import CacheConfig
from infrastructure.performance.performance_monitor import monitor_performance


class TestParallelProcessor(unittest.TestCase):
    """Test parallel processing functionality."""
    
    def setUp(self):
        self.config = ProcessingConfig(max_workers=2, use_processes=True)
        self.processor = ParallelProcessor(self.config)
        
    def test_map_parallel(self):
        """Test parallel map operation."""
        def square(x):
            return x * x
        
        data = list(range(10))
        results = self.processor.map_parallel(square, data)
        expected = [x * x for x in data]
        
        self.assertEqual(results, expected)
        
    def test_process_dataframe_parallel(self):
        """Test parallel DataFrame processing."""
        df = pd.DataFrame(np.random.randn(100, 5))
        
        def process_chunk(chunk):
            return chunk.mean()
        
        result = self.processor.process_dataframe_parallel(
            df, process_chunk, axis=0, chunk_size=20
        )
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 5)  # 5 columns
        
    def test_process_time_series_parallel(self):
        """Test parallel time series processing."""
        data = {
            'AAPL': pd.DataFrame(np.random.randn(100, 4)),
            'GOOGL': pd.DataFrame(np.random.randn(100, 4)),
            'MSFT': pd.DataFrame(np.random.randn(100, 4))
        }
        
        def calculate_volatility(df):
            return df.std().mean()
        
        results = self.processor.process_time_series_parallel(data, calculate_volatility)
        
        self.assertEqual(len(results), 3)
        self.assertIn('AAPL', results)
        self.assertIn('GOOGL', results)
        self.assertIn('MSFT', results)
        
    def test_monte_carlo_simulation_parallel(self):
        """Test parallel Monte Carlo simulation."""
        def simulation(**kwargs):
            n_steps = kwargs.get('n_steps', 100)
            return np.random.randn(n_steps).sum()
        
        results = self.processor.monte_carlo_simulation_parallel(
            simulation, n_simulations=100, n_steps=50
        )
        
        self.assertEqual(len(results), 100)
        self.assertIsInstance(results[0], (int, float))
        
    def test_optimize_parameters_parallel(self):
        """Test parallel parameter optimization."""
        parameter_sets = [
            {'param1': 1, 'param2': 2},
            {'param1': 2, 'param2': 3},
            {'param1': 3, 'param2': 4}
        ]
        
        def objective_func(params):
            return params['param1'] * params['param2']
        
        results = self.processor.optimize_parameters_parallel(
            parameter_sets, objective_func
        )
        
        self.assertEqual(len(results), 3)
        for params, score in results:
            self.assertIsInstance(params, dict)
            self.assertIsInstance(score, (int, float))
            
    def test_performance_stats(self):
        """Test performance statistics."""
        stats = self.processor.get_performance_stats()
        
        self.assertIn('max_workers', stats)
        self.assertIn('cpu_count', stats)
        self.assertIn('memory_usage_gb', stats)
        self.assertEqual(stats['max_workers'], 2)


class TestGPUAccelerator(unittest.TestCase):
    """Test GPU acceleration functionality."""
    
    def setUp(self):
        self.config = GPUConfig(mixed_precision=False)  # Disable for testing
        self.gpu = GPUAccelerator(self.config)
        
    def test_device_setup(self):
        """Test device setup and configuration."""
        self.assertIsNotNone(self.gpu.device)
        self.assertIsNotNone(self.gpu.device_name)
        self.assertIsInstance(self.gpu.is_cuda, bool)
        
    def test_tensor_operations(self):
        """Test basic tensor operations."""
        # Create numpy array
        array = np.array([[1, 2], [3, 4]], dtype=np.float32)
        
        # Convert to tensor
        tensor = self.gpu.from_numpy(array)
        self.assertEqual(tensor.shape, (2, 2))
        
        # Convert back to numpy
        result_array = self.gpu.to_numpy(tensor)
        np.testing.assert_array_equal(array, result_array)
        
    def test_matrix_operations(self):
        """Test accelerated matrix operations."""
        # Create test matrices
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        
        tensor_a = self.gpu.from_numpy(a)
        tensor_b = self.gpu.from_numpy(b)
        
        # Test matrix multiplication
        result_tensor = self.gpu.accelerated_matrix_operations('matmul', tensor_a, tensor_b)
        result = self.gpu.to_numpy(result_tensor)
        
        expected = np.matmul(a, b)
        np.testing.assert_array_almost_equal(result, expected)
        
    def test_covariance_estimation(self):
        """Test GPU covariance estimation."""
        # Generate test data
        np.random.seed(42)
        returns = np.random.randn(100, 5).astype(np.float32)
        returns_tensor = self.gpu.from_numpy(returns)
        
        # Calculate covariance
        cov_tensor = self.gpu.accelerated_covariance_estimation(returns_tensor, method='sample')
        cov_matrix = self.gpu.to_numpy(cov_tensor)
        
        # Verify properties
        self.assertEqual(cov_matrix.shape, (5, 5))
        # Should be symmetric
        np.testing.assert_array_almost_equal(cov_matrix, cov_matrix.T, decimal=5)
        
    def test_portfolio_optimization(self):
        """Test GPU portfolio optimization."""
        n_assets = 5
        expected_returns = self.gpu.from_numpy(np.random.randn(n_assets).astype(np.float32))
        
        # Create positive definite covariance matrix
        random_matrix = np.random.randn(n_assets, n_assets).astype(np.float32)
        cov_matrix = random_matrix @ random_matrix.T + np.eye(n_assets, dtype=np.float32) * 0.1
        cov_tensor = self.gpu.from_numpy(cov_matrix)
        
        # Optimize portfolio
        weights_tensor = self.gpu.accelerated_portfolio_optimization(
            expected_returns, cov_tensor, risk_aversion=1.0
        )
        weights = self.gpu.to_numpy(weights_tensor)
        
        # Verify constraints
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)  # Budget constraint
        self.assertEqual(len(weights), n_assets)
        
    def test_memory_stats(self):
        """Test GPU memory statistics."""
        stats = self.gpu.get_memory_stats()
        
        self.assertIn('device', stats)
        self.assertIn('cuda_available', stats)
        self.assertIn('mps_available', stats)
        self.assertIn('device_type', stats)
        
        if self.gpu.is_cuda:
            self.assertIn('memory_allocated_gb', stats)
            self.assertIn('memory_total_gb', stats)
        elif self.gpu.is_mps:
            # MPS stats might be None
            self.assertIn('memory_allocated_gb', stats)
            self.assertIn('device_count', stats)


class TestMemoryOptimizer(unittest.TestCase):
    """Test memory optimization functionality."""
    
    def setUp(self):
        self.config = MemoryConfig(max_memory_gb=1.0, cache_size_mb=64.0)
        self.optimizer = MemoryOptimizer(self.config)
        
    def test_dataframe_compression(self):
        """Test DataFrame compression."""
        # Create DataFrame with various data types
        df = pd.DataFrame({
            'int_col': np.random.randint(0, 100, 1000),
            'float_col': np.random.randn(1000),
            'category_col': np.random.choice(['A', 'B', 'C'], 1000),
            'string_col': [f'item_{i%10}' for i in range(1000)]
        })
        
        original_memory = df.memory_usage(deep=True).sum()
        compressed_df = self.optimizer.optimize_dataframe(df)
        compressed_memory = compressed_df.memory_usage(deep=True).sum()
        
        # Should reduce memory usage
        self.assertLessEqual(compressed_memory, original_memory)
        
        # Data should be preserved
        self.assertEqual(len(df), len(compressed_df))
        self.assertEqual(list(df.columns), list(compressed_df.columns))
        
    def test_lazy_dataframe(self):
        """Test lazy DataFrame functionality."""
        # Create temporary CSV file
        df = pd.DataFrame(np.random.randn(1000, 5), columns=[f'col_{i}' for i in range(5)])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Create lazy DataFrame
            lazy_df = LazyDataFrame(csv_path, chunk_size=100)
            
            # Test metadata access
            self.assertEqual(len(lazy_df.columns), 5)
            self.assertIn('col_0', lazy_df.columns)
            
            # Test chunk iteration
            chunks = list(lazy_df.iterate_chunks(chunk_size=200))
            self.assertGreater(len(chunks), 0)
            self.assertLessEqual(len(chunks[0]), 200)
            
            # Test full loading
            full_data = lazy_df.load_full()
            self.assertEqual(full_data.shape, (1000, 5))
            
        finally:
            os.unlink(csv_path)
            
    def test_cached_computation(self):
        """Test cached computation decorator."""
        call_count = 0
        
        @self.optimizer.cached_computation("test_cache")
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive computation
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)
        
        # Second call (should use cache)
        result2 = expensive_function(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # Should not increment
        
    def test_batch_processing(self):
        """Test batch processing of large data."""
        # Create test DataFrame
        df = pd.DataFrame(np.random.randn(1000, 3))
        
        def process_batch(batch):
            return batch.mean()
        
        results = list(self.optimizer.batch_process_large_data(
            df, process_batch, batch_size=200
        ))
        
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], pd.Series)
        
    def test_optimization_stats(self):
        """Test optimization statistics."""
        stats = self.optimizer.get_optimization_stats()
        
        self.assertIn('memory_stats', stats)
        self.assertIn('cache_stats', stats)
        self.assertIn('config', stats)


class TestCacheManager(unittest.TestCase):
    """Test cache management functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = CacheConfig(
            memory_cache_size_mb=32.0,
            disk_cache_size_gb=0.1,
            enable_disk_cache=True,
            cache_directory=self.temp_dir
        )
        self.cache_manager = CacheManager(self.config)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_basic_caching(self):
        """Test basic cache operations."""
        # Put item in cache
        self.cache_manager.put("test_key", "test_value", ttl_seconds=60)
        
        # Get item from cache
        result = self.cache_manager.get("test_key")
        self.assertEqual(result, "test_value")
        
        # Test non-existent key
        result = self.cache_manager.get("non_existent")
        self.assertIsNone(result)
        
    def test_cache_decorator(self):
        """Test cache decorator functionality."""
        call_count = 0
        
        @self.cache_manager.cached(ttl_seconds=60, tags=['test'])
        def cached_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = cached_function(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(call_count, 1)
        
        # Second call (should use cache)
        result2 = cached_function(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(call_count, 1)
        
        # Different arguments (should compute)
        result3 = cached_function(2, 3)
        self.assertEqual(result3, 5)
        self.assertEqual(call_count, 2)
        
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        # Put item in cache
        self.cache_manager.put("test_key", "test_value", tags=['test_tag'])
        
        # Verify it's cached
        result = self.cache_manager.get("test_key")
        self.assertEqual(result, "test_value")
        
        # Invalidate by key
        self.cache_manager.invalidate("test_key")
        result = self.cache_manager.get("test_key")
        self.assertIsNone(result)
        
        # Test tag-based invalidation
        self.cache_manager.put("key1", "value1", tags=['tag1'])
        self.cache_manager.put("key2", "value2", tags=['tag1'])
        self.cache_manager.put("key3", "value3", tags=['tag2'])
        
        self.cache_manager.invalidate_by_tags(['tag1'])
        
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertIsNone(self.cache_manager.get("key2"))
        self.assertEqual(self.cache_manager.get("key3"), "value3")
        
    def test_cache_stats(self):
        """Test cache statistics."""
        # Add some items to cache
        for i in range(5):
            self.cache_manager.put(f"key_{i}", f"value_{i}")
        
        stats = self.cache_manager.get_stats()
        
        self.assertIn('memory_cache', stats)
        self.assertIn('config', stats)
        self.assertGreater(stats['memory_cache']['entry_count'], 0)
        
    def test_ttl_expiration(self):
        """Test TTL expiration."""
        # Put item with short TTL
        self.cache_manager.put("short_ttl_key", "value", ttl_seconds=1)
        
        # Should be available immediately
        result = self.cache_manager.get("short_ttl_key")
        self.assertEqual(result, "value")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        result = self.cache_manager.get("short_ttl_key")
        self.assertIsNone(result)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring functionality."""
    
    def setUp(self):
        self.monitor = PerformanceMonitor(enable_profiling=False)
        
    def test_performance_measurement(self):
        """Test performance measurement context manager."""
        with self.monitor.measure_performance("test_function"):
            time.sleep(0.01)  # Simulate work
            
        # Check that metrics were recorded
        self.assertGreater(len(self.monitor.collector.metrics_history), 0)
        
        metrics = self.monitor.collector.metrics_history[-1]
        self.assertEqual(metrics.function_name, "test_function")
        self.assertGreater(metrics.execution_time, 0.01)
        
    def test_monitor_decorator(self):
        """Test performance monitoring decorator."""
        @monitor_performance("decorated_function")
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        self.assertEqual(result, "result")
        
        # Check metrics were recorded
        stats = self.monitor.collector.get_function_stats("decorated_function")
        self.assertGreater(stats['call_count'], 0)
        self.assertGreater(stats['avg_execution_time'], 0)
        
    def test_benchmarking(self):
        """Test function benchmarking."""
        def simple_function(n):
            return sum(range(n))
        
        results = self.monitor.benchmark_function(simple_function, 100, iterations=5)
        
        self.assertIn('function_name', results)
        self.assertIn('iterations', results)
        self.assertIn('avg_execution_time', results)
        self.assertEqual(results['iterations'], 5)
        self.assertGreater(results['avg_execution_time'], 0)
        
    def test_performance_report(self):
        """Test performance report generation."""
        # Generate some metrics
        with self.monitor.measure_performance("test_func_1"):
            time.sleep(0.01)
            
        with self.monitor.measure_performance("test_func_2"):
            time.sleep(0.005)
            
        report = self.monitor.get_performance_report()
        
        self.assertIn('timestamp', report)
        self.assertIn('system_metrics', report)
        self.assertIn('top_functions_by_time', report)
        self.assertIn('total_functions_monitored', report)
        
    def test_system_monitoring(self):
        """Test system monitoring functionality."""
        # Start monitoring
        self.monitor.start_monitoring(interval=0.1)
        
        # Wait a bit
        time.sleep(0.3)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Check that system metrics were collected
        self.assertGreater(len(self.monitor.collector.system_metrics_history), 0)
        
        metrics = self.monitor.collector.system_metrics_history[-1]
        self.assertGreater(metrics.timestamp, 0)
        self.assertGreaterEqual(metrics.cpu_percent, 0)
        self.assertGreaterEqual(metrics.memory_percent, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for performance optimization components."""
    
    def test_parallel_processing_with_caching(self):
        """Test parallel processing combined with caching."""
        # Setup components
        processor = ParallelProcessor(ProcessingConfig(max_workers=2))
        cache_manager = CacheManager(CacheConfig(memory_cache_size_mb=64.0))
        
        @cache_manager.cached(ttl_seconds=60)
        def expensive_computation(data):
            time.sleep(0.01)  # Simulate expensive operation
            return data.sum()
        
        # Generate test data
        data_list = [pd.Series(np.random.randn(100)) for _ in range(10)]
        
        # Process in parallel with caching
        results = processor.map_parallel(expensive_computation, data_list)
        
        self.assertEqual(len(results), 10)
        self.assertTrue(all(isinstance(r, (int, float)) for r in results))
        
    def test_gpu_with_memory_optimization(self):
        """Test GPU acceleration with memory optimization."""
        gpu = GPUAccelerator(GPUConfig())
        memory_optimizer = MemoryOptimizer(MemoryConfig(max_memory_gb=2.0))
        
        # Generate large dataset
        large_data = np.random.randn(1000, 100).astype(np.float32)
        
        with memory_optimizer.memory_context():
            # Convert to GPU tensor
            tensor = gpu.from_numpy(large_data)
            
            # Perform GPU computation
            result_tensor = gpu.accelerated_matrix_operations('matmul', tensor, tensor.T)
            result = gpu.to_numpy(result_tensor)
            
        self.assertEqual(result.shape, (1000, 1000))
        
    def test_full_pipeline_optimization(self):
        """Test complete optimization pipeline."""
        # Setup all components
        processor = ParallelProcessor(ProcessingConfig(max_workers=2))
        cache_manager = CacheManager(CacheConfig(memory_cache_size_mb=32.0))
        memory_optimizer = MemoryOptimizer(MemoryConfig(max_memory_gb=1.0))
        monitor = PerformanceMonitor()
        
        @cache_manager.cached(ttl_seconds=60)
        @monitor_performance("pipeline_function")
        def optimized_computation(data_chunk):
            # Simulate complex computation
            return data_chunk.rolling(window=5).mean().std()
        
        # Generate test data
        large_df = pd.DataFrame(np.random.randn(1000, 10))
        
        # Process with full optimization pipeline
        with memory_optimizer.memory_context():
            results = []
            for chunk in memory_optimizer.batch_process_large_data(
                large_df, optimized_computation, batch_size=200
            ):
                results.append(chunk)
        
        self.assertGreater(len(results), 0)
        
        # Check monitoring results
        stats = monitor.collector.get_function_stats("pipeline_function")
        self.assertGreater(stats['call_count'], 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)