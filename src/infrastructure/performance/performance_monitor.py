"""
Performance monitoring and profiling utilities.

This module provides comprehensive performance monitoring, profiling,
and benchmarking capabilities for the quantitative framework.
"""

import time
import psutil
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from contextlib import contextmanager
import functools
import cProfile
import pstats
import io
from pathlib import Path
import json
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    function_name: str
    timestamp: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float


class PerformanceCollector:
    """
    Collects and stores performance metrics.
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.system_metrics_history = deque(maxlen=max_history)
        self.function_metrics = defaultdict(list)
        self._lock = threading.RLock()
        
    def record_function_metrics(self, metrics: PerformanceMetrics):
        """Record metrics for a function execution."""
        with self._lock:
            self.metrics_history.append(metrics)
            self.function_metrics[metrics.function_name].append(metrics)
            
            # Limit per-function history
            if len(self.function_metrics[metrics.function_name]) > 1000:
                self.function_metrics[metrics.function_name] = \
                    self.function_metrics[metrics.function_name][-1000:]
    
    def record_system_metrics(self, metrics: SystemMetrics):
        """Record system-wide metrics."""
        with self._lock:
            self.system_metrics_history.append(metrics)
    
    def get_function_stats(self, function_name: str) -> Dict[str, Any]:
        """Get statistics for a specific function."""
        with self._lock:
            metrics = self.function_metrics.get(function_name, [])
            if not metrics:
                return {}
            
            execution_times = [m.execution_time for m in metrics]
            memory_usage = [m.memory_usage_mb for m in metrics]
            
            return {
                'function_name': function_name,
                'call_count': len(metrics),
                'avg_execution_time': np.mean(execution_times),
                'min_execution_time': np.min(execution_times),
                'max_execution_time': np.max(execution_times),
                'std_execution_time': np.std(execution_times),
                'avg_memory_usage_mb': np.mean(memory_usage),
                'max_memory_usage_mb': np.max(memory_usage),
                'total_execution_time': np.sum(execution_times)
            }
    
    def get_top_functions(self, metric: str = 'avg_execution_time', limit: int = 10) -> List[Dict[str, Any]]:
        """Get top functions by specified metric."""
        with self._lock:
            function_stats = []
            for func_name in self.function_metrics:
                stats = self.get_function_stats(func_name)
                if stats and metric in stats:
                    function_stats.append(stats)
            
            return sorted(function_stats, key=lambda x: x[metric], reverse=True)[:limit]


class PerformanceMonitor:
    """
    Main performance monitoring system.
    """
    
    def __init__(self, enable_profiling: bool = False):
        self.collector = PerformanceCollector()
        self.enable_profiling = enable_profiling
        self._monitoring = False
        self._monitor_thread = None
        self._profiler = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring in background thread."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Performance monitoring stopped")
    
    def _monitor_system(self, interval: float):
        """Monitor system metrics in background."""
        prev_disk_io = psutil.disk_io_counters()
        prev_network_io = psutil.net_io_counters()
        
        while self._monitoring:
            try:
                # Get current metrics
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                # Calculate deltas
                disk_read_mb = (disk_io.read_bytes - prev_disk_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (disk_io.write_bytes - prev_disk_io.write_bytes) / (1024 * 1024)
                network_sent_mb = (network_io.bytes_sent - prev_network_io.bytes_sent) / (1024 * 1024)
                network_recv_mb = (network_io.bytes_recv - prev_network_io.bytes_recv) / (1024 * 1024)
                
                metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_percent=psutil.cpu_percent(),
                    memory_percent=memory.percent,
                    memory_used_gb=memory.used / (1024**3),
                    memory_available_gb=memory.available / (1024**3),
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    network_sent_mb=network_sent_mb,
                    network_recv_mb=network_recv_mb
                )
                
                self.collector.record_system_metrics(metrics)
                
                # Update previous values
                prev_disk_io = disk_io
                prev_network_io = network_io
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
            
            time.sleep(interval)
    
    @contextmanager
    def measure_performance(self, function_name: str, **additional_metrics):
        """Context manager for measuring performance."""
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss / (1024 * 1024)
        start_cpu = process.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024 * 1024)
            end_cpu = process.cpu_percent()
            
            metrics = PerformanceMetrics(
                execution_time=end_time - start_time,
                memory_usage_mb=max(end_memory - start_memory, 0),
                cpu_usage_percent=end_cpu - start_cpu,
                function_name=function_name,
                timestamp=end_time,
                additional_metrics=additional_metrics
            )
            
            self.collector.record_function_metrics(metrics)
    
    def profile_function(self, sort_by: str = 'cumulative'):
        """Decorator for profiling function execution."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enable_profiling:
                    return func(*args, **kwargs)
                
                profiler = cProfile.Profile()
                profiler.enable()
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    profiler.disable()
                    
                    # Save profile results
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
                    ps.print_stats()
                    
                    profile_output = s.getvalue()
                    logger.info(f"Profile for {func.__name__}:\n{profile_output}")
                
                return result
            return wrapper
        return decorator
    
    def benchmark_function(self, func: Callable, *args, iterations: int = 100, **kwargs) -> Dict[str, Any]:
        """Benchmark function performance over multiple iterations."""
        execution_times = []
        memory_usage = []
        
        for i in range(iterations):
            with self.measure_performance(f"{func.__name__}_benchmark"):
                func(*args, **kwargs)
            
            # Get last recorded metrics
            if self.collector.metrics_history:
                last_metrics = self.collector.metrics_history[-1]
                execution_times.append(last_metrics.execution_time)
                memory_usage.append(last_metrics.memory_usage_mb)
        
        return {
            'function_name': func.__name__,
            'iterations': iterations,
            'avg_execution_time': np.mean(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'std_execution_time': np.std(execution_times),
            'avg_memory_usage_mb': np.mean(memory_usage),
            'max_memory_usage_mb': np.max(memory_usage),
            'total_time': np.sum(execution_times)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            'timestamp': time.time(),
            'system_metrics': self._get_current_system_metrics(),
            'top_functions_by_time': self.collector.get_top_functions('avg_execution_time'),
            'top_functions_by_memory': self.collector.get_top_functions('avg_memory_usage_mb'),
            'top_functions_by_calls': self.collector.get_top_functions('call_count'),
            'total_functions_monitored': len(self.collector.function_metrics),
            'total_measurements': len(self.collector.metrics_history)
        }
    
    def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        memory = psutil.virtual_memory()
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    
    def export_metrics(self, filepath: str):
        """Export collected metrics to file."""
        data = {
            'function_metrics': {},
            'system_metrics': []
        }
        
        # Export function metrics
        for func_name in self.collector.function_metrics:
            data['function_metrics'][func_name] = self.collector.get_function_stats(func_name)
        
        # Export system metrics
        for metrics in self.collector.system_metrics_history:
            data['system_metrics'].append({
                'timestamp': metrics.timestamp,
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'memory_used_gb': metrics.memory_used_gb,
                'memory_available_gb': metrics.memory_available_gb
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Performance metrics exported to {filepath}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(function_name: Optional[str] = None, **additional_metrics):
    """Decorator for monitoring function performance."""
    def decorator(func):
        name = function_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with performance_monitor.measure_performance(name, **additional_metrics):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def benchmark(iterations: int = 100):
    """Decorator for benchmarking function performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Run benchmark
            results = performance_monitor.benchmark_function(func, *args, iterations=iterations, **kwargs)
            logger.info(f"Benchmark results for {func.__name__}: {results}")
            
            # Return single execution result
            return func(*args, **kwargs)
        return wrapper
    return decorator