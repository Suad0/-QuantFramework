"""
Memory optimization utilities for efficient data structures and lazy loading.

This module provides memory-efficient data structures, lazy loading mechanisms,
and memory management utilities to optimize system performance.
"""

import gc
import sys
import weakref
from typing import Any, Dict, Iterator, List, Optional, Union, Callable
import numpy as np
import pandas as pd
from functools import wraps, lru_cache
import logging
from dataclasses import dataclass, field
from collections import OrderedDict
import psutil
import threading
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    max_memory_gb: float = 8.0
    cache_size_mb: float = 512.0
    lazy_loading: bool = True
    compression_level: int = 1  # 0-9, higher = more compression
    gc_threshold: float = 0.8  # Trigger GC when memory usage exceeds this fraction


class MemoryMonitor:
    """
    Monitors system memory usage and triggers optimization actions.
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self._monitoring = False
        self._monitor_thread = None
        self._callbacks = []
        
    def add_callback(self, callback: Callable[[float], None]):
        """Add callback to be called when memory threshold is exceeded."""
        self._callbacks.append(callback)
        
    def start_monitoring(self, interval: float = 5.0):
        """Start memory monitoring in background thread."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
        
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("Memory monitoring stopped")
        
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self._monitoring:
            memory_usage = self.get_memory_usage_fraction()
            
            if memory_usage > self.config.gc_threshold:
                logger.warning(f"Memory usage ({memory_usage:.1%}) exceeds threshold")
                
                # Trigger callbacks
                for callback in self._callbacks:
                    try:
                        callback(memory_usage)
                    except Exception as e:
                        logger.error(f"Memory callback failed: {e}")
                
                # Force garbage collection
                gc.collect()
                
            time.sleep(interval)
    
    def get_memory_usage_fraction(self) -> float:
        """Get current memory usage as fraction of total."""
        memory = psutil.virtual_memory()
        return memory.used / memory.total
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        
        return {
            'system_total_gb': memory.total / (1024**3),
            'system_used_gb': memory.used / (1024**3),
            'system_available_gb': memory.available / (1024**3),
            'system_usage_percent': memory.percent,
            'process_memory_gb': process.memory_info().rss / (1024**3),
            'process_memory_percent': process.memory_percent(),
            'gc_counts': gc.get_count()
        }


class LazyDataFrame:
    """
    Lazy-loading DataFrame that loads data on demand.
    """
    
    def __init__(
        self,
        data_source: Union[str, Callable],
        chunk_size: Optional[int] = None,
        cache_chunks: bool = True
    ):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.cache_chunks = cache_chunks
        self._chunks_cache = {} if cache_chunks else None
        self._metadata = None
        self._loaded_data = None
        
    def _load_metadata(self):
        """Load metadata without loading full data."""
        if self._metadata is not None:
            return
            
        if isinstance(self.data_source, str):
            # For CSV files, read just the header
            sample = pd.read_csv(self.data_source, nrows=1)
            self._metadata = {
                'columns': sample.columns.tolist(),
                'dtypes': sample.dtypes.to_dict()
            }
        elif callable(self.data_source):
            # For callable sources, try to get metadata
            try:
                sample = self.data_source(nrows=1)
                self._metadata = {
                    'columns': sample.columns.tolist(),
                    'dtypes': sample.dtypes.to_dict()
                }
            except:
                self._metadata = {'columns': [], 'dtypes': {}}
    
    @property
    def columns(self) -> List[str]:
        """Get column names without loading data."""
        self._load_metadata()
        return self._metadata.get('columns', [])
    
    @property
    def dtypes(self) -> Dict[str, Any]:
        """Get data types without loading data."""
        self._load_metadata()
        return self._metadata.get('dtypes', {})
    
    def load_chunk(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Load specific chunk of data."""
        cache_key = (start_idx, end_idx)
        
        if self.cache_chunks and cache_key in self._chunks_cache:
            return self._chunks_cache[cache_key]
        
        if isinstance(self.data_source, str):
            chunk = pd.read_csv(
                self.data_source,
                skiprows=start_idx,
                nrows=end_idx - start_idx
            )
        elif callable(self.data_source):
            chunk = self.data_source(start_idx=start_idx, end_idx=end_idx)
        else:
            raise ValueError("Invalid data source type")
        
        if self.cache_chunks:
            self._chunks_cache[cache_key] = chunk
            
        return chunk
    
    def load_full(self) -> pd.DataFrame:
        """Load full DataFrame into memory."""
        if self._loaded_data is not None:
            return self._loaded_data
            
        if isinstance(self.data_source, str):
            self._loaded_data = pd.read_csv(self.data_source)
        elif callable(self.data_source):
            self._loaded_data = self.data_source()
        else:
            raise ValueError("Invalid data source type")
            
        return self._loaded_data
    
    def iterate_chunks(self, chunk_size: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Iterate over data in chunks."""
        chunk_size = chunk_size or self.chunk_size or 10000
        
        if isinstance(self.data_source, str):
            for chunk in pd.read_csv(self.data_source, chunksize=chunk_size):
                yield chunk
        else:
            # For other sources, load full data and chunk it
            data = self.load_full()
            for i in range(0, len(data), chunk_size):
                yield data.iloc[i:i+chunk_size]
    
    def clear_cache(self):
        """Clear cached chunks."""
        if self._chunks_cache:
            self._chunks_cache.clear()
        self._loaded_data = None
        gc.collect()


class MemoryEfficientDataStructures:
    """
    Memory-efficient data structures for financial data.
    """
    
    @staticmethod
    def create_sparse_matrix(
        data: Union[np.ndarray, pd.DataFrame],
        threshold: float = 1e-8
    ) -> Any:
        """Create sparse matrix from dense data."""
        from scipy import sparse
        
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        # Set small values to zero
        data[np.abs(data) < threshold] = 0
        
        # Convert to sparse format
        return sparse.csr_matrix(data)
    
    @staticmethod
    def compress_dataframe(
        df: pd.DataFrame,
        compression_level: int = 1
    ) -> pd.DataFrame:
        """Compress DataFrame to reduce memory usage."""
        compressed_df = df.copy()
        
        for col in compressed_df.columns:
            col_type = compressed_df[col].dtype
            
            if col_type == 'object':
                # Convert to category if beneficial
                if compressed_df[col].nunique() / len(compressed_df) < 0.5:
                    compressed_df[col] = compressed_df[col].astype('category')
                    
            elif 'int' in str(col_type):
                # Downcast integers
                c_min = compressed_df[col].min()
                c_max = compressed_df[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    compressed_df[col] = compressed_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    compressed_df[col] = compressed_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    compressed_df[col] = compressed_df[col].astype(np.int32)
                    
            elif 'float' in str(col_type):
                # Downcast floats
                c_min = compressed_df[col].min()
                c_max = compressed_df[col].max()
                
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    compressed_df[col] = compressed_df[col].astype(np.float32)
        
        return compressed_df
    
    @staticmethod
    def create_memory_mapped_array(
        shape: tuple,
        dtype: np.dtype = np.float64,
        filename: Optional[str] = None
    ) -> np.ndarray:
        """Create memory-mapped array for large datasets."""
        return np.memmap(
            filename,
            dtype=dtype,
            mode='w+',
            shape=shape
        )


class LRUCache:
    """
    Least Recently Used cache with memory management.
    """
    
    def __init__(self, max_size_mb: float = 512.0):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.cache = OrderedDict()
        self.sizes = {}
        self.current_size = 0
        self._lock = threading.RLock()
        
    def _get_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        else:
            return sys.getsizeof(obj)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self._lock:
            size = self._get_size(value)
            
            # Remove existing item if present
            if key in self.cache:
                self.current_size -= self.sizes[key]
                del self.cache[key]
                del self.sizes[key]
            
            # Evict items if necessary
            while self.current_size + size > self.max_size_bytes and self.cache:
                oldest_key = next(iter(self.cache))
                self.current_size -= self.sizes[oldest_key]
                del self.cache[oldest_key]
                del self.sizes[oldest_key]
            
            # Add new item
            if size <= self.max_size_bytes:
                self.cache[key] = value
                self.sizes[key] = size
                self.current_size += size
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self.cache.clear()
            self.sizes.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size_mb': self.current_size / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization': self.current_size / self.max_size_bytes,
                'item_count': len(self.cache)
            }


class MemoryOptimizer:
    """
    Main memory optimization coordinator.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.monitor = MemoryMonitor(self.config)
        self.cache = LRUCache(self.config.cache_size_mb)
        self.data_structures = MemoryEfficientDataStructures()
        
        # Register memory cleanup callback
        self.monitor.add_callback(self._memory_cleanup_callback)
        
    def _memory_cleanup_callback(self, memory_usage: float):
        """Callback for memory cleanup when threshold is exceeded."""
        logger.info(f"Performing memory cleanup (usage: {memory_usage:.1%})")
        
        # Clear cache
        self.cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Log new memory usage
        new_usage = self.monitor.get_memory_usage_fraction()
        logger.info(f"Memory usage after cleanup: {new_usage:.1%}")
    
    @contextmanager
    def memory_context(self, max_memory_gb: Optional[float] = None):
        """Context manager for memory-constrained operations."""
        max_memory = max_memory_gb or self.config.max_memory_gb
        initial_usage = self.monitor.get_memory_usage_fraction()
        
        try:
            yield
        finally:
            current_usage = self.monitor.get_memory_usage_fraction()
            if current_usage > initial_usage + 0.1:  # 10% increase
                logger.warning("Significant memory increase detected, cleaning up")
                self._memory_cleanup_callback(current_usage)
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        return self.data_structures.compress_dataframe(
            df, 
            self.config.compression_level
        )
    
    def create_lazy_dataframe(
        self,
        data_source: Union[str, Callable],
        **kwargs
    ) -> LazyDataFrame:
        """Create lazy-loading DataFrame."""
        return LazyDataFrame(data_source, **kwargs)
    
    def cached_computation(self, cache_key: str):
        """Decorator for caching expensive computations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check cache first
                result = self.cache.get(cache_key)
                if result is not None:
                    return result
                
                # Compute and cache result
                with self.memory_context():
                    result = func(*args, **kwargs)
                    self.cache.put(cache_key, result)
                    
                return result
            return wrapper
        return decorator
    
    def batch_process_large_data(
        self,
        data_source: Union[str, pd.DataFrame, LazyDataFrame],
        processing_func: Callable,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> Iterator[Any]:
        """Process large datasets in memory-efficient batches."""
        batch_size = batch_size or 10000
        
        if isinstance(data_source, LazyDataFrame):
            for chunk in data_source.iterate_chunks(batch_size):
                with self.memory_context():
                    yield processing_func(chunk, **kwargs)
                    
        elif isinstance(data_source, str):
            for chunk in pd.read_csv(data_source, chunksize=batch_size):
                with self.memory_context():
                    yield processing_func(chunk, **kwargs)
                    
        elif isinstance(data_source, pd.DataFrame):
            for i in range(0, len(data_source), batch_size):
                chunk = data_source.iloc[i:i+batch_size]
                with self.memory_context():
                    yield processing_func(chunk, **kwargs)
        else:
            raise ValueError("Unsupported data source type")
    
    def start_monitoring(self):
        """Start memory monitoring."""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitor.stop_monitoring()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        return {
            'memory_stats': self.monitor.get_memory_stats(),
            'cache_stats': self.cache.get_stats(),
            'config': {
                'max_memory_gb': self.config.max_memory_gb,
                'cache_size_mb': self.config.cache_size_mb,
                'lazy_loading': self.config.lazy_loading,
                'compression_level': self.config.compression_level,
                'gc_threshold': self.config.gc_threshold
            }
        }