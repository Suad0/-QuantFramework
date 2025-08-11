"""
Parallel processing utilities for CPU-intensive calculations.

This module provides multiprocessing support for computationally intensive tasks
such as backtesting, optimization, and feature engineering.
"""

import multiprocessing as mp
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from functools import partial
import logging
from dataclasses import dataclass
import time
import psutil

logger = logging.getLogger(__name__)


def _run_simulation_with_seed(args):
    """Helper function for Monte Carlo simulation with seed."""
    simulation_func, seed, kwargs = args
    # Set different random seed for each simulation
    np.random.seed(seed)
    return simulation_func(**kwargs)


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""
    max_workers: Optional[int] = None
    chunk_size: Optional[int] = None
    use_processes: bool = True  # True for CPU-bound, False for I/O-bound
    memory_limit_gb: float = 8.0
    timeout_seconds: Optional[float] = None


class ParallelProcessor:
    """
    Handles parallel processing for CPU-intensive calculations.
    
    Provides utilities for distributing work across multiple processes
    with automatic resource management and error handling.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._setup_workers()
        
    def _setup_workers(self):
        """Setup worker configuration based on system resources."""
        if self.config.max_workers is None:
            # Use 75% of available cores, leaving some for system processes
            self.config.max_workers = max(1, int(mp.cpu_count() * 0.75))
        
        # Ensure we respect the configured max_workers even if it's set explicitly
        # Monitor memory usage
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < self.config.memory_limit_gb:
            # Reduce workers if memory is limited
            memory_ratio = available_memory_gb / self.config.memory_limit_gb
            suggested_workers = max(1, int(self.config.max_workers * memory_ratio))
            # Only reduce if memory is really constrained
            if memory_ratio < 0.5:
                self.config.max_workers = suggested_workers
            
        logger.info(f"Initialized ParallelProcessor with {self.config.max_workers} workers")
    
    def map_parallel(
        self,
        func: Callable,
        iterable: List[Any],
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> List[Any]:
        """
        Apply function to iterable in parallel.
        
        Args:
            func: Function to apply
            iterable: Items to process
            chunk_size: Size of chunks for processing
            **kwargs: Additional arguments for function
            
        Returns:
            List of results
        """
        if not iterable:
            return []
            
        chunk_size = chunk_size or self.config.chunk_size or max(1, len(iterable) // self.config.max_workers)
        
        # Prepare function with kwargs
        if kwargs:
            func = partial(func, **kwargs)
            
        start_time = time.time()
        
        try:
            if self.config.use_processes:
                with mp.Pool(processes=self.config.max_workers) as pool:
                    results = pool.map(func, iterable, chunksize=chunk_size)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    results = list(executor.map(func, iterable))
                    
            processing_time = time.time() - start_time
            logger.info(f"Parallel processing completed in {processing_time:.2f}s for {len(iterable)} items")
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            raise
    
    def process_dataframe_parallel(
        self,
        df: pd.DataFrame,
        func: Callable,
        axis: int = 0,
        chunk_size: Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process DataFrame in parallel along specified axis.
        
        Args:
            df: DataFrame to process
            func: Function to apply to each chunk
            axis: Axis to split along (0 for rows, 1 for columns)
            chunk_size: Size of chunks
            **kwargs: Additional arguments for function
            
        Returns:
            Processed DataFrame
        """
        if df.empty:
            return df
            
        chunk_size = chunk_size or max(1, len(df) // self.config.max_workers)
        
        # Split DataFrame into chunks
        if axis == 0:
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        else:
            chunks = [df.iloc[:, i:i+chunk_size] for i in range(0, df.shape[1], chunk_size)]
        
        # Process chunks in parallel
        processed_chunks = self.map_parallel(func, chunks, **kwargs)
        
        # Combine results
        if axis == 0:
            result = pd.concat(processed_chunks, axis=0, ignore_index=True)
        else:
            result = pd.concat(processed_chunks, axis=1)
            
        return result
    
    def process_time_series_parallel(
        self,
        data: Dict[str, pd.DataFrame],
        func: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multiple time series in parallel.
        
        Args:
            data: Dictionary of symbol -> DataFrame
            func: Function to apply to each time series
            **kwargs: Additional arguments for function
            
        Returns:
            Dictionary of symbol -> result
        """
        symbols = list(data.keys())
        dataframes = list(data.values())
        
        # Process in parallel
        results = self.map_parallel(func, dataframes, **kwargs)
        
        # Return as dictionary
        return dict(zip(symbols, results))
    
    def backtest_strategies_parallel(
        self,
        strategies: List[Any],
        data: pd.DataFrame,
        backtest_func: Callable,
        **kwargs
    ) -> List[Any]:
        """
        Run multiple strategy backtests in parallel.
        
        Args:
            strategies: List of strategy objects
            data: Market data
            backtest_func: Backtesting function
            **kwargs: Additional arguments for backtesting
            
        Returns:
            List of backtest results
        """
        # Create partial function with data and kwargs
        backtest_partial = partial(backtest_func, data=data, **kwargs)
        
        # Run backtests in parallel
        return self.map_parallel(backtest_partial, strategies)
    
    def optimize_parameters_parallel(
        self,
        parameter_sets: List[Dict[str, Any]],
        objective_func: Callable,
        **kwargs
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Optimize strategy parameters in parallel.
        
        Args:
            parameter_sets: List of parameter dictionaries to test
            objective_func: Function to evaluate parameters
            **kwargs: Additional arguments for objective function
            
        Returns:
            List of (parameters, score) tuples
        """
        # Create evaluation function
        def evaluate_params(params: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
            try:
                score = objective_func(params, **kwargs)
                return params, score
            except Exception as e:
                logger.warning(f"Parameter evaluation failed for {params}: {e}")
                return params, float('-inf')
        
        return self.map_parallel(evaluate_params, parameter_sets)
    
    def calculate_indicators_parallel(
        self,
        data: pd.DataFrame,
        indicators: List[Callable],
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate multiple technical indicators in parallel.
        
        Args:
            data: Price data
            indicators: List of indicator functions
            **kwargs: Additional arguments for indicators
            
        Returns:
            DataFrame with calculated indicators
        """
        # Calculate indicators in parallel
        def calc_indicator(indicator_func: Callable) -> pd.Series:
            try:
                return indicator_func(data, **kwargs)
            except Exception as e:
                logger.warning(f"Indicator calculation failed: {e}")
                return pd.Series(index=data.index, dtype=float)
        
        results = self.map_parallel(calc_indicator, indicators)
        
        # Combine results into DataFrame
        indicator_df = pd.concat(results, axis=1)
        indicator_df.columns = [func.__name__ for func in indicators]
        
        return indicator_df
    
    def monte_carlo_simulation_parallel(
        self,
        simulation_func: Callable,
        n_simulations: int,
        **kwargs
    ) -> List[Any]:
        """
        Run Monte Carlo simulations in parallel.
        
        Args:
            simulation_func: Function to run single simulation
            n_simulations: Number of simulations to run
            **kwargs: Arguments for simulation function
            
        Returns:
            List of simulation results
        """
        # Create simulation tasks with different seeds and kwargs
        simulation_tasks = [
            (simulation_func, i, kwargs) for i in range(n_simulations)
        ]
        
        return self.map_parallel(_run_simulation_with_seed, simulation_tasks)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            'max_workers': self.config.max_workers,
            'cpu_count': mp.cpu_count(),
            'memory_usage_gb': psutil.virtual_memory().used / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'use_processes': self.config.use_processes
        }


class BatchProcessor:
    """
    Handles batch processing of large datasets with memory management.
    """
    
    def __init__(self, batch_size: int = 10000, memory_limit_gb: float = 4.0):
        self.batch_size = batch_size
        self.memory_limit_gb = memory_limit_gb
        self.parallel_processor = ParallelProcessor()
    
    def process_large_dataset(
        self,
        data_source: Union[str, pd.DataFrame],
        processing_func: Callable,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Process large dataset in batches to manage memory usage.
        
        Args:
            data_source: Path to data file or DataFrame
            processing_func: Function to apply to each batch
            output_path: Optional path to save results
            **kwargs: Additional arguments for processing function
            
        Returns:
            Processed DataFrame if output_path is None, otherwise None
        """
        results = []
        
        if isinstance(data_source, str):
            # Read data in chunks
            chunk_iter = pd.read_csv(data_source, chunksize=self.batch_size)
        else:
            # Split DataFrame into chunks
            chunk_iter = [
                data_source.iloc[i:i+self.batch_size] 
                for i in range(0, len(data_source), self.batch_size)
            ]
        
        for i, chunk in enumerate(chunk_iter):
            logger.info(f"Processing batch {i+1}")
            
            # Check memory usage
            memory_usage = psutil.virtual_memory().used / (1024**3)
            if memory_usage > self.memory_limit_gb:
                logger.warning(f"Memory usage ({memory_usage:.2f}GB) exceeds limit")
                # Force garbage collection
                import gc
                gc.collect()
            
            # Process chunk
            processed_chunk = processing_func(chunk, **kwargs)
            
            if output_path:
                # Save to file incrementally
                mode = 'w' if i == 0 else 'a'
                header = i == 0
                processed_chunk.to_csv(output_path, mode=mode, header=header, index=False)
            else:
                results.append(processed_chunk)
        
        if not output_path and results:
            return pd.concat(results, ignore_index=True)
        
        return None