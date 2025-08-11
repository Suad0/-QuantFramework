"""
Time series cross-validation techniques for financial data.

This module provides specialized cross-validation methods that respect
the temporal nature of financial data and prevent look-ahead bias.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import BaseCrossValidator

from src.domain.exceptions import ValidationError


class TimeSeriesValidator(ABC):
    """Abstract base class for time series validation."""
    
    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits for time series data.
        
        Args:
            X: Feature data with datetime index
            y: Target data (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        pass
    
    @abstractmethod
    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, groups=None) -> int:
        """Get the number of splits."""
        pass


class WalkForwardValidator(TimeSeriesValidator, BaseCrossValidator):
    """
    Walk-forward cross-validation for time series data.
    
    This validator ensures that training data always precedes test data,
    preventing look-ahead bias common in financial modeling.
    """
    
    def __init__(
        self,
        train_size: Optional[int] = None,
        test_size: int = 1,
        gap: int = 0,
        expanding_window: bool = False,
        min_train_size: Optional[int] = None
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_size: Size of training window (None for expanding)
            test_size: Size of test window
            gap: Gap between train and test sets
            expanding_window: If True, use expanding window (ignore train_size)
            min_train_size: Minimum training size for expanding window
        """
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding_window = expanding_window
        self.min_train_size = min_train_size or 100
        
        if not expanding_window and train_size is None:
            raise ValidationError("train_size must be specified when not using expanding window")
        
        if test_size <= 0:
            raise ValidationError("test_size must be positive")
        
        if gap < 0:
            raise ValidationError("gap cannot be negative")
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate walk-forward splits.
        
        Args:
            X: Feature data with datetime index
            y: Target data (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValidationError("X must have a DatetimeIndex for time series validation")
        
        n_samples = len(X)
        
        if self.expanding_window:
            # Expanding window validation
            start_idx = self.min_train_size
            
            while start_idx + self.gap + self.test_size <= n_samples:
                train_end = start_idx
                test_start = train_end + self.gap
                test_end = test_start + self.test_size
                
                train_indices = np.arange(0, train_end)
                test_indices = np.arange(test_start, test_end)
                
                yield train_indices, test_indices
                
                start_idx += self.test_size
        else:
            # Fixed window validation
            start_idx = self.train_size
            
            while start_idx + self.gap + self.test_size <= n_samples:
                train_start = start_idx - self.train_size
                train_end = start_idx
                test_start = train_end + self.gap
                test_end = test_start + self.test_size
                
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                
                yield train_indices, test_indices
                
                start_idx += self.test_size
    
    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, groups=None) -> int:
        """Get the number of splits."""
        if X is None:
            return 0
        
        n_samples = len(X)
        
        if self.expanding_window:
            start_idx = self.min_train_size
            n_splits = 0
            
            while start_idx + self.gap + self.test_size <= n_samples:
                n_splits += 1
                start_idx += self.test_size
            
            return n_splits
        else:
            start_idx = self.train_size
            n_splits = 0
            
            while start_idx + self.gap + self.test_size <= n_samples:
                n_splits += 1
                start_idx += self.test_size
            
            return n_splits


class TimeSeriesSplit(TimeSeriesValidator, BaseCrossValidator):
    """
    Time series split with fixed number of folds.
    
    Similar to sklearn's TimeSeriesSplit but with additional financial-specific features.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """
        Initialize time series split.
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set (None for equal splits)
            gap: Gap between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        
        if n_splits <= 1:
            raise ValidationError("n_splits must be greater than 1")
        
        if gap < 0:
            raise ValidationError("gap cannot be negative")
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time series splits.
        
        Args:
            X: Feature data with datetime index
            y: Target data (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValidationError("X must have a DatetimeIndex for time series validation")
        
        n_samples = len(X)
        
        if self.test_size is None:
            # Equal splits
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size
        
        for i in range(self.n_splits):
            # Calculate split points
            test_start = (i + 1) * (n_samples // (self.n_splits + 1))
            test_end = test_start + test_size
            train_end = test_start - self.gap
            
            if test_end > n_samples:
                test_end = n_samples
            
            if train_end <= 0:
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, groups=None) -> int:
        """Get the number of splits."""
        return self.n_splits


class PurgedGroupTimeSeriesSplit(TimeSeriesValidator, BaseCrossValidator):
    """
    Purged group time series split for financial data.
    
    This validator handles overlapping samples and ensures proper purging
    to prevent data leakage in financial applications.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        group_gap: timedelta = timedelta(days=1),
        purge_gap: timedelta = timedelta(hours=1)
    ):
        """
        Initialize purged group time series split.
        
        Args:
            n_splits: Number of splits
            group_gap: Gap between groups
            purge_gap: Purging gap to prevent leakage
        """
        self.n_splits = n_splits
        self.group_gap = group_gap
        self.purge_gap = purge_gap
        
        if n_splits <= 1:
            raise ValidationError("n_splits must be greater than 1")
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged group splits.
        
        Args:
            X: Feature data with datetime index
            y: Target data (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValidationError("X must have a DatetimeIndex for time series validation")
        
        # Create time-based groups
        times = X.index
        n_samples = len(X)
        
        # Calculate split points based on time
        time_range = times.max() - times.min()
        split_duration = time_range / (self.n_splits + 1)
        
        for i in range(self.n_splits):
            # Calculate time boundaries
            test_start_time = times.min() + (i + 1) * split_duration
            test_end_time = test_start_time + split_duration
            
            # Find indices for test set
            test_mask = (times >= test_start_time) & (times < test_end_time)
            test_indices = np.where(test_mask)[0]
            
            if len(test_indices) == 0:
                continue
            
            # Find train indices (before test period with purging)
            train_end_time = test_start_time - self.purge_gap
            train_mask = times < train_end_time
            train_indices = np.where(train_mask)[0]
            
            if len(train_indices) == 0:
                continue
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, groups=None) -> int:
        """Get the number of splits."""
        return self.n_splits


class BlockingTimeSeriesSplit(TimeSeriesValidator, BaseCrossValidator):
    """
    Blocking time series split for handling autocorrelation.
    
    This validator creates blocks of consecutive samples to reduce
    the impact of autocorrelation in financial time series.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        block_size: int = 10,
        gap: int = 1
    ):
        """
        Initialize blocking time series split.
        
        Args:
            n_splits: Number of splits
            block_size: Size of each block
            gap: Gap between blocks
        """
        self.n_splits = n_splits
        self.block_size = block_size
        self.gap = gap
        
        if n_splits <= 1:
            raise ValidationError("n_splits must be greater than 1")
        
        if block_size <= 0:
            raise ValidationError("block_size must be positive")
        
        if gap < 0:
            raise ValidationError("gap cannot be negative")
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate blocking splits.
        
        Args:
            X: Feature data with datetime index
            y: Target data (optional)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        # Calculate total space needed for each split
        total_block_space = self.block_size + self.gap
        
        for i in range(self.n_splits):
            # Calculate test block position
            test_start = (i + 1) * (n_samples // (self.n_splits + 1))
            test_end = min(test_start + self.block_size, n_samples)
            
            # Calculate train indices (all data before test block with gap)
            train_end = test_start - self.gap
            
            if train_end <= 0 or test_start >= n_samples:
                continue
            
            # Create blocked train indices
            train_indices = []
            for j in range(0, train_end, total_block_space):
                block_end = min(j + self.block_size, train_end)
                train_indices.extend(range(j, block_end))
            
            test_indices = list(range(test_start, test_end))
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, groups=None) -> int:
        """Get the number of splits."""
        return self.n_splits