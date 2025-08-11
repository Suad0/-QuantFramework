"""
Financial data-aware feature scaling and preprocessing.

This module provides specialized preprocessing techniques for financial data
that handle the unique characteristics of financial time series.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

from src.domain.exceptions import ValidationError


class FinancialScaler(BaseEstimator, TransformerMixin):
    """
    Financial data-aware scaler that handles the unique characteristics
    of financial time series data.
    """
    
    def __init__(
        self,
        method: str = 'robust',
        handle_outliers: bool = True,
        outlier_threshold: float = 3.0,
        winsorize: bool = False,
        winsorize_limits: Tuple[float, float] = (0.01, 0.01),
        feature_specific_scaling: Optional[Dict[str, str]] = None
    ):
        """
        Initialize financial scaler.
        
        Args:
            method: Scaling method ('standard', 'robust', 'minmax', 'returns')
            handle_outliers: Whether to handle outliers
            outlier_threshold: Threshold for outlier detection (in standard deviations)
            winsorize: Whether to winsorize extreme values
            winsorize_limits: Lower and upper limits for winsorization
            feature_specific_scaling: Dict mapping feature names to scaling methods
        """
        self.method = method
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.winsorize = winsorize
        self.winsorize_limits = winsorize_limits
        self.feature_specific_scaling = feature_specific_scaling or {}
        
        self.scalers_: Dict[str, BaseEstimator] = {}
        self.outlier_bounds_: Dict[str, Tuple[float, float]] = {}
        self.feature_stats_: Dict[str, Dict[str, float]] = {}
        
        if method not in ['standard', 'robust', 'minmax', 'returns']:
            raise ValidationError(f"Unknown scaling method: {method}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FinancialScaler':
        """
        Fit the scaler to the data.
        
        Args:
            X: Feature data
            y: Target data (ignored)
            
        Returns:
            Self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("X must be a pandas DataFrame")
        
        self.feature_names_ = X.columns.tolist()
        
        for column in X.columns:
            # Get scaling method for this feature
            scaling_method = self.feature_specific_scaling.get(column, self.method)
            
            # Extract column data
            col_data = X[column].dropna()
            
            if len(col_data) == 0:
                warnings.warn(f"Column {column} has no valid data")
                continue
            
            # Calculate feature statistics
            self.feature_stats_[column] = {
                'mean': col_data.mean(),
                'std': col_data.std(),
                'median': col_data.median(),
                'mad': (col_data - col_data.median()).abs().median(),
                'min': col_data.min(),
                'max': col_data.max(),
                'skew': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            }
            
            # Handle outliers if requested
            if self.handle_outliers:
                if scaling_method == 'robust':
                    # Use median and MAD for outlier detection
                    median = col_data.median()
                    mad = (col_data - median).abs().median()
                    lower_bound = median - self.outlier_threshold * mad
                    upper_bound = median + self.outlier_threshold * mad
                else:
                    # Use mean and std for outlier detection
                    mean = col_data.mean()
                    std = col_data.std()
                    lower_bound = mean - self.outlier_threshold * std
                    upper_bound = mean + self.outlier_threshold * std
                
                self.outlier_bounds_[column] = (lower_bound, upper_bound)
            
            # Create and fit appropriate scaler
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'returns':
                # For returns, use a custom approach
                scaler = ReturnsScaler()
            else:
                raise ValidationError(f"Unknown scaling method: {scaling_method}")
            
            # Prepare data for fitting
            fit_data = col_data.values.reshape(-1, 1)
            
            # Handle outliers before fitting
            if self.handle_outliers:
                lower_bound, upper_bound = self.outlier_bounds_[column]
                if self.winsorize:
                    # Winsorize extreme values
                    fit_data = np.clip(fit_data, 
                                     np.percentile(fit_data, self.winsorize_limits[0] * 100),
                                     np.percentile(fit_data, (1 - self.winsorize_limits[1]) * 100))
                else:
                    # Remove outliers
                    mask = (fit_data >= lower_bound) & (fit_data <= upper_bound)
                    fit_data = fit_data[mask.flatten()]
            
            if len(fit_data) > 0:
                scaler.fit(fit_data)
                self.scalers_[column] = scaler
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted scalers.
        
        Args:
            X: Feature data to transform
            
        Returns:
            Transformed data
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("X must be a pandas DataFrame")
        
        if not hasattr(self, 'scalers_'):
            raise ValidationError("Scaler has not been fitted yet")
        
        X_transformed = X.copy()
        
        for column in X.columns:
            if column not in self.scalers_:
                warnings.warn(f"Column {column} was not seen during fitting")
                continue
            
            col_data = X[column].values.reshape(-1, 1)
            
            # Handle outliers if configured
            if self.handle_outliers and column in self.outlier_bounds_:
                lower_bound, upper_bound = self.outlier_bounds_[column]
                
                if self.winsorize:
                    # Winsorize extreme values
                    col_data = np.clip(col_data, lower_bound, upper_bound)
                else:
                    # Set outliers to NaN
                    mask = (col_data < lower_bound) | (col_data > upper_bound)
                    col_data[mask] = np.nan
            
            # Transform using fitted scaler
            scaler = self.scalers_[column]
            
            # Handle NaN values
            nan_mask = np.isnan(col_data.flatten())
            if nan_mask.any():
                # Transform only non-NaN values
                valid_data = col_data[~nan_mask]
                if len(valid_data) > 0:
                    transformed_valid = scaler.transform(valid_data.reshape(-1, 1))
                    
                    # Reconstruct full array with NaNs
                    transformed_col = np.full(col_data.flatten().shape, np.nan)
                    transformed_col[~nan_mask] = transformed_valid.flatten()
                else:
                    transformed_col = col_data.flatten()
            else:
                transformed_col = scaler.transform(col_data).flatten()
            
            X_transformed[column] = transformed_col
        
        return X_transformed
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data.
        
        Args:
            X: Transformed data
            
        Returns:
            Original scale data
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("X must be a pandas DataFrame")
        
        if not hasattr(self, 'scalers_'):
            raise ValidationError("Scaler has not been fitted yet")
        
        X_inverse = X.copy()
        
        for column in X.columns:
            if column not in self.scalers_:
                continue
            
            col_data = X[column].values.reshape(-1, 1)
            scaler = self.scalers_[column]
            
            # Handle NaN values
            nan_mask = np.isnan(col_data.flatten())
            if nan_mask.any():
                valid_data = col_data[~nan_mask]
                if len(valid_data) > 0:
                    inverse_valid = scaler.inverse_transform(valid_data.reshape(-1, 1))
                    
                    inverse_col = np.full(col_data.flatten().shape, np.nan)
                    inverse_col[~nan_mask] = inverse_valid.flatten()
                else:
                    inverse_col = col_data.flatten()
            else:
                inverse_col = scaler.inverse_transform(col_data).flatten()
            
            X_inverse[column] = inverse_col
        
        return X_inverse
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Get feature statistics calculated during fitting."""
        return self.feature_stats_.copy()


class ReturnsScaler(BaseEstimator, TransformerMixin):
    """
    Specialized scaler for financial returns data.
    
    This scaler handles the unique properties of returns:
    - Typically centered around zero
    - Heavy tails and skewness
    - Volatility clustering
    """
    
    def __init__(self, method: str = 'standardize'):
        """
        Initialize returns scaler.
        
        Args:
            method: Scaling method ('standardize', 'normalize', 'rank')
        """
        self.method = method
        
        if method not in ['standardize', 'normalize', 'rank']:
            raise ValidationError(f"Unknown returns scaling method: {method}")
    
    def fit(self, X: np.ndarray, y=None) -> 'ReturnsScaler':
        """Fit the scaler to returns data."""
        X = np.asarray(X).flatten()
        X_clean = X[~np.isnan(X)]
        
        if self.method == 'standardize':
            self.mean_ = np.mean(X_clean)
            self.std_ = np.std(X_clean)
        elif self.method == 'normalize':
            self.min_ = np.min(X_clean)
            self.max_ = np.max(X_clean)
        elif self.method == 'rank':
            # For rank transformation, we don't need to store parameters
            pass
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform returns data."""
        X = np.asarray(X)
        original_shape = X.shape
        X_flat = X.flatten()
        
        if self.method == 'standardize':
            X_transformed = (X_flat - self.mean_) / self.std_
        elif self.method == 'normalize':
            X_transformed = (X_flat - self.min_) / (self.max_ - self.min_)
        elif self.method == 'rank':
            # Rank transformation
            X_transformed = pd.Series(X_flat).rank(pct=True).values
        
        return X_transformed.reshape(original_shape)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform returns data."""
        X = np.asarray(X)
        original_shape = X.shape
        X_flat = X.flatten()
        
        if self.method == 'standardize':
            X_inverse = X_flat * self.std_ + self.mean_
        elif self.method == 'normalize':
            X_inverse = X_flat * (self.max_ - self.min_) + self.min_
        elif self.method == 'rank':
            # Rank transformation is not easily invertible
            warnings.warn("Rank transformation is not invertible")
            X_inverse = X_flat
        
        return X_inverse.reshape(original_shape)


class FinancialPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive preprocessor for financial data.
    
    Combines multiple preprocessing steps specifically designed for financial data.
    """
    
    def __init__(
        self,
        handle_missing: str = 'forward_fill',
        scaling_method: str = 'robust',
        handle_outliers: bool = True,
        feature_engineering: bool = True,
        lag_features: Optional[List[int]] = None,
        rolling_features: Optional[List[int]] = None,
        difference_features: bool = False
    ):
        """
        Initialize financial preprocessor.
        
        Args:
            handle_missing: Method for handling missing values
            scaling_method: Scaling method to use
            handle_outliers: Whether to handle outliers
            feature_engineering: Whether to create additional features
            lag_features: List of lag periods for lag features
            rolling_features: List of window sizes for rolling features
            difference_features: Whether to create difference features
        """
        self.handle_missing = handle_missing
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.feature_engineering = feature_engineering
        self.lag_features = lag_features or []
        self.rolling_features = rolling_features or []
        self.difference_features = difference_features
        
        self.imputer_ = None
        self.scaler_ = None
        self.original_columns_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'FinancialPreprocessor':
        """
        Fit the preprocessor to the data.
        
        Args:
            X: Feature data
            y: Target data (ignored)
            
        Returns:
            Self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("X must be a pandas DataFrame")
        
        self.original_columns_ = X.columns.tolist()
        
        # Create engineered features if requested
        if self.feature_engineering:
            X_engineered = self._create_features(X)
        else:
            X_engineered = X.copy()
        
        # Fit imputer
        if self.handle_missing == 'simple':
            self.imputer_ = SimpleImputer(strategy='median')
            self.imputer_.fit(X_engineered)
        
        # Fit scaler
        self.scaler_ = FinancialScaler(
            method=self.scaling_method,
            handle_outliers=self.handle_outliers
        )
        
        # Handle missing values before scaling
        if self.handle_missing == 'forward_fill':
            X_filled = X_engineered.ffill().bfill()
        elif self.handle_missing == 'simple':
            X_filled = pd.DataFrame(
                self.imputer_.transform(X_engineered),
                columns=X_engineered.columns,
                index=X_engineered.index
            )
        else:
            X_filled = X_engineered.fillna(0)  # Fill remaining NaNs with 0
        
        self.scaler_.fit(X_filled)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted preprocessor.
        
        Args:
            X: Feature data to transform
            
        Returns:
            Transformed data
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("X must be a pandas DataFrame")
        
        if self.scaler_ is None:
            raise ValidationError("Preprocessor has not been fitted yet")
        
        # Create engineered features if requested
        if self.feature_engineering:
            X_engineered = self._create_features(X)
        else:
            X_engineered = X.copy()
        
        # Handle missing values
        if self.handle_missing == 'forward_fill':
            X_filled = X_engineered.ffill().bfill()
        elif self.handle_missing == 'simple' and self.imputer_ is not None:
            X_filled = pd.DataFrame(
                self.imputer_.transform(X_engineered),
                columns=X_engineered.columns,
                index=X_engineered.index
            )
        else:
            X_filled = X_engineered.fillna(0)  # Fill remaining NaNs with 0
        
        # Scale features
        X_scaled = self.scaler_.transform(X_filled)
        
        return X_scaled
    
    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for financial data."""
        X_features = X.copy()
        
        # Create lag features
        for col in self.original_columns_:
            if col in X.columns:
                for lag in self.lag_features:
                    X_features[f"{col}_lag_{lag}"] = X[col].shift(lag)
        
        # Create rolling features
        for col in self.original_columns_:
            if col in X.columns:
                for window in self.rolling_features:
                    X_features[f"{col}_rolling_mean_{window}"] = X[col].rolling(window).mean()
                    X_features[f"{col}_rolling_std_{window}"] = X[col].rolling(window).std()
        
        # Create difference features
        if self.difference_features:
            for col in self.original_columns_:
                if col in X.columns:
                    X_features[f"{col}_diff"] = X[col].diff()
                    X_features[f"{col}_pct_change"] = X[col].pct_change(fill_method=None)
        
        return X_features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features after transformation."""
        if self.scaler_ is None:
            raise ValidationError("Preprocessor has not been fitted yet")
        
        return self.scaler_.feature_names_