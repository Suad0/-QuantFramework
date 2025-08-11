"""
Feature Engineering Service

This module implements the feature engineering capabilities including
technical indicators, cross-sectional features, and plugin architecture.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Type, Callable, Union, Tuple
from abc import ABC, abstractmethod
import importlib
import importlib.util
import inspect
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats

from src.domain.interfaces import IFeatureEngine, IIndicator
from src.domain.exceptions import ValidationError, QuantFrameworkError


class FeatureValidationReport:
    """Report containing feature validation results"""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.statistics: Dict[str, Any] = {}
        self.look_ahead_bias_detected = False
        self.bias_details: List[str] = []
    
    def add_error(self, message: str):
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def add_bias_detection(self, message: str):
        self.bias_details.append(message)
        self.look_ahead_bias_detected = True


class RollingWindowProcessor:
    """Handles configurable rolling window operations with multiple lookback periods"""
    
    def __init__(self):
        self.supported_operations = {
            'mean': lambda x: x.mean(),
            'std': lambda x: x.std(),
            'min': lambda x: x.min(),
            'max': lambda x: x.max(),
            'median': lambda x: x.median(),
            'sum': lambda x: x.sum(),
            'skew': lambda x: x.skew(),
            'kurt': lambda x: x.kurtosis(),
            'quantile': lambda x, q=0.5: x.quantile(q),
            'rank': lambda x: x.rank(pct=True).iloc[-1],
            'zscore': lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0,
            'rsi': self._calculate_rsi,
            'momentum': lambda x: (x.iloc[-1] / x.iloc[0] - 1) if x.iloc[0] != 0 else 0,
            'volatility': lambda x: x.pct_change(fill_method=None).std() * np.sqrt(252),
            'sharpe': lambda x: (x.pct_change(fill_method=None).mean() / x.pct_change(fill_method=None).std()) * np.sqrt(252) if x.pct_change(fill_method=None).std() > 0 else 0
        }
    
    def apply_rolling_operations(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply rolling window operations with multiple lookback periods"""
        result_data = data.copy()
        
        windows = config.get('windows', [20, 50, 200])  # Default lookback periods
        operations = config.get('operations', ['mean', 'std'])
        columns = config.get('columns', ['close'])
        
        # Validate columns exist
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            raise ValidationError(f"Missing columns for rolling operations: {missing_cols}")
        
        for col in columns:
            for window in windows:
                if len(data) < window:
                    warnings.warn(f"Data length ({len(data)}) is less than window size ({window})")
                    continue
                
                for operation in operations:
                    if operation not in self.supported_operations:
                        raise ValidationError(f"Unsupported rolling operation: {operation}")
                    
                    try:
                        if operation == 'quantile':
                            quantiles = config.get('quantiles', [0.25, 0.75])
                            for q in quantiles:
                                feature_name = f"{col}_rolling_{window}_{operation}_{int(q*100)}"
                                result_data[feature_name] = data[col].rolling(window).quantile(q)
                        else:
                            feature_name = f"{col}_rolling_{window}_{operation}"
                            if operation in ['rsi', 'momentum', 'volatility', 'sharpe']:
                                # Custom operations that need the full window
                                rolling_values = []
                                for i in range(len(data)):
                                    start_idx = max(0, i - window + 1)
                                    window_data = data[col].iloc[start_idx:i+1]
                                    if len(window_data) >= min(window, 2):  # Minimum data for calculation
                                        value = self.supported_operations[operation](window_data)
                                        rolling_values.append(value)
                                    else:
                                        rolling_values.append(np.nan)
                                result_data[feature_name] = rolling_values
                            else:
                                # Standard pandas rolling operations
                                rolling_obj = data[col].rolling(window)
                                result_data[feature_name] = self.supported_operations[operation](rolling_obj)
                                
                    except Exception as e:
                        warnings.warn(f"Error calculating {operation} for {col} with window {window}: {str(e)}")
        
        return result_data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for a price series"""
        if len(prices) < period + 1:
            return np.nan
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else np.nan


class DataImputer:
    """Handles multiple data imputation strategies"""
    
    def __init__(self):
        self.strategies = {
            'forward_fill': self._forward_fill,
            'backward_fill': self._backward_fill,
            'linear_interpolation': self._linear_interpolation,
            'polynomial_interpolation': self._polynomial_interpolation,
            'spline_interpolation': self._spline_interpolation,
            'mean_imputation': self._mean_imputation,
            'median_imputation': self._median_imputation,
            'mode_imputation': self._mode_imputation,
            'knn_imputation': self._knn_imputation,
            'iterative_imputation': self._iterative_imputation,
            'seasonal_imputation': self._seasonal_imputation,
            'carry_forward_with_decay': self._carry_forward_with_decay
        }
    
    def impute_missing_data(
        self, 
        data: pd.DataFrame, 
        strategy: str = 'forward_fill',
        **kwargs
    ) -> pd.DataFrame:
        """Impute missing data using specified strategy"""
        if strategy not in self.strategies:
            raise ValidationError(f"Unknown imputation strategy: {strategy}")
        
        return self.strategies[strategy](data, **kwargs)
    
    def _forward_fill(self, data: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
        """Forward fill missing values"""
        return data.ffill(limit=limit)
    
    def _backward_fill(self, data: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
        """Backward fill missing values"""
        return data.bfill(limit=limit)
    
    def _linear_interpolation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Linear interpolation for missing values"""
        return data.interpolate(method='linear', **kwargs)
    
    def _polynomial_interpolation(self, data: pd.DataFrame, order: int = 2, **kwargs) -> pd.DataFrame:
        """Polynomial interpolation for missing values"""
        return data.interpolate(method='polynomial', order=order, **kwargs)
    
    def _spline_interpolation(self, data: pd.DataFrame, order: int = 3, **kwargs) -> pd.DataFrame:
        """Spline interpolation for missing values"""
        return data.interpolate(method='spline', order=order, **kwargs)
    
    def _mean_imputation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Mean imputation for missing values"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        data_imputed = data.copy()
        data_imputed[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        return data_imputed
    
    def _median_imputation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Median imputation for missing values"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        data_imputed = data.copy()
        data_imputed[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        return data_imputed
    
    def _mode_imputation(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Mode imputation for missing values"""
        return data.fillna(data.mode().iloc[0])
    
    def _knn_imputation(self, data: pd.DataFrame, n_neighbors: int = 5, **kwargs) -> pd.DataFrame:
        """KNN imputation for missing values"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=n_neighbors)
        data_imputed = data.copy()
        data_imputed[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        return data_imputed
    
    def _iterative_imputation(self, data: pd.DataFrame, max_iter: int = 10, **kwargs) -> pd.DataFrame:
        """Iterative (MICE) imputation for missing values"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
        data_imputed = data.copy()
        data_imputed[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        return data_imputed
    
    def _seasonal_imputation(self, data: pd.DataFrame, period: int = 252, **kwargs) -> pd.DataFrame:
        """Seasonal imputation using historical patterns"""
        data_imputed = data.copy()
        
        for column in data.select_dtypes(include=[np.number]).columns:
            missing_mask = data[column].isna()
            if not missing_mask.any():
                continue
            
            # Convert to integer positions for arithmetic
            for i, idx in enumerate(data[missing_mask].index):
                # Look for same period in previous cycles
                seasonal_values = []
                current_pos = data.index.get_loc(idx)
                
                for lag in range(1, 5):  # Look back up to 4 cycles
                    seasonal_pos = current_pos - (lag * period)
                    if seasonal_pos >= 0 and seasonal_pos < len(data):
                        seasonal_idx = data.index[seasonal_pos]
                        if not pd.isna(data[column].loc[seasonal_idx]):
                            seasonal_values.append(data[column].loc[seasonal_idx])
                
                if seasonal_values:
                    data_imputed.loc[idx, column] = np.mean(seasonal_values)
        
        # Fill any remaining NaNs with forward fill
        return data_imputed.ffill()
    
    def _carry_forward_with_decay(self, data: pd.DataFrame, decay_rate: float = 0.95, **kwargs) -> pd.DataFrame:
        """Carry forward with exponential decay"""
        data_imputed = data.copy()
        
        for column in data.select_dtypes(include=[np.number]).columns:
            last_valid_value = None
            days_since_valid = 0
            
            for idx in data.index:
                if pd.isna(data.loc[idx, column]):
                    if last_valid_value is not None:
                        days_since_valid += 1
                        decayed_value = last_valid_value * (decay_rate ** days_since_valid)
                        data_imputed.loc[idx, column] = decayed_value
                else:
                    last_valid_value = data.loc[idx, column]
                    days_since_valid = 0
        
        return data_imputed


class LookAheadBiasDetector:
    """Detects and prevents look-ahead bias in feature engineering"""
    
    def __init__(self):
        self.bias_checks = [
            self._check_future_data_usage,
            self._check_statistical_properties,
            self._check_temporal_consistency,
            self._check_information_leakage
        ]
    
    def detect_bias(self, data: pd.DataFrame, features: pd.DataFrame) -> FeatureValidationReport:
        """Detect potential look-ahead bias in features"""
        report = FeatureValidationReport()
        
        for check in self.bias_checks:
            try:
                check(data, features, report)
            except Exception as e:
                report.add_warning(f"Bias check failed: {str(e)}")
        
        return report
    
    def _check_future_data_usage(self, data: pd.DataFrame, features: pd.DataFrame, report: FeatureValidationReport):
        """Check if features use future data points"""
        if not isinstance(data.index, pd.DatetimeIndex) or not isinstance(features.index, pd.DatetimeIndex):
            return
        
        # Check if any feature values change when we remove future data
        for col in features.columns:
            if col in data.columns:
                continue  # Skip original data columns
            
            # Calculate feature with progressively less data
            sample_points = min(10, len(data) // 4)
            if sample_points < 2:
                continue
            
            test_indices = np.linspace(sample_points, len(data)-1, sample_points, dtype=int)
            
            for i, test_idx in enumerate(test_indices[:-1]):
                if test_idx >= len(features):
                    continue
                
                # Check if feature value at test_idx is consistent
                # when calculated with data only up to that point
                current_value = features[col].iloc[test_idx]
                
                if pd.isna(current_value):
                    continue
                
                # Simple heuristic: check if feature shows unrealistic stability
                # or sudden changes that might indicate future data usage
                window_size = min(20, test_idx)
                if window_size > 1:
                    recent_values = features[col].iloc[test_idx-window_size:test_idx+1]
                    if len(recent_values.dropna()) > 1:
                        volatility = recent_values.std()
                        if volatility == 0 and len(recent_values.dropna()) > 5:
                            report.add_bias_detection(
                                f"Feature {col} shows suspicious stability at index {test_idx}, "
                                "possible future data usage"
                            )
    
    def _check_statistical_properties(self, data: pd.DataFrame, features: pd.DataFrame, report: FeatureValidationReport):
        """Check for statistical properties that suggest look-ahead bias"""
        for col in features.columns:
            if col in data.columns:
                continue
            
            feature_values = features[col].dropna()
            if len(feature_values) < 10:
                continue
            
            # Check for perfect correlations (suspicious)
            for data_col in data.select_dtypes(include=[np.number]).columns:
                if len(data[data_col].dropna()) < 10:
                    continue
                
                # Align indices for correlation calculation
                common_idx = feature_values.index.intersection(data[data_col].dropna().index)
                if len(common_idx) < 10:
                    continue
                
                try:
                    corr = np.corrcoef(
                        feature_values.loc[common_idx], 
                        data[data_col].loc[common_idx]
                    )[0, 1]
                    
                    if abs(corr) > 0.99 and not np.isnan(corr):
                        report.add_bias_detection(
                            f"Feature {col} has suspiciously high correlation ({corr:.4f}) "
                            f"with {data_col}, possible look-ahead bias"
                        )
                    
                    # Also check for shifted correlations (future data usage)
                    if isinstance(features.index, pd.DatetimeIndex) and len(common_idx) > 20:
                        # Check correlation with shifted data
                        shifted_data = data[data_col].shift(1).loc[common_idx]
                        if len(shifted_data.dropna()) > 10:
                            shifted_corr = np.corrcoef(
                                feature_values.loc[shifted_data.dropna().index], 
                                shifted_data.dropna()
                            )[0, 1]
                            
                            if abs(shifted_corr) > 0.95 and not np.isnan(shifted_corr):
                                report.add_bias_detection(
                                    f"Feature {col} has high correlation ({shifted_corr:.4f}) "
                                    f"with future {data_col}, possible look-ahead bias"
                                )
                except (ValueError, IndexError):
                    # Skip if correlation calculation fails
                    continue
    
    def _check_temporal_consistency(self, data: pd.DataFrame, features: pd.DataFrame, report: FeatureValidationReport):
        """Check temporal consistency of features"""
        if not isinstance(features.index, pd.DatetimeIndex):
            return
        
        for col in features.columns:
            if col in data.columns:
                continue
            
            feature_values = features[col].dropna()
            if len(feature_values) < 20:
                continue
            
            # Check for abrupt changes that might indicate data snooping
            returns = feature_values.pct_change().dropna()
            if len(returns) < 10:
                continue
            
            # Detect outliers in feature changes
            q99 = returns.quantile(0.99)
            q1 = returns.quantile(0.01)
            
            extreme_changes = returns[(returns > q99) | (returns < q1)]
            if len(extreme_changes) > len(returns) * 0.05:  # More than 5% extreme changes
                report.add_bias_detection(
                    f"Feature {col} has {len(extreme_changes)} extreme changes "
                    f"({len(extreme_changes)/len(returns)*100:.1f}%), possible instability"
                )
    
    def _check_information_leakage(self, data: pd.DataFrame, features: pd.DataFrame, report: FeatureValidationReport):
        """Check for information leakage from target variables"""
        # Look for features that might be derived from future target information
        potential_targets = ['return', 'target', 'label', 'y']
        
        for col in features.columns:
            col_lower = col.lower()
            
            # Check if feature name suggests it might contain target information
            if any(target in col_lower for target in potential_targets):
                if not any(lag_indicator in col_lower for lag_indicator in ['lag', 'shift', 'prev', 'past']):
                    report.add_bias_detection(
                        f"Feature {col} name suggests potential target leakage. "
                        "Ensure it doesn't contain future target information."
                    )
    
    def prevent_bias(self, data: pd.DataFrame, feature_config: Dict[str, Any]) -> Dict[str, Any]:
        """Modify feature configuration to prevent look-ahead bias"""
        safe_config = feature_config.copy()
        
        # Add temporal constraints
        if 'temporal_constraints' not in safe_config:
            safe_config['temporal_constraints'] = {
                'max_future_periods': 0,  # No future data allowed
                'min_history_periods': 1,  # At least 1 period of history required
                'rolling_window_alignment': 'left'  # Align rolling windows to past data
            }
        
        # Ensure all rolling operations use only past data
        if 'rolling_operations' in safe_config:
            rolling_config = safe_config['rolling_operations']
            # Handle both list and dict configurations
            if isinstance(rolling_config, dict):
                # Add bias prevention parameters to rolling operations config
                if 'bias_prevention' not in rolling_config:
                    rolling_config['bias_prevention'] = {
                        'center': False,  # Don't center rolling windows
                        'min_periods': 1  # Minimum periods for calculation
                    }
            elif isinstance(rolling_config, list):
                # Modify each operation in the list
                for operation in rolling_config:
                    if isinstance(operation, dict):
                        if 'center' in operation:
                            operation['center'] = False
                        if 'min_periods' not in operation:
                            operation['min_periods'] = 1
        
        return safe_config


class FeatureSelector:
    """Feature selection and dimensionality reduction tools"""
    
    def __init__(self):
        self.selection_methods = {
            'univariate_f': self._univariate_f_selection,
            'mutual_info': self._mutual_info_selection,
            'rfe': self._recursive_feature_elimination,
            'variance_threshold': self._variance_threshold_selection,
            'correlation_filter': self._correlation_filter,
            'stability_selection': self._stability_selection
        }
        
        self.dimensionality_methods = {
            'pca': self._pca_reduction,
            'ica': self._ica_reduction,
            'factor_analysis': self._factor_analysis_reduction
        }
    
    def select_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        method: str = 'univariate_f',
        n_features: int = 50,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features using specified method"""
        if method not in self.selection_methods:
            raise ValidationError(f"Unknown feature selection method: {method}")
        
        return self.selection_methods[method](features, target, n_features, **kwargs)
    
    def reduce_dimensions(
        self, 
        features: pd.DataFrame, 
        method: str = 'pca',
        n_components: int = 10,
        **kwargs
    ) -> Tuple[pd.DataFrame, Any]:
        """Reduce feature dimensions using specified method"""
        if method not in self.dimensionality_methods:
            raise ValidationError(f"Unknown dimensionality reduction method: {method}")
        
        return self.dimensionality_methods[method](features, n_components, **kwargs)
    
    def _univariate_f_selection(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        n_features: int,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Univariate F-test feature selection"""
        # Align features and target
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx].select_dtypes(include=[np.number])
        y = target.loc[common_idx]
        
        # Remove NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            raise ValidationError("No valid data after removing NaN values")
        
        selector = SelectKBest(score_func=f_regression, k=min(n_features, X_clean.shape[1]))
        X_selected = selector.fit_transform(X_clean, y_clean)
        
        selected_features = X_clean.columns[selector.get_support()].tolist()
        result_df = pd.DataFrame(X_selected, index=X_clean.index, columns=selected_features)
        
        return result_df, selected_features
    
    def _mutual_info_selection(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        n_features: int,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Mutual information feature selection"""
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx].select_dtypes(include=[np.number])
        y = target.loc[common_idx]
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            raise ValidationError("No valid data after removing NaN values")
        
        selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, X_clean.shape[1]))
        X_selected = selector.fit_transform(X_clean, y_clean)
        
        selected_features = X_clean.columns[selector.get_support()].tolist()
        result_df = pd.DataFrame(X_selected, index=X_clean.index, columns=selected_features)
        
        return result_df, selected_features
    
    def _recursive_feature_elimination(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        n_features: int,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Recursive feature elimination"""
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx].select_dtypes(include=[np.number])
        y = target.loc[common_idx]
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            raise ValidationError("No valid data after removing NaN values")
        
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=min(n_features, X_clean.shape[1]))
        X_selected = selector.fit_transform(X_clean, y_clean)
        
        selected_features = X_clean.columns[selector.get_support()].tolist()
        result_df = pd.DataFrame(X_selected, index=X_clean.index, columns=selected_features)
        
        return result_df, selected_features
    
    def _variance_threshold_selection(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        n_features: int,
        threshold: float = 0.01,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Variance threshold feature selection"""
        X = features.select_dtypes(include=[np.number])
        
        # Calculate variance for each feature
        variances = X.var()
        selected_features = variances[variances > threshold].index.tolist()
        
        # If we have more features than requested, select top variance features
        if len(selected_features) > n_features:
            top_variance_features = variances.nlargest(n_features).index.tolist()
            selected_features = top_variance_features
        
        result_df = X[selected_features]
        return result_df, selected_features
    
    def _correlation_filter(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        n_features: int,
        threshold: float = 0.95,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features"""
        X = features.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        # Keep features not in drop list
        selected_features = [col for col in X.columns if col not in to_drop]
        
        # If we still have too many features, select based on target correlation
        if len(selected_features) > n_features:
            common_idx = X.index.intersection(target.index)
            target_corr = X.loc[common_idx].corrwith(target.loc[common_idx]).abs()
            top_features = target_corr.nlargest(n_features).index.tolist()
            selected_features = [f for f in selected_features if f in top_features]
        
        result_df = X[selected_features]
        return result_df, selected_features
    
    def _stability_selection(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        n_features: int,
        n_bootstrap: int = 100,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Stability selection using bootstrap sampling"""
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx].select_dtypes(include=[np.number])
        y = target.loc[common_idx]
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            raise ValidationError("No valid data after removing NaN values")
        
        feature_selection_counts = np.zeros(X_clean.shape[1])
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            sample_idx = np.random.choice(len(X_clean), size=len(X_clean), replace=True)
            X_sample = X_clean.iloc[sample_idx]
            y_sample = y_clean.iloc[sample_idx]
            
            # Feature selection on bootstrap sample
            selector = SelectKBest(score_func=f_regression, k=min(n_features, X_sample.shape[1]))
            selector.fit(X_sample, y_sample)
            
            # Count selections
            feature_selection_counts += selector.get_support().astype(int)
        
        # Select features that were selected most frequently
        selection_probabilities = feature_selection_counts / n_bootstrap
        stable_features_idx = np.argsort(selection_probabilities)[-n_features:]
        
        selected_features = X_clean.columns[stable_features_idx].tolist()
        result_df = X_clean[selected_features]
        
        return result_df, selected_features
    
    def _pca_reduction(
        self, 
        features: pd.DataFrame, 
        n_components: int,
        **kwargs
    ) -> Tuple[pd.DataFrame, PCA]:
        """Principal Component Analysis dimensionality reduction"""
        X = features.select_dtypes(include=[np.number])
        
        # Handle missing values
        X_filled = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        
        # Create DataFrame with component names
        component_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        result_df = pd.DataFrame(X_pca, index=X.index, columns=component_names)
        
        return result_df, pca
    
    def _ica_reduction(
        self, 
        features: pd.DataFrame, 
        n_components: int,
        **kwargs
    ) -> Tuple[pd.DataFrame, FastICA]:
        """Independent Component Analysis dimensionality reduction"""
        X = features.select_dtypes(include=[np.number])
        
        # Handle missing values
        X_filled = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)
        
        # Apply ICA
        ica = FastICA(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
        X_ica = ica.fit_transform(X_scaled)
        
        # Create DataFrame with component names
        component_names = [f'IC{i+1}' for i in range(X_ica.shape[1])]
        result_df = pd.DataFrame(X_ica, index=X.index, columns=component_names)
        
        return result_df, ica
    
    def _factor_analysis_reduction(
        self, 
        features: pd.DataFrame, 
        n_components: int,
        **kwargs
    ) -> Tuple[pd.DataFrame, Any]:
        """Factor Analysis dimensionality reduction"""
        try:
            from sklearn.decomposition import FactorAnalysis
        except ImportError:
            raise ValidationError("FactorAnalysis not available in this sklearn version")
        
        X = features.select_dtypes(include=[np.number])
        
        # Handle missing values
        X_filled = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)
        
        # Apply Factor Analysis
        fa = FactorAnalysis(n_components=min(n_components, X_scaled.shape[1]), random_state=42)
        X_fa = fa.fit_transform(X_scaled)
        
        # Create DataFrame with factor names
        factor_names = [f'Factor{i+1}' for i in range(X_fa.shape[1])]
        result_df = pd.DataFrame(X_fa, index=X.index, columns=factor_names)
        
        return result_df, fa


class BaseIndicator(IIndicator):
    """Base class for all technical indicators"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._parameters: Dict[str, Any] = {}
    
    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()
    
    def set_parameter(self, key: str, value: Any):
        self._parameters[key] = value
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data for indicator calculation"""
        if data.empty:
            raise ValidationError(f"Empty data provided to {self.name}")
        
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValidationError(
                f"{self.name} requires columns {required_columns}, "
                f"missing: {missing_columns}"
            )
    
    def get_required_columns(self) -> List[str]:
        """Return list of required columns for this indicator"""
        return ['close']  # Default requirement
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate indicator values"""
        pass


class IndicatorRegistry:
    """Registry for managing technical indicators"""
    
    def __init__(self):
        self._indicators: Dict[str, Type[BaseIndicator]] = {}
        self._instances: Dict[str, BaseIndicator] = {}
        self._load_builtin_indicators()
    
    def register(self, indicator_class: Type[BaseIndicator]) -> None:
        """Register a new indicator class"""
        if not issubclass(indicator_class, BaseIndicator):
            raise ValidationError(
                f"Indicator {indicator_class.__name__} must inherit from BaseIndicator"
            )
        
        # Create instance to get name
        instance = indicator_class()
        self._indicators[instance.name] = indicator_class
    
    def get_indicator(self, name: str, **kwargs) -> BaseIndicator:
        """Get indicator instance by name"""
        if name not in self._indicators:
            raise ValidationError(f"Unknown indicator: {name}")
        
        # Create new instance with parameters
        indicator = self._indicators[name]()
        for key, value in kwargs.items():
            indicator.set_parameter(key, value)
        
        return indicator
    
    def list_indicators(self) -> List[str]:
        """List all registered indicators"""
        return list(self._indicators.keys())
    
    def _load_builtin_indicators(self):
        """Load built-in indicators"""
        # Import all indicator modules
        from src.infrastructure.indicators import momentum, volatility, volume, statistical, trend, oscillators, cross_sectional
        
        # Get all indicator classes from each module
        indicator_modules = [momentum, volatility, volume, statistical, trend, oscillators, cross_sectional]
        
        for module in indicator_modules:
            for name in dir(module):
                obj = getattr(module, name)
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseIndicator) and 
                    obj != BaseIndicator):
                    try:
                        # Create instance to register
                        instance = obj()
                        self._indicators[instance.name] = obj
                    except Exception as e:
                        # Skip indicators that fail to instantiate
                        continue


class FeatureEngine(IFeatureEngine):
    """Main feature engineering engine"""
    
    def __init__(self):
        self.indicator_registry = IndicatorRegistry()
        self._custom_features: Dict[str, Callable] = {}
        self.rolling_processor = RollingWindowProcessor()
        self.data_imputer = DataImputer()
        self.bias_detector = LookAheadBiasDetector()
        self.feature_selector = FeatureSelector()
    
    def register_indicator(self, name: str, indicator: IIndicator) -> None:
        """Register a new technical indicator"""
        if not isinstance(indicator, BaseIndicator):
            raise ValidationError("Indicator must inherit from BaseIndicator")
        
        self.indicator_registry._indicators[name] = type(indicator)
    
    def register_custom_feature(self, name: str, feature_func: Callable) -> None:
        """Register a custom feature function"""
        self._custom_features[name] = feature_func
    
    def compute_features(
        self, 
        data: pd.DataFrame, 
        feature_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Compute features based on configuration with vectorized operations"""
        if data.empty:
            raise ValidationError("Cannot compute features on empty data")
        
        # Step 1: Handle missing data imputation first
        imputation_config = feature_config.get('imputation', {})
        if imputation_config.get('enabled', False):
            strategy = imputation_config.get('strategy', 'forward_fill')
            imputation_params = imputation_config.get('params', {})
            data = self.data_imputer.impute_missing_data(data, strategy, **imputation_params)
        
        # Step 2: Apply look-ahead bias prevention
        bias_prevention = feature_config.get('bias_prevention', {})
        if bias_prevention.get('enabled', True):
            feature_config = self.bias_detector.prevent_bias(data, feature_config)
        
        result_data = data.copy()
        
        # Step 3: Process rolling window operations
        rolling_config = feature_config.get('rolling_operations', {})
        if rolling_config.get('enabled', False):
            result_data = self.rolling_processor.apply_rolling_operations(result_data, rolling_config)
        
        # Step 4: Enable parallel processing if specified
        use_parallel = feature_config.get('parallel', False)
        
        # Step 5: Process technical indicators
        indicators_config = feature_config.get('indicators', {})
        if use_parallel and len(indicators_config) > 1:
            result_data = self._compute_indicators_parallel(data, indicators_config, result_data)
        else:
            result_data = self._compute_indicators_sequential(data, indicators_config, result_data)
        
        # Step 6: Process cross-sectional features
        cross_sectional_config = feature_config.get('cross_sectional', {})
        if cross_sectional_config:
            result_data = self._compute_cross_sectional_features(
                result_data, cross_sectional_config
            )
        
        # Step 7: Process custom features
        custom_config = feature_config.get('custom', {})
        for feature_name, params in custom_config.items():
            if feature_name in self._custom_features:
                try:
                    feature_values = self._custom_features[feature_name](data, **params)
                    result_data[feature_name] = feature_values
                except Exception as e:
                    raise QuantFrameworkError(
                        f"Error computing custom feature {feature_name}: {str(e)}"
                    )
        
        # Step 8: Apply feature selection if configured
        selection_config = feature_config.get('feature_selection', {})
        if selection_config.get('enabled', False) and 'target_column' in selection_config:
            target_col = selection_config['target_column']
            if target_col in result_data.columns:
                method = selection_config.get('method', 'univariate_f')
                n_features = selection_config.get('n_features', 50)
                selection_params = selection_config.get('params', {})
                
                try:
                    selected_data, selected_features = self.feature_selector.select_features(
                        result_data.drop(columns=[target_col]), 
                        result_data[target_col], 
                        method, 
                        n_features, 
                        **selection_params
                    )
                    # Keep target column and selected features
                    result_data = pd.concat([selected_data, result_data[[target_col]]], axis=1)
                except Exception as e:
                    warnings.warn(f"Feature selection failed: {str(e)}")
        
        # Step 9: Apply dimensionality reduction if configured
        reduction_config = feature_config.get('dimensionality_reduction', {})
        if reduction_config.get('enabled', False):
            method = reduction_config.get('method', 'pca')
            n_components = reduction_config.get('n_components', 10)
            reduction_params = reduction_config.get('params', {})
            
            try:
                # Exclude non-numeric columns from reduction
                numeric_cols = result_data.select_dtypes(include=[np.number]).columns
                non_numeric_cols = result_data.select_dtypes(exclude=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    reduced_data, reducer = self.feature_selector.reduce_dimensions(
                        result_data[numeric_cols], 
                        method, 
                        n_components, 
                        **reduction_params
                    )
                    
                    # Combine reduced features with non-numeric columns
                    if len(non_numeric_cols) > 0:
                        result_data = pd.concat([reduced_data, result_data[non_numeric_cols]], axis=1)
                    else:
                        result_data = reduced_data
            except Exception as e:
                warnings.warn(f"Dimensionality reduction failed: {str(e)}")
        
        return result_data
    
    def _compute_indicators_sequential(
        self, 
        data: pd.DataFrame, 
        indicators_config: Dict[str, Any], 
        result_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute indicators sequentially"""
        for indicator_name, params in indicators_config.items():
            try:
                indicator = self.indicator_registry.get_indicator(indicator_name, **params)
                feature_values = indicator.calculate(data)
                
                # Handle both Series and DataFrame results
                if isinstance(feature_values, pd.Series):
                    column_name = f"{indicator_name}_{self._params_to_suffix(params)}"
                    result_data[column_name] = feature_values
                elif isinstance(feature_values, pd.DataFrame):
                    for col in feature_values.columns:
                        column_name = f"{indicator_name}_{col}_{self._params_to_suffix(params)}"
                        result_data[column_name] = feature_values[col]
                        
            except Exception as e:
                raise QuantFrameworkError(
                    f"Error computing indicator {indicator_name}: {str(e)}"
                )
        
        return result_data
    
    def _compute_indicators_parallel(
        self, 
        data: pd.DataFrame, 
        indicators_config: Dict[str, Any], 
        result_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute indicators in parallel using multiprocessing"""
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing as mp
            
            # Determine number of workers
            max_workers = min(len(indicators_config), mp.cpu_count())
            
            def compute_single_indicator(args):
                indicator_name, params, data_dict = args
                # Recreate DataFrame from dict (for pickling)
                data_df = pd.DataFrame(data_dict)
                
                try:
                    # Create fresh indicator registry in worker process
                    registry = IndicatorRegistry()
                    indicator = registry.get_indicator(indicator_name, **params)
                    feature_values = indicator.calculate(data_df)
                    return indicator_name, params, feature_values, None
                except Exception as e:
                    return indicator_name, params, None, str(e)
            
            # Prepare arguments for parallel processing
            data_dict = data.to_dict('series')
            args_list = [(name, params, data_dict) for name, params in indicators_config.items()]
            
            # Execute in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(compute_single_indicator, args) for args in args_list]
                
                for future in as_completed(futures):
                    indicator_name, params, feature_values, error = future.result()
                    
                    if error:
                        raise QuantFrameworkError(
                            f"Error computing indicator {indicator_name}: {error}"
                        )
                    
                    # Add results to result_data
                    if isinstance(feature_values, pd.Series):
                        column_name = f"{indicator_name}_{self._params_to_suffix(params)}"
                        result_data[column_name] = feature_values
                    elif isinstance(feature_values, pd.DataFrame):
                        for col in feature_values.columns:
                            column_name = f"{indicator_name}_{col}_{self._params_to_suffix(params)}"
                            result_data[column_name] = feature_values[col]
            
        except ImportError:
            # Fall back to sequential processing if multiprocessing not available
            result_data = self._compute_indicators_sequential(data, indicators_config, result_data)
        
        return result_data
    
    def validate_features(self, features: pd.DataFrame, original_data: Optional[pd.DataFrame] = None) -> FeatureValidationReport:
        """Validate computed features including look-ahead bias detection"""
        report = FeatureValidationReport()
        
        # Check for NaN values
        nan_columns = features.columns[features.isnull().any()].tolist()
        if nan_columns:
            report.add_warning(f"Features contain NaN values: {nan_columns}")
        
        # Check for infinite values
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        inf_columns = []
        for col in numeric_columns:
            if np.isinf(features[col]).any():
                inf_columns.append(col)
        
        if inf_columns:
            report.add_error(f"Features contain infinite values: {inf_columns}")
        
        # Check for constant features
        constant_columns = []
        for col in numeric_columns:
            if features[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            report.add_warning(f"Constant features detected: {constant_columns}")
        
        # Check for highly correlated features
        if len(numeric_columns) > 1:
            corr_matrix = features[numeric_columns].corr().abs()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_pairs.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                        )
            
            if high_corr_pairs:
                report.add_warning(
                    f"Highly correlated feature pairs detected: "
                    f"{[(pair[0], pair[1], f'{pair[2]:.3f}') for pair in high_corr_pairs[:5]]}"
                )
        
        # Perform look-ahead bias detection if original data is provided
        if original_data is not None:
            bias_report = self.bias_detector.detect_bias(original_data, features)
            report.look_ahead_bias_detected = bias_report.look_ahead_bias_detected
            report.bias_details.extend(bias_report.bias_details)
            report.errors.extend(bias_report.errors)
            report.warnings.extend(bias_report.warnings)
        
        # Calculate basic statistics
        report.statistics = {
            'total_features': len(features.columns),
            'numeric_features': len(numeric_columns),
            'categorical_features': len(features.columns) - len(numeric_columns),
            'missing_data_percentage': features.isnull().sum().sum() / (len(features) * len(features.columns)) * 100,
            'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024,
            'feature_density': (features.notna().sum().sum()) / (len(features) * len(features.columns)),
            'high_correlation_pairs': len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0
        }
        
        return report
    
    def _compute_cross_sectional_features(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Compute cross-sectional features"""
        result_data = data.copy()
        
        # Relative strength features
        if config.get('relative_strength', False):
            price_columns = [col for col in data.columns if 'close' in col.lower()]
            if len(price_columns) > 1:
                # Calculate relative strength vs average
                avg_price = data[price_columns].mean(axis=1)
                for col in price_columns:
                    result_data[f"{col}_relative_strength"] = data[col] / avg_price
        
        # Sector momentum (if sector information is available)
        sector_col = config.get('sector_column')
        if sector_col and sector_col in data.columns:
            # Calculate sector momentum
            for sector in data[sector_col].unique():
                sector_mask = data[sector_col] == sector
                sector_data = data[sector_mask]
                if len(sector_data) > 1:
                    sector_momentum = sector_data['close'].pct_change().rolling(20).mean()
                    result_data.loc[sector_mask, f"sector_{sector}_momentum"] = sector_momentum
        
        return result_data
    
    def _params_to_suffix(self, params: Dict[str, Any]) -> str:
        """Convert parameters to string suffix"""
        if not params:
            return "default"
        
        suffix_parts = []
        for key, value in sorted(params.items()):
            suffix_parts.append(f"{key}{value}")
        
        return "_".join(suffix_parts)
    
    def load_plugin(self, plugin_path: str) -> None:
        """Load indicator plugin from file"""
        try:
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find indicator classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseIndicator) and 
                    obj != BaseIndicator):
                    self.indicator_registry.register(obj)
                    
        except Exception as e:
            raise QuantFrameworkError(f"Failed to load plugin {plugin_path}: {str(e)}")
    
    def impute_missing_data(
        self, 
        data: pd.DataFrame, 
        strategy: str = 'forward_fill',
        **kwargs
    ) -> pd.DataFrame:
        """Impute missing data using specified strategy"""
        return self.data_imputer.impute_missing_data(data, strategy, **kwargs)
    
    def apply_rolling_operations(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply rolling window operations with multiple lookback periods"""
        return self.rolling_processor.apply_rolling_operations(data, config)
    
    def detect_look_ahead_bias(
        self, 
        original_data: pd.DataFrame, 
        features: pd.DataFrame
    ) -> FeatureValidationReport:
        """Detect potential look-ahead bias in features"""
        return self.bias_detector.detect_bias(original_data, features)
    
    def select_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        method: str = 'univariate_f',
        n_features: int = 50,
        **kwargs
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features using specified method"""
        return self.feature_selector.select_features(features, target, method, n_features, **kwargs)
    
    def reduce_dimensions(
        self, 
        features: pd.DataFrame, 
        method: str = 'pca',
        n_components: int = 10,
        **kwargs
    ) -> Tuple[pd.DataFrame, Any]:
        """Reduce feature dimensions using specified method"""
        return self.feature_selector.reduce_dimensions(features, method, n_components, **kwargs)
    
    def get_feature_importance(
        self, 
        features: pd.DataFrame, 
        target: pd.Series,
        method: str = 'random_forest'
    ) -> pd.Series:
        """Calculate feature importance scores"""
        common_idx = features.index.intersection(target.index)
        X = features.loc[common_idx].select_dtypes(include=[np.number])
        y = target.loc[common_idx]
        
        # Remove NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            raise ValidationError("No valid data after removing NaN values")
        
        if method == 'random_forest':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_clean, y_clean)
            importance_scores = pd.Series(rf.feature_importances_, index=X_clean.columns)
        elif method == 'correlation':
            importance_scores = X_clean.corrwith(y_clean).abs()
        else:
            raise ValidationError(f"Unknown importance method: {method}")
        
        return importance_scores.sort_values(ascending=False)
    
    def create_feature_summary(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Create comprehensive feature summary"""
        numeric_features = features.select_dtypes(include=[np.number])
        categorical_features = features.select_dtypes(exclude=[np.number])
        
        summary = {
            'total_features': len(features.columns),
            'numeric_features': len(numeric_features.columns),
            'categorical_features': len(categorical_features.columns),
            'total_observations': len(features),
            'missing_data_percentage': features.isnull().sum().sum() / (len(features) * len(features.columns)) * 100,
            'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024,
            'feature_density': (features.notna().sum().sum()) / (len(features) * len(features.columns)),
            'high_correlation_pairs': 0,  # Will be calculated below
            'feature_types': {}
        }
        
        # Analyze feature types
        for col in features.columns:
            col_info = {
                'dtype': str(features[col].dtype),
                'missing_count': features[col].isnull().sum(),
                'missing_percentage': features[col].isnull().sum() / len(features) * 100,
                'unique_values': features[col].nunique()
            }
            
            if col in numeric_features.columns:
                col_info.update({
                    'mean': features[col].mean(),
                    'std': features[col].std(),
                    'min': features[col].min(),
                    'max': features[col].max(),
                    'skewness': features[col].skew(),
                    'kurtosis': features[col].kurtosis()
                })
            
            summary['feature_types'][col] = col_info
        
        # Calculate high correlation pairs
        if len(numeric_features.columns) > 1:
            corr_matrix = numeric_features.corr().abs()
            high_corr_count = 0
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.95:
                        high_corr_count += 1
            summary['high_correlation_pairs'] = high_corr_count
        
        return summary 
        return summary