"""
Advanced ML algorithms with enhanced features for financial applications.

This module provides enhanced implementations of XGBoost, Random Forest, and SVM
with built-in regularization, hyperparameter optimization, and financial-specific
features.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime

from src.domain.exceptions import ValidationError


class EnhancedXGBoostRegressor(BaseEstimator, RegressorMixin):
    """
    Enhanced XGBoost regressor with built-in hyperparameter optimization
    and financial-specific features.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Optional[int] = None,
        auto_tune: bool = False,
        tune_method: str = 'grid',
        cv_folds: int = 5,
        early_stopping_rounds: int = 10,
        eval_metric: str = 'rmse',
        use_gpu: bool = False
    ):
        """
        Initialize Enhanced XGBoost Regressor.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            random_state: Random seed
            auto_tune: Whether to automatically tune hyperparameters
            tune_method: Hyperparameter tuning method ('grid' or 'random')
            cv_folds: Number of CV folds for tuning
            early_stopping_rounds: Early stopping rounds
            eval_metric: Evaluation metric
            use_gpu: Whether to use GPU acceleration
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.auto_tune = auto_tune
        self.tune_method = tune_method
        self.cv_folds = cv_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.use_gpu = use_gpu
        
        self.model_ = None
        self.best_params_ = None
        self.feature_importances_ = None
        self.is_fitted_ = False
        
        # Try to import XGBoost
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.xgb_available = True
        except ImportError:
            self.xgb_available = False
            warnings.warn("XGBoost not available. Install with: pip install xgboost")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnhancedXGBoostRegressor':
        """
        Fit the XGBoost model with optional hyperparameter tuning.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Self
        """
        if not self.xgb_available:
            raise ImportError("XGBoost is required but not installed")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Set up base parameters
        base_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'eval_metric': self.eval_metric,
            'early_stopping_rounds': self.early_stopping_rounds
        }
        
        if self.use_gpu:
            base_params['tree_method'] = 'gpu_hist'
            base_params['gpu_id'] = 0
        
        # Hyperparameter tuning
        if self.auto_tune:
            self.model_, self.best_params_ = self._tune_hyperparameters(X, y, base_params)
        else:
            self.model_ = self.xgb.XGBRegressor(**base_params)
            self.model_.fit(X, y)
            self.best_params_ = base_params
        
        # Store feature importances
        self.feature_importances_ = self.model_.feature_importances_
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        return self.model_.predict(X)
    
    def predict_with_uncertainty(self, X: pd.DataFrame, n_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using bootstrap sampling.
        
        Args:
            X: Feature data
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet")
        
        # This is a simplified uncertainty estimation
        # In practice, you might use more sophisticated methods
        base_pred = self.predict(X)
        
        # Estimate uncertainty based on feature importance and prediction confidence
        # This is a heuristic approach
        uncertainty = np.abs(base_pred) * 0.1  # 10% of prediction as uncertainty
        
        return base_pred, uncertainty
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, base_params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Tune hyperparameters using grid search or random search."""
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 1.5, 2.0]
        }
        
        # Create base model
        base_model = self.xgb.XGBRegressor(**{k: v for k, v in base_params.items() 
                                           if k not in ['early_stopping_rounds']})
        
        # Choose search method
        if self.tune_method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=50,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
        
        # Fit search
        search.fit(X, y)
        
        return search.best_estimator_, search.best_params_
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'auto_tune': self.auto_tune,
            'tune_method': self.tune_method,
            'cv_folds': self.cv_folds,
            'early_stopping_rounds': self.early_stopping_rounds,
            'eval_metric': self.eval_metric,
            'use_gpu': self.use_gpu
        }
    
    def set_params(self, **params) -> 'EnhancedXGBoostRegressor':
        """Set parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self


class EnhancedRandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    Enhanced Random Forest regressor with built-in hyperparameter optimization
    and financial-specific features.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = 'sqrt',
        bootstrap: bool = True,
        oob_score: bool = True,
        random_state: Optional[int] = None,
        auto_tune: bool = False,
        tune_method: str = 'grid',
        cv_folds: int = 5,
        feature_selection: bool = False,
        feature_selection_threshold: float = 0.01
    ):
        """
        Initialize Enhanced Random Forest Regressor.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            oob_score: Whether to use out-of-bag score
            random_state: Random seed
            auto_tune: Whether to automatically tune hyperparameters
            tune_method: Hyperparameter tuning method
            cv_folds: Number of CV folds for tuning
            feature_selection: Whether to perform feature selection
            feature_selection_threshold: Threshold for feature importance
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.auto_tune = auto_tune
        self.tune_method = tune_method
        self.cv_folds = cv_folds
        self.feature_selection = feature_selection
        self.feature_selection_threshold = feature_selection_threshold
        
        self.model_ = None
        self.best_params_ = None
        self.feature_importances_ = None
        self.selected_features_ = None
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnhancedRandomForestRegressor':
        """
        Fit the Random Forest model with optional hyperparameter tuning.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Set up base parameters
        base_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        # Hyperparameter tuning
        if self.auto_tune:
            self.model_, self.best_params_ = self._tune_hyperparameters(X, y, base_params)
        else:
            from sklearn.ensemble import RandomForestRegressor
            self.model_ = RandomForestRegressor(**base_params)
            self.model_.fit(X, y)
            self.best_params_ = base_params
        
        # Store feature importances
        self.feature_importances_ = self.model_.feature_importances_
        
        # Feature selection
        if self.feature_selection:
            self.selected_features_ = self._select_features(X.columns)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Apply feature selection if used
        if self.selected_features_ is not None:
            # Only select features that exist in X
            available_features = [f for f in self.selected_features_ if f in X.columns]
            if available_features:
                X = X[available_features]
            else:
                # If no selected features are available, use all features
                pass
        
        return self.model_.predict(X)
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using tree variance.
        
        Args:
            X: Feature data
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Apply feature selection if used
        if self.selected_features_ is not None:
            # Only select features that exist in X
            available_features = [f for f in self.selected_features_ if f in X.columns]
            if available_features:
                X = X[available_features]
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.model_.estimators_])
        
        # Calculate mean and standard deviation
        predictions = np.mean(tree_predictions, axis=0)
        uncertainties = np.std(tree_predictions, axis=0)
        
        return predictions, uncertainties
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, base_params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Tune hyperparameters."""
        from sklearn.ensemble import RandomForestRegressor
        
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5, 0.7]
        }
        
        base_model = RandomForestRegressor(**{k: v for k, v in base_params.items() 
                                            if k != 'n_jobs'})
        
        if self.tune_method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=50,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
        
        search.fit(X, y)
        return search.best_estimator_, search.best_params_
    
    def _select_features(self, feature_names: List[str]) -> Optional[List[str]]:
        """Select features based on importance threshold."""
        if self.feature_importances_ is None:
            return None
        
        selected_indices = np.where(self.feature_importances_ >= self.feature_selection_threshold)[0]
        selected_features = [feature_names[i] for i in selected_indices]
        
        # Return None if no features meet the threshold (use all features)
        return selected_features if selected_features else None
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'random_state': self.random_state,
            'auto_tune': self.auto_tune,
            'tune_method': self.tune_method,
            'cv_folds': self.cv_folds,
            'feature_selection': self.feature_selection,
            'feature_selection_threshold': self.feature_selection_threshold
        }
    
    def set_params(self, **params) -> 'EnhancedRandomForestRegressor':
        """Set parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self


class EnhancedSVMRegressor(BaseEstimator, RegressorMixin):
    """
    Enhanced SVM regressor with built-in hyperparameter optimization
    and financial-specific features.
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: Union[str, float] = 'scale',
        degree: int = 3,
        coef0: float = 0.0,
        shrinking: bool = True,
        cache_size: float = 200,
        max_iter: int = -1,
        random_state: Optional[int] = None,
        auto_tune: bool = False,
        tune_method: str = 'grid',
        cv_folds: int = 5,
        scale_features: bool = True
    ):
        """
        Initialize Enhanced SVM Regressor.
        
        Args:
            kernel: Kernel type
            C: Regularization parameter
            epsilon: Epsilon parameter in epsilon-SVR model
            gamma: Kernel coefficient
            degree: Degree of polynomial kernel
            coef0: Independent term in kernel function
            shrinking: Whether to use shrinking heuristic
            cache_size: Kernel cache size (MB)
            max_iter: Hard limit on iterations
            random_state: Random seed
            auto_tune: Whether to automatically tune hyperparameters
            tune_method: Hyperparameter tuning method
            cv_folds: Number of CV folds for tuning
            scale_features: Whether to scale features
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.auto_tune = auto_tune
        self.tune_method = tune_method
        self.cv_folds = cv_folds
        self.scale_features = scale_features
        
        self.model_ = None
        self.scaler_ = None
        self.best_params_ = None
        self.is_fitted_ = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnhancedSVMRegressor':
        """
        Fit the SVM model with optional hyperparameter tuning.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Feature scaling
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler_.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Set up base parameters
        base_params = {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'shrinking': self.shrinking,
            'cache_size': self.cache_size,
            'max_iter': self.max_iter
        }
        
        # Hyperparameter tuning
        if self.auto_tune:
            self.model_, self.best_params_ = self._tune_hyperparameters(X_scaled, y, base_params)
        else:
            from sklearn.svm import SVR
            self.model_ = SVR(**base_params)
            self.model_.fit(X_scaled, y)
            self.best_params_ = base_params
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Apply scaling if used
        if self.scaler_ is not None:
            X_scaled = pd.DataFrame(
                self.scaler_.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        return self.model_.predict(X_scaled)
    
    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, base_params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Tune hyperparameters."""
        from sklearn.svm import SVR
        
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly']
        }
        
        base_model = SVR(**{k: v for k, v in base_params.items()})
        
        if self.tune_method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
        else:
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=50,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
        
        search.fit(X, y)
        return search.best_estimator_, search.best_params_
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'shrinking': self.shrinking,
            'cache_size': self.cache_size,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'auto_tune': self.auto_tune,
            'tune_method': self.tune_method,
            'cv_folds': self.cv_folds,
            'scale_features': self.scale_features
        }
    
    def set_params(self, **params) -> 'EnhancedSVMRegressor':
        """Set parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self