"""
Model ensemble framework for robust predictions.

This module provides ensemble methods that combine multiple models
to create more robust and accurate predictions for financial applications.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings

from src.domain.exceptions import ValidationError


class EnsembleMethod(Enum):
    """Ensemble combination methods."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    DYNAMIC_WEIGHTING = "dynamic_weighting"


class BaseEnsemble(ABC, BaseEstimator):
    """Abstract base class for ensemble methods."""
    
    def __init__(self, models: List[BaseEstimator], method: EnsembleMethod):
        """
        Initialize base ensemble.
        
        Args:
            models: List of base models
            method: Ensemble combination method
        """
        self.models = models
        self.method = method
        self.is_fitted_ = False
        
        if not models:
            raise ValidationError("At least one model must be provided")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseEnsemble':
        """Fit the ensemble to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ensemble."""
        pass
    
    def _validate_fitted(self):
        """Check if ensemble has been fitted."""
        if not self.is_fitted_:
            raise ValidationError("Ensemble has not been fitted yet")


class ModelEnsemble(BaseEnsemble, RegressorMixin):
    """
    Model ensemble for regression tasks.
    
    Combines multiple regression models using various ensemble methods
    to create more robust predictions.
    """
    
    def __init__(
        self,
        models: List[BaseEstimator],
        method: EnsembleMethod = EnsembleMethod.SIMPLE_AVERAGE,
        weights: Optional[List[float]] = None,
        meta_model: Optional[BaseEstimator] = None,
        cv_folds: int = 5,
        performance_metric: str = 'mse'
    ):
        """
        Initialize model ensemble.
        
        Args:
            models: List of base models
            method: Ensemble combination method
            weights: Manual weights for weighted average (if applicable)
            meta_model: Meta-model for stacking (if applicable)
            cv_folds: Number of CV folds for weight calculation
            performance_metric: Metric for performance-based weighting
        """
        super().__init__(models, method)
        
        self.weights = weights
        self.meta_model = meta_model
        self.cv_folds = cv_folds
        self.performance_metric = performance_metric
        
        self.model_weights_: Optional[np.ndarray] = None
        self.model_performances_: Dict[str, float] = {}
        self.fitted_models_: List[BaseEstimator] = []
        
        # Validate inputs
        if method == EnsembleMethod.WEIGHTED_AVERAGE and weights is not None:
            if len(weights) != len(models):
                raise ValidationError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                raise ValidationError("Weights must sum to 1.0")
        
        if method == EnsembleMethod.STACKING and meta_model is None:
            raise ValidationError("Meta-model must be provided for stacking")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ModelEnsemble':
        """
        Fit the ensemble to training data.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("X must be a pandas DataFrame")
        
        if not isinstance(y, pd.Series):
            raise ValidationError("y must be a pandas Series")
        
        # Fit all base models
        self.fitted_models_ = []
        for i, model in enumerate(self.models):
            try:
                fitted_model = model.fit(X, y)
                self.fitted_models_.append(fitted_model)
            except Exception as e:
                warnings.warn(f"Model {i} failed to fit: {str(e)}")
                continue
        
        if not self.fitted_models_:
            raise ValidationError("No models could be fitted successfully")
        
        # Calculate model weights based on method
        if self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            if self.weights is not None:
                self.model_weights_ = np.array(self.weights[:len(self.fitted_models_)])
            else:
                self.model_weights_ = self._calculate_performance_weights(X, y)
        
        elif self.method == EnsembleMethod.DYNAMIC_WEIGHTING:
            self.model_weights_ = self._calculate_dynamic_weights(X, y)
        
        elif self.method == EnsembleMethod.STACKING:
            self._fit_meta_model(X, y)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Feature data
            
        Returns:
            Ensemble predictions
        """
        self._validate_fitted()
        
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("X must be a pandas DataFrame")
        
        # Get predictions from all models
        predictions = []
        for model in self.fitted_models_:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Model prediction failed: {str(e)}")
                continue
        
        if not predictions:
            raise ValidationError("No models could make predictions")
        
        predictions = np.array(predictions).T  # Shape: (n_samples, n_models)
        
        # Combine predictions based on method
        if self.method == EnsembleMethod.SIMPLE_AVERAGE:
            return np.mean(predictions, axis=1)
        
        elif self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            if self.model_weights_ is not None:
                weights = self.model_weights_[:predictions.shape[1]]
                weights = weights / weights.sum()  # Normalize
                return np.average(predictions, axis=1, weights=weights)
            else:
                return np.mean(predictions, axis=1)
        
        elif self.method == EnsembleMethod.MEDIAN:
            return np.median(predictions, axis=1)
        
        elif self.method == EnsembleMethod.STACKING:
            return self.meta_model.predict(predictions)
        
        elif self.method == EnsembleMethod.DYNAMIC_WEIGHTING:
            return self._dynamic_weighted_prediction(X, predictions)
        
        else:
            return np.mean(predictions, axis=1)
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Feature data
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        self._validate_fitted()
        
        # Get predictions from all models
        predictions = []
        for model in self.fitted_models_:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception:
                continue
        
        predictions = np.array(predictions).T
        
        # Calculate ensemble prediction and uncertainty
        ensemble_pred = self.predict(X)
        uncertainty = np.std(predictions, axis=1)
        
        return ensemble_pred, uncertainty
    
    def get_model_weights(self) -> Optional[np.ndarray]:
        """Get the weights assigned to each model."""
        return self.model_weights_
    
    def get_model_performances(self) -> Dict[str, float]:
        """Get performance metrics for each model."""
        return self.model_performances_.copy()
    
    def _calculate_performance_weights(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Calculate weights based on cross-validation performance."""
        performances = []
        
        for model in self.fitted_models_:
            try:
                if self.performance_metric == 'mse':
                    scores = cross_val_score(
                        model, X, y, cv=self.cv_folds, 
                        scoring='neg_mean_squared_error'
                    )
                    performance = -np.mean(scores)  # Convert to positive MSE
                elif self.performance_metric == 'mae':
                    scores = cross_val_score(
                        model, X, y, cv=self.cv_folds,
                        scoring='neg_mean_absolute_error'
                    )
                    performance = -np.mean(scores)
                else:
                    performance = 1.0  # Default weight
                
                performances.append(performance)
            except Exception:
                performances.append(float('inf'))  # Worst performance
        
        performances = np.array(performances)
        
        # Convert to weights (inverse of error)
        weights = 1.0 / (performances + 1e-8)  # Add small epsilon to avoid division by zero
        weights = weights / weights.sum()  # Normalize
        
        return weights
    
    def _calculate_dynamic_weights(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Calculate dynamic weights based on recent performance."""
        # This is a simplified version - in practice, you might use
        # more sophisticated methods like exponential weighting
        return self._calculate_performance_weights(X, y)
    
    def _fit_meta_model(self, X: pd.DataFrame, y: pd.Series):
        """Fit meta-model for stacking ensemble."""
        # Generate out-of-fold predictions for meta-model training
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.fitted_models_)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            
            for i, model in enumerate(self.models):
                # Fit model on fold training data
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train_fold, y_train_fold)
                
                # Predict on fold validation data
                pred = model_copy.predict(X_val_fold)
                meta_features[val_idx, i] = pred
        
        # Fit meta-model on out-of-fold predictions
        self.meta_model.fit(meta_features, y)
    
    def _dynamic_weighted_prediction(self, X: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """Make dynamic weighted predictions based on input features."""
        # Simplified dynamic weighting - could be enhanced with more sophisticated methods
        if self.model_weights_ is not None:
            weights = self.model_weights_[:predictions.shape[1]]
            weights = weights / weights.sum()
            return np.average(predictions, axis=1, weights=weights)
        else:
            return np.mean(predictions, axis=1)


class ClassificationEnsemble(BaseEnsemble, ClassifierMixin):
    """
    Model ensemble for classification tasks.
    
    Combines multiple classification models using various ensemble methods.
    """
    
    def __init__(
        self,
        models: List[BaseEstimator],
        method: EnsembleMethod = EnsembleMethod.VOTING,
        weights: Optional[List[float]] = None,
        voting: str = 'hard'
    ):
        """
        Initialize classification ensemble.
        
        Args:
            models: List of base models
            method: Ensemble combination method
            weights: Manual weights for weighted voting
            voting: Voting type ('hard' or 'soft')
        """
        super().__init__(models, method)
        
        self.weights = weights
        self.voting = voting
        self.fitted_models_: List[BaseEstimator] = []
        self.classes_: Optional[np.ndarray] = None
        
        if voting not in ['hard', 'soft']:
            raise ValidationError("Voting must be 'hard' or 'soft'")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ClassificationEnsemble':
        """
        Fit the ensemble to training data.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Self
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError("X must be a pandas DataFrame")
        
        if not isinstance(y, pd.Series):
            raise ValidationError("y must be a pandas Series")
        
        self.classes_ = np.unique(y)
        
        # Fit all base models
        self.fitted_models_ = []
        for model in self.models:
            try:
                fitted_model = model.fit(X, y)
                self.fitted_models_.append(fitted_model)
            except Exception as e:
                warnings.warn(f"Model failed to fit: {str(e)}")
                continue
        
        if not self.fitted_models_:
            raise ValidationError("No models could be fitted successfully")
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Feature data
            
        Returns:
            Ensemble predictions
        """
        self._validate_fitted()
        
        if self.voting == 'hard':
            return self._hard_voting_predict(X)
        else:
            return self._soft_voting_predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using the ensemble.
        
        Args:
            X: Feature data
            
        Returns:
            Class probabilities
        """
        self._validate_fitted()
        
        probabilities = []
        for model in self.fitted_models_:
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)
                    probabilities.append(proba)
                except Exception:
                    continue
        
        if not probabilities:
            raise ValidationError("No models support probability prediction")
        
        # Average probabilities
        avg_proba = np.mean(probabilities, axis=0)
        return avg_proba
    
    def _hard_voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using hard voting."""
        predictions = []
        for model in self.fitted_models_:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception:
                continue
        
        if not predictions:
            raise ValidationError("No models could make predictions")
        
        predictions = np.array(predictions).T
        
        # Majority voting
        ensemble_pred = []
        for sample_preds in predictions:
            unique, counts = np.unique(sample_preds, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            ensemble_pred.append(majority_class)
        
        return np.array(ensemble_pred)
    
    def _soft_voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using soft voting."""
        try:
            probabilities = self.predict_proba(X)
            return self.classes_[np.argmax(probabilities, axis=1)]
        except Exception:
            # Fall back to hard voting
            return self._hard_voting_predict(X)


class AdaptiveEnsemble(ModelEnsemble):
    """
    Adaptive ensemble that adjusts weights based on recent performance.
    
    This ensemble continuously adapts model weights based on recent
    prediction accuracy, making it suitable for non-stationary financial data.
    """
    
    def __init__(
        self,
        models: List[BaseEstimator],
        adaptation_window: int = 100,
        learning_rate: float = 0.01,
        min_weight: float = 0.01
    ):
        """
        Initialize adaptive ensemble.
        
        Args:
            models: List of base models
            adaptation_window: Window size for performance tracking
            learning_rate: Learning rate for weight updates
            min_weight: Minimum weight for any model
        """
        super().__init__(models, EnsembleMethod.DYNAMIC_WEIGHTING)
        
        self.adaptation_window = adaptation_window
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        
        self.recent_errors_: List[List[float]] = []
        self.prediction_count_ = 0
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AdaptiveEnsemble':
        """Fit the adaptive ensemble."""
        super().fit(X, y)
        
        # Initialize equal weights
        n_models = len(self.fitted_models_)
        self.model_weights_ = np.ones(n_models) / n_models
        
        # Initialize error tracking
        self.recent_errors_ = [[] for _ in range(n_models)]
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make adaptive predictions."""
        predictions = []
        for model in self.fitted_models_:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception:
                continue
        
        predictions = np.array(predictions).T
        
        # Weighted average with current weights
        weights = self.model_weights_[:predictions.shape[1]]
        weights = weights / weights.sum()
        
        return np.average(predictions, axis=1, weights=weights)
    
    def update_weights(self, X: pd.DataFrame, y_true: pd.Series):
        """
        Update model weights based on recent performance.
        
        Args:
            X: Feature data used for predictions
            y_true: True target values
        """
        # Get individual model predictions
        predictions = []
        for model in self.fitted_models_:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception:
                predictions.append(np.full(len(X), np.nan))
        
        # Calculate errors for each model
        for i, pred in enumerate(predictions):
            if not np.isnan(pred).all():
                error = mean_squared_error(y_true, pred)
                self.recent_errors_[i].append(error)
                
                # Keep only recent errors
                if len(self.recent_errors_[i]) > self.adaptation_window:
                    self.recent_errors_[i].pop(0)
        
        # Update weights based on recent performance
        avg_errors = []
        for errors in self.recent_errors_:
            if errors:
                avg_errors.append(np.mean(errors))
            else:
                avg_errors.append(float('inf'))
        
        avg_errors = np.array(avg_errors)
        
        # Convert errors to weights (inverse relationship)
        new_weights = 1.0 / (avg_errors + 1e-8)
        new_weights = new_weights / new_weights.sum()
        
        # Apply minimum weight constraint
        new_weights = np.maximum(new_weights, self.min_weight)
        new_weights = new_weights / new_weights.sum()
        
        # Update weights with learning rate
        self.model_weights_ = (
            (1 - self.learning_rate) * self.model_weights_ +
            self.learning_rate * new_weights
        )
        
        self.prediction_count_ += len(X)