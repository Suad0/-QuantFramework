"""
Uncertainty quantification for ML models in financial applications.

This module provides various methods for quantifying prediction uncertainty,
including bootstrap sampling, Bayesian approaches, and ensemble-based methods.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils import resample
from scipy import stats
import warnings
from datetime import datetime
from abc import ABC, abstractmethod

from src.domain.exceptions import ValidationError


class UncertaintyQuantifier(ABC):
    """Abstract base class for uncertainty quantification methods."""
    
    @abstractmethod
    def quantify_uncertainty(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Quantify prediction uncertainty.
        
        Args:
            model: Trained model
            X: Input features
            y: True targets (optional, for some methods)
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dictionary containing predictions and uncertainty measures
        """
        pass


class BootstrapUncertaintyQuantifier(UncertaintyQuantifier):
    """Bootstrap-based uncertainty quantification."""
    
    def __init__(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None
    ):
        """
        Initialize bootstrap uncertainty quantifier.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            random_state: Random seed
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        
        if not 0 < confidence_level < 1:
            raise ValidationError("Confidence level must be between 0 and 1")
    
    def quantify_uncertainty(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Quantify uncertainty using bootstrap sampling.
        
        Args:
            model: Trained model
            X: Input features
            y: Training targets
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with predictions, confidence intervals, and uncertainty
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Store original model state
        original_model = model
        
        # Bootstrap predictions
        bootstrap_predictions = []
        
        np.random.seed(self.random_state)
        
        for i in range(self.n_bootstrap):
            # Create bootstrap sample
            X_boot, y_boot = resample(
                X, y, 
                random_state=self.random_state + i if self.random_state else None
            )
            
            # Clone and train model on bootstrap sample
            from sklearn.base import clone
            boot_model = clone(original_model)
            
            try:
                boot_model.fit(X_boot, y_boot)
                boot_pred = boot_model.predict(X)
                bootstrap_predictions.append(boot_pred)
            except Exception as e:
                warnings.warn(f"Bootstrap iteration {i} failed: {e}")
                continue
        
        if not bootstrap_predictions:
            raise ValidationError("All bootstrap iterations failed")
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calculate statistics
        mean_pred = np.mean(bootstrap_predictions, axis=0)
        std_pred = np.std(bootstrap_predictions, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
        
        return {
            'predictions': mean_pred,
            'uncertainty': std_pred,
            'confidence_lower': lower_bound,
            'confidence_upper': upper_bound,
            'confidence_level': self.confidence_level,
            'n_bootstrap': len(bootstrap_predictions)
        }


class EnsembleUncertaintyQuantifier(UncertaintyQuantifier):
    """Ensemble-based uncertainty quantification."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize ensemble uncertainty quantifier.
        
        Args:
            confidence_level: Confidence level for intervals
        """
        self.confidence_level = confidence_level
        
        if not 0 < confidence_level < 1:
            raise ValidationError("Confidence level must be between 0 and 1")
    
    def quantify_uncertainty(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Quantify uncertainty using ensemble predictions.
        
        Args:
            model: Ensemble model (must have estimators_ attribute)
            X: Input features
            y: Not used for ensemble method
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with predictions, confidence intervals, and uncertainty
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Check if model is an ensemble
        if not hasattr(model, 'estimators_'):
            raise ValidationError("Model must be an ensemble with estimators_ attribute")
        
        # Get predictions from all estimators
        estimator_predictions = []
        for estimator in model.estimators_:
            try:
                pred = estimator.predict(X)
                estimator_predictions.append(pred)
            except Exception as e:
                warnings.warn(f"Estimator prediction failed: {e}")
                continue
        
        if not estimator_predictions:
            raise ValidationError("No estimator predictions available")
        
        estimator_predictions = np.array(estimator_predictions)
        
        # Calculate statistics
        mean_pred = np.mean(estimator_predictions, axis=0)
        std_pred = np.std(estimator_predictions, axis=0)
        
        # Calculate confidence intervals assuming normal distribution
        alpha = 1 - self.confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        margin_of_error = z_score * std_pred
        lower_bound = mean_pred - margin_of_error
        upper_bound = mean_pred + margin_of_error
        
        return {
            'predictions': mean_pred,
            'uncertainty': std_pred,
            'confidence_lower': lower_bound,
            'confidence_upper': upper_bound,
            'confidence_level': self.confidence_level,
            'n_estimators': len(estimator_predictions)
        }


class BayesianUncertaintyQuantifier(UncertaintyQuantifier):
    """Bayesian uncertainty quantification using variational inference."""
    
    def __init__(
        self,
        n_samples: int = 1000,
        confidence_level: float = 0.95,
        prior_std: float = 1.0
    ):
        """
        Initialize Bayesian uncertainty quantifier.
        
        Args:
            n_samples: Number of posterior samples
            confidence_level: Confidence level for intervals
            prior_std: Standard deviation of prior distribution
        """
        self.n_samples = n_samples
        self.confidence_level = confidence_level
        self.prior_std = prior_std
        
        if not 0 < confidence_level < 1:
            raise ValidationError("Confidence level must be between 0 and 1")
    
    def quantify_uncertainty(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Quantify uncertainty using Bayesian approach.
        
        Note: This is a simplified implementation. For full Bayesian inference,
        consider using specialized libraries like PyMC3 or TensorFlow Probability.
        
        Args:
            model: Trained model
            X: Input features
            y: Training targets
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with predictions, confidence intervals, and uncertainty
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Get base prediction
        base_pred = model.predict(X)
        
        # Estimate prediction uncertainty based on residuals
        # This is a simplified approach - full Bayesian inference would be more complex
        train_pred = model.predict(X)  # Assuming X is training data for simplicity
        residuals = y - train_pred
        residual_std = np.std(residuals)
        
        # Sample from posterior predictive distribution
        # Simplified: assume normal distribution with estimated variance
        posterior_samples = []
        for _ in range(self.n_samples):
            # Add noise based on residual standard deviation
            noise = np.random.normal(0, residual_std, size=len(base_pred))
            sample_pred = base_pred + noise
            posterior_samples.append(sample_pred)
        
        posterior_samples = np.array(posterior_samples)
        
        # Calculate statistics
        mean_pred = np.mean(posterior_samples, axis=0)
        std_pred = np.std(posterior_samples, axis=0)
        
        # Calculate credible intervals
        alpha = 1 - self.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(posterior_samples, lower_percentile, axis=0)
        upper_bound = np.percentile(posterior_samples, upper_percentile, axis=0)
        
        return {
            'predictions': mean_pred,
            'uncertainty': std_pred,
            'confidence_lower': lower_bound,
            'confidence_upper': upper_bound,
            'confidence_level': self.confidence_level,
            'n_samples': self.n_samples,
            'residual_std': residual_std
        }


class UncertaintyQuantificationManager:
    """Manager for different uncertainty quantification methods."""
    
    def __init__(self):
        """Initialize uncertainty quantification manager."""
        self.methods = {
            'bootstrap': BootstrapUncertaintyQuantifier,
            'ensemble': EnsembleUncertaintyQuantifier,
            'bayesian': BayesianUncertaintyQuantifier
        }
    
    def quantify_uncertainty(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        method: str = 'bootstrap',
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Quantify prediction uncertainty using specified method.
        
        Args:
            model: Trained model
            X: Input features
            y: Training targets (required for some methods)
            method: Uncertainty quantification method
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary with uncertainty quantification results
        """
        if method not in self.methods:
            raise ValidationError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")
        
        # Create quantifier instance
        quantifier_class = self.methods[method]
        quantifier = quantifier_class(**kwargs)
        
        # Quantify uncertainty
        return quantifier.quantify_uncertainty(model, X, y)
    
    def compare_methods(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        methods: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compare different uncertainty quantification methods.
        
        Args:
            model: Trained model
            X: Input features
            y: Training targets
            methods: List of methods to compare
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary mapping method names to uncertainty results
        """
        if methods is None:
            methods = list(self.methods.keys())
        
        results = {}
        
        for method in methods:
            try:
                result = self.quantify_uncertainty(model, X, y, method, **kwargs)
                results[method] = result
            except Exception as e:
                warnings.warn(f"Method {method} failed: {e}")
                continue
        
        return results
    
    def get_prediction_intervals(
        self,
        uncertainty_result: Dict[str, np.ndarray],
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate prediction intervals for different confidence levels.
        
        Args:
            uncertainty_result: Result from uncertainty quantification
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary mapping confidence levels to (lower, upper) bounds
        """
        predictions = uncertainty_result['predictions']
        uncertainty = uncertainty_result['uncertainty']
        
        intervals = {}
        
        for conf_level in confidence_levels:
            # Calculate z-score for confidence level
            alpha = 1 - conf_level
            z_score = stats.norm.ppf(1 - alpha / 2)
            
            # Calculate bounds
            margin = z_score * uncertainty
            lower = predictions - margin
            upper = predictions + margin
            
            intervals[conf_level] = (lower, upper)
        
        return intervals