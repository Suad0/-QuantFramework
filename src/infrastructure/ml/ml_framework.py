"""
Main ML framework that integrates all ML components.

This module provides the main MLFramework class that implements the IMLFramework
interface and coordinates all ML operations including model training, prediction,
validation, and ensemble management.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from datetime import datetime

from src.domain.interfaces import IMLFramework, IMLModel
from src.domain.exceptions import ValidationError
from .model_registry import ModelRegistry, ModelMetadata
from .cross_validation import WalkForwardValidator, TimeSeriesSplit
from .preprocessing import FinancialPreprocessor
from .ensemble import ModelEnsemble, EnsembleMethod
from .hyperparameter_optimization import HyperparameterOptimizer, RegularizationManager


class MLModel(IMLModel):
    """
    Wrapper class for ML models that implements the IMLModel interface.
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        preprocessor: Optional[FinancialPreprocessor] = None,
        metadata: Optional[ModelMetadata] = None
    ):
        """
        Initialize ML model wrapper.
        
        Args:
            model: Trained scikit-learn model
            preprocessor: Optional preprocessor
            metadata: Model metadata
        """
        self.model = model
        self.preprocessor = preprocessor
        self.metadata = metadata
        self._feature_names: Optional[List[str]] = None
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the model.
        
        Args:
            features: Input features
            
        Returns:
            Predictions as pandas Series
        """
        if not isinstance(features, pd.DataFrame):
            raise ValidationError("Features must be a pandas DataFrame")
        
        # Preprocess features if preprocessor is available
        if self.preprocessor is not None:
            features_processed = self.preprocessor.transform(features)
        else:
            features_processed = features
        
        # Make predictions
        predictions = self.model.predict(features_processed)
        
        return pd.Series(predictions, index=features.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance_dict = {}
        
        # Try different methods to get feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        else:
            warnings.warn("Model does not support feature importance")
            return importance_dict
        
        # Get feature names
        if self._feature_names is not None:
            feature_names = self._feature_names
        elif self.preprocessor is not None:
            try:
                feature_names = self.preprocessor.get_feature_names()
            except:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create importance dictionary
        for name, importance in zip(feature_names, importances):
            importance_dict[name] = float(importance)
        
        return importance_dict
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        import pickle
        from pathlib import Path
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'metadata': self.metadata,
            'feature_names': self._feature_names
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        import pickle
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.preprocessor = model_data.get('preprocessor')
        self.metadata = model_data.get('metadata')
        self._feature_names = model_data.get('feature_names')


class MLFramework(IMLFramework):
    """
    Main ML framework that coordinates all ML operations.
    """
    
    def __init__(
        self,
        registry_path: str = "models",
        default_cv_method: str = "walk_forward",
        default_preprocessing: bool = True
    ):
        """
        Initialize ML framework.
        
        Args:
            registry_path: Path for model registry
            default_cv_method: Default cross-validation method
            default_preprocessing: Whether to use preprocessing by default
        """
        self.registry = ModelRegistry(registry_path)
        self.default_cv_method = default_cv_method
        self.default_preprocessing = default_preprocessing
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.regularization_manager = RegularizationManager()
        
        # Supported algorithms
        self.supported_algorithms = {
            'linear_regression': self._create_linear_regression,
            'ridge': self._create_ridge,
            'lasso': self._create_lasso,
            'random_forest': self._create_random_forest,
            'enhanced_random_forest': self._create_enhanced_random_forest,
            'gradient_boosting': self._create_gradient_boosting,
            'xgboost': self._create_xgboost,
            'enhanced_xgboost': self._create_enhanced_xgboost,
            'svm': self._create_svm,
            'enhanced_svm': self._create_enhanced_svm,
            'neural_network': self._create_neural_network,
            'lstm': self._create_lstm,
            'lstm_attention': self._create_lstm_attention
        }
    
    def train_model(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        config: Dict[str, Any]
    ) -> MLModel:
        """
        Train a machine learning model.
        
        Args:
            features: Training features
            targets: Training targets
            config: Training configuration
            
        Returns:
            Trained ML model
        """
        if not isinstance(features, pd.DataFrame):
            raise ValidationError("Features must be a pandas DataFrame")
        
        if not isinstance(targets, pd.Series):
            raise ValidationError("Targets must be a pandas Series")
        
        # Extract configuration
        algorithm = config.get('algorithm', 'random_forest')
        model_params = config.get('model_params', {})
        preprocessing_config = config.get('preprocessing', {})
        cv_config = config.get('cross_validation', {})
        hyperparameter_config = config.get('hyperparameter_optimization', {})
        regularization_config = config.get('regularization', {})
        register_model = config.get('register_model', True)
        model_name = config.get('model_name', f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Validate algorithm
        if algorithm not in self.supported_algorithms:
            raise ValidationError(f"Unsupported algorithm: {algorithm}")
        
        # Create preprocessor if requested
        preprocessor = None
        if preprocessing_config.get('enabled', self.default_preprocessing):
            preprocessor = FinancialPreprocessor(**preprocessing_config.get('params', {}))
            
            # Fit and transform features
            features_processed = preprocessor.fit_transform(features)
        else:
            features_processed = features
        
        # Create base model
        base_model = self.supported_algorithms[algorithm](**model_params)
        
        # Apply hyperparameter optimization if requested
        if hyperparameter_config.get('enabled', False):
            optimization_method = hyperparameter_config.get('method', 'random')
            param_space = hyperparameter_config.get('param_space')
            
            if param_space is None:
                # Use default parameter space
                param_space = self.hyperparameter_optimizer.get_default_param_space(algorithm)
            
            if param_space:
                optimizer_kwargs = hyperparameter_config.get('optimizer_kwargs', {})
                best_params, best_score, optimization_info = self.hyperparameter_optimizer.optimize(
                    base_model, param_space, features_processed, targets,
                    method=optimization_method, **optimizer_kwargs
                )
                
                # Update model with best parameters
                base_model.set_params(**best_params)
        
        # Apply regularization if requested
        regularization_techniques = regularization_config.get('techniques', [])
        if regularization_techniques:
            regularization_kwargs = regularization_config.get('kwargs', {})
            model = self.regularization_manager.apply_regularization(
                base_model, regularization_techniques, **regularization_kwargs
            )
        else:
            model = base_model
        
        # Train the final model
        model.fit(features_processed, targets)
        
        # Create ML model wrapper
        ml_model = MLModel(model, preprocessor)
        ml_model._feature_names = features.columns.tolist()
        
        # Perform cross-validation if requested
        cv_scores = {}
        if cv_config.get('enabled', True):
            try:
                cv_scores = self._perform_cross_validation(
                    model, features_processed, targets, cv_config
                )
            except Exception as e:
                warnings.warn(f"Cross-validation failed: {e}")
                cv_scores = {}
        
        # Register model if requested
        if register_model:
            metadata = ModelMetadata(
                model_id=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=model_name,
                version="1.0.0",
                model_type="regression",
                algorithm=algorithm,
                created_at=datetime.now(),
                created_by="ml_framework",
                description=f"Model trained with {algorithm}",
                parameters=model_params,
                performance_metrics=self._calculate_performance_metrics(model, features_processed, targets),
                feature_names=features.columns.tolist(),
                target_name=targets.name or "target",
                training_data_hash=self.registry.calculate_data_hash(features),
                validation_method=cv_config.get('method', self.default_cv_method),
                cross_validation_scores=cv_scores
            )
            
            ml_model.metadata = metadata
            self.registry.register_model(model, metadata)
        
        return ml_model
    
    def predict(
        self,
        model: MLModel,
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained ML model
            features: Input features
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if not isinstance(model, MLModel):
            raise ValidationError("Model must be an MLModel instance")
        
        if not isinstance(features, pd.DataFrame):
            raise ValidationError("Features must be a pandas DataFrame")
        
        # Make predictions
        predictions = model.predict(features)
        
        # Calculate prediction confidence if possible
        confidence = None
        if hasattr(model.model, 'predict_proba'):
            try:
                probabilities = model.model.predict_proba(features)
                confidence = np.max(probabilities, axis=1)
            except:
                pass
        
        return {
            'predictions': predictions,
            'confidence': confidence,
            'model_metadata': model.metadata,
            'feature_names': features.columns.tolist(),
            'prediction_timestamp': datetime.now()
        }
    
    def validate_model(
        self,
        model: MLModel,
        test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate model performance on test data.
        
        Args:
            model: Trained ML model
            test_data: Test data with features and target
            
        Returns:
            Validation results
        """
        if not isinstance(model, MLModel):
            raise ValidationError("Model must be an MLModel instance")
        
        if not isinstance(test_data, pd.DataFrame):
            raise ValidationError("Test data must be a pandas DataFrame")
        
        # Assume last column is target
        features = test_data.iloc[:, :-1]
        targets = test_data.iloc[:, -1]
        
        # Make predictions
        predictions = model.predict(features)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions))
        }
        
        # Calculate financial-specific metrics
        financial_metrics = self._calculate_financial_metrics(targets, predictions)
        metrics.update(financial_metrics)
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'validation_timestamp': datetime.now()
        }
    
    def create_ensemble(
        self,
        models: List[Union[MLModel, str]],
        method: EnsembleMethod = EnsembleMethod.SIMPLE_AVERAGE,
        **kwargs
    ) -> ModelEnsemble:
        """
        Create an ensemble from multiple models.
        
        Args:
            models: List of MLModel instances or model names from registry
            method: Ensemble method
            **kwargs: Additional ensemble parameters
            
        Returns:
            Model ensemble
        """
        base_models = []
        
        for model in models:
            if isinstance(model, str):
                # Load from registry
                registered_model = self.registry.get_model(model)
                if registered_model is None:
                    raise ValidationError(f"Model {model} not found in registry")
                base_models.append(registered_model)
            elif isinstance(model, MLModel):
                base_models.append(model.model)
            else:
                raise ValidationError("Models must be MLModel instances or model names")
        
        return ModelEnsemble(base_models, method, **kwargs)
    
    def _perform_cross_validation(
        self,
        model: BaseEstimator,
        features: pd.DataFrame,
        targets: pd.Series,
        cv_config: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Perform cross-validation on the model."""
        cv_method = cv_config.get('method', self.default_cv_method)
        cv_params = cv_config.get('params', {})
        
        # Create cross-validator
        if cv_method == 'walk_forward':
            cv = WalkForwardValidator(**cv_params)
        elif cv_method == 'time_series':
            cv = TimeSeriesSplit(**cv_params)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_params.get('n_splits', 5))
        
        # Perform cross-validation
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        cv_results = cross_validate(model, features, targets, cv=cv, scoring=scoring)
        
        return {
            'mse': (-cv_results['test_neg_mean_squared_error']).tolist(),
            'mae': (-cv_results['test_neg_mean_absolute_error']).tolist(),
            'r2': cv_results['test_r2'].tolist()
        }
    
    def _calculate_performance_metrics(
        self,
        model: BaseEstimator,
        features: pd.DataFrame,
        targets: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics for the model."""
        predictions = model.predict(features)
        
        return {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions))
        }
    
    def _calculate_financial_metrics(
        self,
        targets: pd.Series,
        predictions: pd.Series
    ) -> Dict[str, float]:
        """Calculate financial-specific metrics."""
        # Directional accuracy
        target_direction = np.sign(targets.diff().dropna())
        pred_direction = np.sign(predictions.diff().dropna())
        directional_accuracy = (target_direction == pred_direction).mean()
        
        # Hit rate (percentage of correct predictions)
        hit_rate = (np.sign(targets) == np.sign(predictions)).mean()
        
        return {
            'directional_accuracy': directional_accuracy,
            'hit_rate': hit_rate
        }
    
    # Model creation methods
    def _create_linear_regression(self, **params):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)
    
    def _create_ridge(self, **params):
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    
    def _create_lasso(self, **params):
        from sklearn.linear_model import Lasso
        return Lasso(**params)
    
    def _create_random_forest(self, **params):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    
    def _create_gradient_boosting(self, **params):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(**params)
    
    def _create_svm(self, **params):
        from sklearn.svm import SVR
        return SVR(**params)
    
    def _create_neural_network(self, **params):
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(**params)
    
    def _create_xgboost(self, **params):
        try:
            import xgboost as xgb
            return xgb.XGBRegressor(**params)
        except ImportError:
            warnings.warn("XGBoost not installed. Install with: pip install xgboost")
            # Fallback to gradient boosting
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(**params)
    
    def _create_lstm(self, **params):
        """Create LSTM model using a simple wrapper."""
        from .lstm_wrapper import SimpleLSTMWrapper
        return SimpleLSTMWrapper(**params)
    
    def _create_lstm_attention(self, **params):
        """Create LSTM model with attention mechanisms."""
        from .lstm_wrapper import LSTMWrapper
        # Enable attention by default for this variant
        params.setdefault('use_attention', True)
        return LSTMWrapper(**params)
    
    def _create_enhanced_xgboost(self, **params):
        """Create enhanced XGBoost model with auto-tuning."""
        from .advanced_algorithms import EnhancedXGBoostRegressor
        return EnhancedXGBoostRegressor(**params)
    
    def _create_enhanced_random_forest(self, **params):
        """Create enhanced Random Forest model with auto-tuning."""
        from .advanced_algorithms import EnhancedRandomForestRegressor
        return EnhancedRandomForestRegressor(**params)
    
    def _create_enhanced_svm(self, **params):
        """Create enhanced SVM model with auto-tuning."""
        from .advanced_algorithms import EnhancedSVMRegressor
        return EnhancedSVMRegressor(**params)