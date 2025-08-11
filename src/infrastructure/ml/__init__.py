"""
Machine Learning infrastructure for the quantitative framework.

This module provides ML capabilities including:
- Model registry with versioning and metadata management
- Time series cross-validation techniques
- Financial data-aware feature scaling and preprocessing
- Model ensemble framework for robust predictions
"""

from .model_registry import ModelRegistry, ModelMetadata
from .cross_validation import TimeSeriesValidator, WalkForwardValidator
from .preprocessing import FinancialScaler, FinancialPreprocessor
from .ensemble import ModelEnsemble, EnsembleMethod
from .ml_framework import MLFramework

__all__ = [
    'ModelRegistry',
    'ModelMetadata',
    'TimeSeriesValidator',
    'WalkForwardValidator',
    'FinancialScaler',
    'FinancialPreprocessor',
    'ModelEnsemble',
    'EnsembleMethod',
    'MLFramework'
]