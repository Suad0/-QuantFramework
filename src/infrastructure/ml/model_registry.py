"""
Model registry for managing ML models with versioning and metadata.

This module provides functionality to store, version, and manage machine learning
models with comprehensive metadata tracking for financial applications.
"""

import json
import pickle
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import pandas as pd
from sklearn.base import BaseEstimator

from src.domain.exceptions import ValidationError


@dataclass
class ModelMetadata:
    """Metadata for a machine learning model."""
    
    model_id: str
    name: str
    version: str
    model_type: str
    algorithm: str
    created_at: datetime
    created_by: str
    description: str
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    feature_names: List[str]
    target_name: str
    training_data_hash: str
    validation_method: str
    cross_validation_scores: Dict[str, List[float]]
    feature_importance: Optional[Dict[str, float]] = None
    tags: List[str] = None
    is_active: bool = True
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.model_id:
            raise ValidationError("Model ID cannot be empty")
        if not self.name:
            raise ValidationError("Model name cannot be empty")
        if not self.version:
            raise ValidationError("Model version cannot be empty")
        if self.tags is None:
            self.tags = []


class ModelRegistry:
    """Registry for managing ML models with versioning and metadata."""
    
    def __init__(self, registry_path: Union[str, Path] = "models"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to store model files and metadata
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.registry_path / "models").mkdir(exist_ok=True)
        (self.registry_path / "metadata").mkdir(exist_ok=True)
        
        self._models_cache: Dict[str, Any] = {}
        self._metadata_cache: Dict[str, ModelMetadata] = {}
    
    def register_model(
        self,
        model: BaseEstimator,
        metadata: ModelMetadata,
        overwrite: bool = False
    ) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model: Trained scikit-learn model
            metadata: Model metadata
            overwrite: Whether to overwrite existing model
            
        Returns:
            Model ID of the registered model
            
        Raises:
            ValidationError: If model already exists and overwrite is False
        """
        model_key = f"{metadata.name}_{metadata.version}"
        
        if not overwrite and self._model_exists(model_key):
            raise ValidationError(f"Model {model_key} already exists. Use overwrite=True to replace.")
        
        # Save model
        model_path = self.registry_path / "models" / f"{model_key}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_path = self.registry_path / "metadata" / f"{model_key}.json"
        with open(metadata_path, 'w') as f:
            # Convert datetime to string for JSON serialization
            metadata_dict = asdict(metadata)
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            json.dump(metadata_dict, f, indent=2)
        
        # Update caches
        self._models_cache[model_key] = model
        self._metadata_cache[model_key] = metadata
        
        return metadata.model_id
    
    def get_model(self, name: str, version: str = "latest") -> Optional[BaseEstimator]:
        """
        Retrieve a model from the registry.
        
        Args:
            name: Model name
            version: Model version or "latest"
            
        Returns:
            Trained model or None if not found
        """
        if version == "latest":
            version = self._get_latest_version(name)
            if not version:
                return None
        
        model_key = f"{name}_{version}"
        
        # Check cache first
        if model_key in self._models_cache:
            return self._models_cache[model_key]
        
        # Load from disk
        model_path = self.registry_path / "models" / f"{model_key}.pkl"
        if not model_path.exists():
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Cache for future use
        self._models_cache[model_key] = model
        return model
    
    def get_metadata(self, name: str, version: str = "latest") -> Optional[ModelMetadata]:
        """
        Retrieve model metadata.
        
        Args:
            name: Model name
            version: Model version or "latest"
            
        Returns:
            Model metadata or None if not found
        """
        if version == "latest":
            version = self._get_latest_version(name)
            if not version:
                return None
        
        model_key = f"{name}_{version}"
        
        # Check cache first
        if model_key in self._metadata_cache:
            return self._metadata_cache[model_key]
        
        # Load from disk
        metadata_path = self.registry_path / "metadata" / f"{model_key}.json"
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Convert string back to datetime
        metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
        metadata = ModelMetadata(**metadata_dict)
        
        # Cache for future use
        self._metadata_cache[model_key] = metadata
        return metadata
    
    def list_models(self, name_filter: Optional[str] = None) -> List[ModelMetadata]:
        """
        List all models in the registry.
        
        Args:
            name_filter: Optional filter by model name
            
        Returns:
            List of model metadata
        """
        models = []
        metadata_dir = self.registry_path / "metadata"
        
        for metadata_file in metadata_dir.glob("*.json"):
            model_key = metadata_file.stem
            metadata = self.get_metadata(*model_key.split('_', 1))
            
            if metadata and (not name_filter or name_filter in metadata.name):
                models.append(metadata)
        
        # Sort by creation date (newest first)
        return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def delete_model(self, name: str, version: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if model was deleted, False if not found
        """
        model_key = f"{name}_{version}"
        
        model_path = self.registry_path / "models" / f"{model_key}.pkl"
        metadata_path = self.registry_path / "metadata" / f"{model_key}.json"
        
        deleted = False
        
        if model_path.exists():
            model_path.unlink()
            deleted = True
        
        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True
        
        # Remove from caches
        self._models_cache.pop(model_key, None)
        self._metadata_cache.pop(model_key, None)
        
        return deleted
    
    def get_model_versions(self, name: str) -> List[str]:
        """
        Get all versions of a model.
        
        Args:
            name: Model name
            
        Returns:
            List of version strings
        """
        versions = []
        metadata_dir = self.registry_path / "metadata"
        
        for metadata_file in metadata_dir.glob(f"{name}_*.json"):
            model_key = metadata_file.stem
            if model_key.startswith(f"{name}_"):
                version = model_key[len(name) + 1:]
                versions.append(version)
        
        return sorted(versions)
    
    def update_model_status(self, name: str, version: str, is_active: bool) -> bool:
        """
        Update model active status.
        
        Args:
            name: Model name
            version: Model version
            is_active: New active status
            
        Returns:
            True if updated successfully, False if model not found
        """
        metadata = self.get_metadata(name, version)
        if not metadata:
            return False
        
        metadata.is_active = is_active
        
        # Save updated metadata
        model_key = f"{name}_{version}"
        metadata_path = self.registry_path / "metadata" / f"{model_key}.json"
        
        with open(metadata_path, 'w') as f:
            metadata_dict = asdict(metadata)
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            json.dump(metadata_dict, f, indent=2)
        
        # Update cache
        self._metadata_cache[model_key] = metadata
        
        return True
    
    def calculate_data_hash(self, data: pd.DataFrame) -> str:
        """
        Calculate hash of training data for tracking.
        
        Args:
            data: Training data
            
        Returns:
            SHA-256 hash of the data
        """
        # Create a string representation of the data
        data_str = data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _model_exists(self, model_key: str) -> bool:
        """Check if a model exists in the registry."""
        model_path = self.registry_path / "models" / f"{model_key}.pkl"
        return model_path.exists()
    
    def _get_latest_version(self, name: str) -> Optional[str]:
        """Get the latest version of a model."""
        versions = self.get_model_versions(name)
        if not versions:
            return None
        
        # For now, use simple string sorting
        # In production, you might want semantic versioning
        return sorted(versions)[-1]