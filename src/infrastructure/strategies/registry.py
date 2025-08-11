"""
Strategy registry for dynamic loading and management of trading strategies.

This module provides a centralized registry for managing trading strategies,
including dynamic loading, registration, and lifecycle management.
"""

import importlib
import inspect
from typing import Dict, List, Type, Any, Optional
from pathlib import Path
from datetime import datetime
import json

from ...domain.interfaces import IStrategy
from ...domain.entities import Strategy
from ...domain.exceptions import StrategyError, ValidationError
from ...infrastructure.logging.logger import get_logger


class StrategyRegistry:
    """
    Registry for managing trading strategies with dynamic loading capabilities.
    
    Provides functionality to register, load, and manage trading strategies
    with support for plugin architecture and dynamic discovery.
    """
    
    def __init__(self, logger=None):
        """
        Initialize strategy registry.
        
        Args:
            logger: Logger instance for logging operations
        """
        self.logger = logger or get_logger(__name__)
        self._strategies: Dict[str, Type[IStrategy]] = {}
        self._strategy_metadata: Dict[str, Dict[str, Any]] = {}
        self._strategy_instances: Dict[str, IStrategy] = {}
        self._plugin_paths: List[Path] = []
        
    def register_strategy(
        self, 
        strategy_class: Type[IStrategy], 
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a strategy class in the registry.
        
        Args:
            strategy_class: Strategy class implementing IStrategy interface
            name: Optional name for the strategy (defaults to class name)
            metadata: Optional metadata for the strategy
            
        Raises:
            StrategyError: If strategy registration fails
            ValidationError: If strategy class is invalid
        """
        try:
            # Validate strategy class
            if not issubclass(strategy_class, IStrategy):
                raise ValidationError(
                    f"Strategy class {strategy_class.__name__} must implement IStrategy interface"
                )
            
            strategy_name = name or strategy_class.__name__
            
            # Check for duplicate registration
            if strategy_name in self._strategies:
                self.logger.warning(
                    f"Strategy {strategy_name} already registered, overwriting"
                )
            
            # Register strategy
            self._strategies[strategy_name] = strategy_class
            self._strategy_metadata[strategy_name] = {
                'class_name': strategy_class.__name__,
                'module': strategy_class.__module__,
                'registered_at': datetime.now().isoformat(),
                'description': getattr(strategy_class, '__doc__', ''),
                **(metadata or {})
            }
            
            self.logger.info(f"Successfully registered strategy: {strategy_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register strategy {strategy_class.__name__}: {str(e)}")
            raise StrategyError(
                f"Strategy registration failed: {str(e)}",
                error_code="REGISTRATION_FAILED"
            ) from e
    
    def unregister_strategy(self, name: str) -> None:
        """
        Unregister a strategy from the registry.
        
        Args:
            name: Name of the strategy to unregister
            
        Raises:
            StrategyError: If strategy is not found
        """
        if name not in self._strategies:
            raise StrategyError(
                f"Strategy {name} not found in registry",
                error_code="STRATEGY_NOT_FOUND"
            )
        
        # Remove from all registries
        del self._strategies[name]
        del self._strategy_metadata[name]
        
        # Remove instance if exists
        if name in self._strategy_instances:
            del self._strategy_instances[name]
        
        self.logger.info(f"Successfully unregistered strategy: {name}")
    
    def get_strategy_class(self, name: str) -> Type[IStrategy]:
        """
        Get strategy class by name.
        
        Args:
            name: Name of the strategy
            
        Returns:
            Strategy class
            
        Raises:
            StrategyError: If strategy is not found
        """
        if name not in self._strategies:
            raise StrategyError(
                f"Strategy {name} not found in registry",
                error_code="STRATEGY_NOT_FOUND"
            )
        
        return self._strategies[name]
    
    def create_strategy_instance(
        self, 
        name: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> IStrategy:
        """
        Create an instance of a registered strategy.
        
        Args:
            name: Name of the strategy
            parameters: Optional parameters for strategy initialization
            
        Returns:
            Strategy instance
            
        Raises:
            StrategyError: If strategy creation fails
        """
        try:
            strategy_class = self.get_strategy_class(name)
            
            # Create instance with parameters
            if parameters:
                instance = strategy_class(**parameters)
            else:
                instance = strategy_class()
            
            # Cache instance
            instance_key = f"{name}_{id(instance)}"
            self._strategy_instances[instance_key] = instance
            
            self.logger.info(f"Created strategy instance: {name}")
            return instance
            
        except Exception as e:
            self.logger.error(f"Failed to create strategy instance {name}: {str(e)}")
            raise StrategyError(
                f"Strategy instance creation failed: {str(e)}",
                error_code="INSTANCE_CREATION_FAILED"
            ) from e
    
    def list_strategies(self) -> List[str]:
        """
        List all registered strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    def get_strategy_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered strategy.
        
        Args:
            name: Name of the strategy
            
        Returns:
            Strategy metadata
            
        Raises:
            StrategyError: If strategy is not found
        """
        if name not in self._strategy_metadata:
            raise StrategyError(
                f"Strategy {name} not found in registry",
                error_code="STRATEGY_NOT_FOUND"
            )
        
        return self._strategy_metadata[name].copy()
    
    def add_plugin_path(self, path: Path) -> None:
        """
        Add a path for plugin discovery.
        
        Args:
            path: Path to search for strategy plugins
        """
        if not path.exists():
            raise ValidationError(f"Plugin path does not exist: {path}")
        
        self._plugin_paths.append(path)
        self.logger.info(f"Added plugin path: {path}")
    
    def discover_strategies(self, path: Optional[Path] = None) -> int:
        """
        Discover and register strategies from plugin paths.
        
        Args:
            path: Optional specific path to search (uses all plugin paths if None)
            
        Returns:
            Number of strategies discovered and registered
            
        Raises:
            StrategyError: If discovery fails
        """
        discovered_count = 0
        search_paths = [path] if path else self._plugin_paths
        
        for search_path in search_paths:
            try:
                discovered_count += self._discover_strategies_in_path(search_path)
            except Exception as e:
                self.logger.error(f"Failed to discover strategies in {search_path}: {str(e)}")
                raise StrategyError(
                    f"Strategy discovery failed: {str(e)}",
                    error_code="DISCOVERY_FAILED"
                ) from e
        
        self.logger.info(f"Discovered {discovered_count} strategies")
        return discovered_count
    
    def _discover_strategies_in_path(self, path: Path) -> int:
        """
        Discover strategies in a specific path.
        
        Args:
            path: Path to search for strategies
            
        Returns:
            Number of strategies discovered
        """
        discovered_count = 0
        
        # Search for Python files
        for py_file in path.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            try:
                # Import module
                module_name = py_file.stem
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find strategy classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (obj != IStrategy and 
                        issubclass(obj, IStrategy) and 
                        obj.__module__ == module.__name__):
                        
                        self.register_strategy(obj, name)
                        discovered_count += 1
                        
            except Exception as e:
                self.logger.warning(f"Failed to load module {py_file}: {str(e)}")
                continue
        
        return discovered_count
    
    def export_registry(self, file_path: Path) -> None:
        """
        Export registry metadata to a JSON file.
        
        Args:
            file_path: Path to export file
        """
        try:
            export_data = {
                'strategies': self._strategy_metadata,
                'exported_at': datetime.now().isoformat(),
                'total_strategies': len(self._strategies)
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported registry to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export registry: {str(e)}")
            raise StrategyError(
                f"Registry export failed: {str(e)}",
                error_code="EXPORT_FAILED"
            ) from e
    
    def import_registry(self, file_path: Path) -> None:
        """
        Import registry metadata from a JSON file.
        
        Args:
            file_path: Path to import file
        """
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            # Update metadata (strategies need to be registered separately)
            for name, metadata in import_data.get('strategies', {}).items():
                if name in self._strategy_metadata:
                    self._strategy_metadata[name].update(metadata)
            
            self.logger.info(f"Imported registry from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to import registry: {str(e)}")
            raise StrategyError(
                f"Registry import failed: {str(e)}",
                error_code="IMPORT_FAILED"
            ) from e
    
    def clear_registry(self) -> None:
        """Clear all registered strategies."""
        self._strategies.clear()
        self._strategy_metadata.clear()
        self._strategy_instances.clear()
        self.logger.info("Cleared strategy registry")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            'total_strategies': len(self._strategies),
            'total_instances': len(self._strategy_instances),
            'plugin_paths': len(self._plugin_paths),
            'strategies': list(self._strategies.keys())
        }