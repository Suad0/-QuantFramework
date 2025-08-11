"""
Dependency injection container for the quantitative framework.
Provides loose coupling between components through dependency inversion.
"""
from typing import Dict, Any, Type, TypeVar, Callable, Optional, Union
from abc import ABC
import inspect
from functools import wraps
from ..domain.exceptions import ConfigurationError, InvalidOperationError

T = TypeVar('T')


class DIContainer:
    """Simple but effective dependency injection container"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._transients: Dict[str, Type] = {}
        self._scoped: Dict[str, Type] = {}
        self._scope_instances: Dict[str, Dict[str, Any]] = {}
        self._current_scope: Optional[str] = None
    
    def register_singleton(self, interface: Type[T], implementation: Union[Type[T], T]) -> 'DIContainer':
        """Register a singleton service (one instance for application lifetime)"""
        key = self._get_key(interface)
        
        if isinstance(implementation, type):
            # Register class to be instantiated later
            self._singletons[key] = implementation
        else:
            # Register instance directly
            self._services[key] = implementation
        
        return self
    
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> 'DIContainer':
        """Register a transient service (new instance each time)"""
        key = self._get_key(interface)
        self._transients[key] = implementation
        return self
    
    def register_scoped(self, interface: Type[T], implementation: Type[T]) -> 'DIContainer':
        """Register a scoped service (one instance per scope)"""
        key = self._get_key(interface)
        self._scoped[key] = implementation
        return self
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> 'DIContainer':
        """Register a factory function for creating instances"""
        key = self._get_key(interface)
        self._factories[key] = factory
        return self
    
    def register_instance(self, interface: Type[T], instance: T) -> 'DIContainer':
        """Register a specific instance"""
        key = self._get_key(interface)
        self._services[key] = instance
        return self
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service instance"""
        key = self._get_key(interface)
        
        # Check if already instantiated
        if key in self._services:
            return self._services[key]
        
        # Check singletons
        if key in self._singletons:
            if isinstance(self._singletons[key], type):
                instance = self._create_instance(self._singletons[key])
                self._services[key] = instance
                return instance
            else:
                return self._singletons[key]
        
        # Check scoped
        if key in self._scoped:
            return self._resolve_scoped(key, self._scoped[key])
        
        # Check transients
        if key in self._transients:
            return self._create_instance(self._transients[key])
        
        # Check factories
        if key in self._factories:
            return self._factories[key]()
        
        raise ConfigurationError(
            f"Service {interface.__name__} is not registered",
            error_code="SERVICE_NOT_REGISTERED",
            context={"interface": interface.__name__}
        )
    
    def create_scope(self, scope_name: str) -> 'DIScope':
        """Create a new dependency injection scope"""
        if scope_name in self._scope_instances:
            raise InvalidOperationError(
                f"Scope '{scope_name}' already exists",
                error_code="SCOPE_EXISTS"
            )
        
        self._scope_instances[scope_name] = {}
        return DIScope(self, scope_name)
    
    def _resolve_scoped(self, key: str, implementation: Type[T]) -> T:
        """Resolve scoped service"""
        if self._current_scope is None:
            raise InvalidOperationError(
                "Cannot resolve scoped service outside of scope",
                error_code="NO_ACTIVE_SCOPE"
            )
        
        scope_instances = self._scope_instances.get(self._current_scope, {})
        
        if key not in scope_instances:
            instance = self._create_instance(implementation)
            scope_instances[key] = instance
            self._scope_instances[self._current_scope] = scope_instances
        
        return scope_instances[key]
    
    def _create_instance(self, implementation: Type[T]) -> T:
        """Create instance with dependency injection"""
        try:
            # Get constructor signature
            sig = inspect.signature(implementation.__init__)
            params = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                
                # Try to resolve parameter type
                if param.annotation != inspect.Parameter.empty:
                    try:
                        params[param_name] = self.resolve(param.annotation)
                    except ConfigurationError:
                        # If parameter has default value, use it
                        if param.default != inspect.Parameter.empty:
                            params[param_name] = param.default
                        else:
                            raise
                elif param.default != inspect.Parameter.empty:
                    params[param_name] = param.default
                else:
                    raise ConfigurationError(
                        f"Cannot resolve parameter '{param_name}' for {implementation.__name__}",
                        error_code="PARAMETER_RESOLUTION_FAILED"
                    )
            
            return implementation(**params)
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create instance of {implementation.__name__}: {str(e)}",
                error_code="INSTANCE_CREATION_FAILED",
                context={"implementation": implementation.__name__}
            )
    
    def _get_key(self, interface: Type) -> str:
        """Get string key for interface"""
        return f"{interface.__module__}.{interface.__name__}"
    
    def _set_current_scope(self, scope_name: Optional[str]) -> None:
        """Set current scope (internal use)"""
        self._current_scope = scope_name
    
    def _dispose_scope(self, scope_name: str) -> None:
        """Dispose of scope and its instances"""
        if scope_name in self._scope_instances:
            # Call dispose on instances that support it
            for instance in self._scope_instances[scope_name].values():
                if hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception:
                        pass  # Ignore disposal errors
            
            del self._scope_instances[scope_name]


class DIScope:
    """Dependency injection scope context manager"""
    
    def __init__(self, container: DIContainer, scope_name: str):
        self._container = container
        self._scope_name = scope_name
        self._previous_scope = None
    
    def __enter__(self) -> 'DIScope':
        self._previous_scope = self._container._current_scope
        self._container._set_current_scope(self._scope_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._container._set_current_scope(self._previous_scope)
        self._container._dispose_scope(self._scope_name)


def inject(interface: Type[T]) -> Callable:
    """Decorator for dependency injection"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get container from global registry or create new one
            container = get_container()
            service = container.resolve(interface)
            return func(service, *args, **kwargs)
        return wrapper
    return decorator


# Global container registry
_global_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Get global container instance"""
    global _global_container
    if _global_container is None:
        _global_container = DIContainer()
    return _global_container


def set_container(container: DIContainer) -> None:
    """Set global container instance"""
    global _global_container
    _global_container = container


def configure_container() -> DIContainer:
    """Configure and return the DI container with default registrations"""
    container = DIContainer()
    
    # Register default implementations here when they're created
    # This will be expanded as we implement concrete classes
    
    return container