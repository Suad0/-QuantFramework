"""
Configuration for the dependency injection container.
Registers all services and their implementations.
"""
from typing import Optional
from ..domain.interfaces import ILogger, IConfigManager
from .container import DIContainer, get_container


def configure_container() -> DIContainer:
    """Configure and return the DI container with all service registrations"""
    container = get_container()
    
    # Register infrastructure services first
    def config_factory():
        from .config.config_manager import get_config_manager
        return get_config_manager()
    
    def logger_factory():
        from .logging.logger import get_logger
        return get_logger("quantframework")
    
    container.register_factory(IConfigManager, config_factory)
    container.register_factory(ILogger, logger_factory)
    
    # Additional services will be registered as they are implemented
    
    return container


def initialize_application():
    """Initialize the application with all dependencies"""
    container = configure_container()
    
    # Perform any additional initialization here
    # For example: validate configuration, setup database connections, etc.
    
    # Validate that essential services are available
    try:
        config_manager = container.resolve(IConfigManager)
        logger = container.resolve(ILogger)
        
        logger.info("Dependency injection container initialized successfully")
        
    except Exception as e:
        print(f"Failed to initialize essential services: {e}")
        raise
    
    return container