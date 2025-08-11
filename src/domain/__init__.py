"""
Domain layer - Core business entities, value objects, and interfaces.

This package contains the core business logic and domain model for the quantitative framework.
It defines the entities, value objects, and interfaces that make up the domain model.
"""

# Re-export key domain components for easier imports
from .entities import Portfolio, Position, Strategy
from .value_objects import Price, Return, Signal, RiskMetrics, PerformanceMetrics
from .exceptions import (
    QuantFrameworkError, DataError, ValidationError, OptimizationError, 
    BacktestError, RiskError, MLError, ConfigurationError
)
from .interfaces import (
    IDataManager, IFeatureEngine, IStrategy, IBacktestEngine,
    IPortfolioOptimizer, IRiskManager, IMLFramework, IRepository,
    IPortfolioRepository, IStrategyRepository, IEventPublisher,
    ILogger, IConfigManager
)