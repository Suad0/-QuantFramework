"""
Custom exception hierarchy for the quantitative framework.
Provides specific error types for different failure scenarios.
"""
from typing import Optional, Dict, Any


class QuantFrameworkError(Exception):
    """Base exception for the quantitative framework"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataError(QuantFrameworkError):
    """Data-related errors"""
    pass


class DataSourceError(DataError):
    """Errors related to data source connectivity or availability"""
    pass


class DataQualityError(DataError):
    """Errors related to data quality issues"""
    pass


class DataValidationError(DataError):
    """Errors related to data validation failures"""
    pass


class CacheError(DataError):
    """Errors related to caching operations"""
    pass


class ValidationError(QuantFrameworkError):
    """Validation errors"""
    pass


class ParameterValidationError(ValidationError):
    """Errors related to parameter validation"""
    pass


class SignalValidationError(ValidationError):
    """Errors related to signal validation"""
    pass


class PortfolioValidationError(ValidationError):
    """Errors related to portfolio validation"""
    pass


class OptimizationError(QuantFrameworkError):
    """Portfolio optimization errors"""
    pass


class ConstraintError(OptimizationError):
    """Errors related to optimization constraints"""
    pass


class ConvergenceError(OptimizationError):
    """Errors when optimization fails to converge"""
    pass


class BacktestError(QuantFrameworkError):
    """Backtesting errors"""
    pass


class StrategyError(BacktestError):
    """Errors related to strategy execution"""
    pass


class TradingSimulationError(BacktestError):
    """Errors during trading simulation"""
    pass


class PerformanceCalculationError(BacktestError):
    """Errors during performance metric calculations"""
    pass


class RiskError(QuantFrameworkError):
    """Risk management errors"""
    pass


class RiskLimitExceededError(RiskError):
    """Error when risk limits are exceeded"""
    pass


class RiskCalculationError(RiskError):
    """Errors during risk metric calculations"""
    pass


class MLError(QuantFrameworkError):
    """Machine learning errors"""
    pass


class ModelTrainingError(MLError):
    """Errors during model training"""
    pass


class ModelPredictionError(MLError):
    """Errors during model prediction"""
    pass


class ModelValidationError(MLError):
    """Errors during model validation"""
    pass


class FeatureEngineeringError(MLError):
    """Errors during feature engineering"""
    pass


class ConfigurationError(QuantFrameworkError):
    """Configuration-related errors"""
    pass


class MissingConfigurationError(ConfigurationError):
    """Error when required configuration is missing"""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Error when configuration values are invalid"""
    pass


class InfrastructureError(QuantFrameworkError):
    """Infrastructure-related errors"""
    pass


class DatabaseError(InfrastructureError):
    """Database-related errors"""
    pass


class NetworkError(InfrastructureError):
    """Network-related errors"""
    pass


class AuthenticationError(InfrastructureError):
    """Authentication-related errors"""
    pass


class AuthorizationError(InfrastructureError):
    """Authorization-related errors"""
    pass


class ResourceError(QuantFrameworkError):
    """Resource-related errors"""
    pass


class InsufficientResourcesError(ResourceError):
    """Error when system resources are insufficient"""
    pass


class ResourceNotFoundError(ResourceError):
    """Error when required resource is not found"""
    pass


class BusinessLogicError(QuantFrameworkError):
    """Business logic errors"""
    pass


class InvalidOperationError(BusinessLogicError):
    """Error when operation is not valid in current state"""
    pass


class ConcurrencyError(QuantFrameworkError):
    """Concurrency-related errors"""
    pass


class LockTimeoutError(ConcurrencyError):
    """Error when lock acquisition times out"""
    pass


class DeadlockError(ConcurrencyError):
    """Error when deadlock is detected"""
    pass


# Utility functions for error handling
def create_error_context(**kwargs) -> Dict[str, Any]:
    """Create error context dictionary"""
    return {k: v for k, v in kwargs.items() if v is not None}


def wrap_exception(
    original_exception: Exception, 
    new_exception_class: type, 
    message: str,
    error_code: Optional[str] = None,
    **context
) -> QuantFrameworkError:
    """Wrap an exception with additional context"""
    error_context = create_error_context(
        original_exception=str(original_exception),
        original_type=type(original_exception).__name__,
        **context
    )
    
    return new_exception_class(
        message=message,
        error_code=error_code,
        context=error_context
    )