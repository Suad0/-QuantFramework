"""
Comprehensive error handling middleware for the quantitative framework.
Provides centralized error handling, logging, and response formatting.
"""

import traceback
import sys
from typing import Any, Dict, Optional, Callable, Type
from functools import wraps
from datetime import datetime

from ...domain.exceptions import (
    QuantFrameworkError, DataError, ValidationError, OptimizationError,
    BacktestError, RiskError, MLError, ConfigurationError, InfrastructureError,
    ResourceError, BusinessLogicError, ConcurrencyError
)
from ...domain.interfaces import ILogger
from ..logging.logger import get_logger


class ErrorContext:
    """Context information for error handling."""
    
    def __init__(
        self,
        operation: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id
        self.additional_context = additional_context or {}
        self.timestamp = datetime.now()


class ErrorHandler:
    """
    Centralized error handler that provides consistent error processing,
    logging, and response formatting across the application.
    """
    
    def __init__(self, logger: Optional[ILogger] = None):
        """
        Initialize error handler.
        
        Args:
            logger: Logger instance for error logging
        """
        self.logger = logger or get_logger(__name__)
        self._error_handlers: Dict[Type[Exception], Callable] = {}
        self._setup_default_handlers()
    
    def _setup_default_handlers(self) -> None:
        """Setup default error handlers for different exception types."""
        self._error_handlers.update({
            DataError: self._handle_data_error,
            ValidationError: self._handle_validation_error,
            OptimizationError: self._handle_optimization_error,
            BacktestError: self._handle_backtest_error,
            RiskError: self._handle_risk_error,
            MLError: self._handle_ml_error,
            ConfigurationError: self._handle_configuration_error,
            InfrastructureError: self._handle_infrastructure_error,
            ResourceError: self._handle_resource_error,
            BusinessLogicError: self._handle_business_logic_error,
            ConcurrencyError: self._handle_concurrency_error,
            QuantFrameworkError: self._handle_framework_error,
            ValueError: self._handle_value_error,
            TypeError: self._handle_type_error,
            KeyError: self._handle_key_error,
            AttributeError: self._handle_attribute_error,
            Exception: self._handle_generic_error
        })
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        reraise: bool = False
    ) -> Dict[str, Any]:
        """
        Handle an error with appropriate logging and response formatting.
        
        Args:
            error: The exception to handle
            context: Error context information
            reraise: Whether to reraise the exception after handling
            
        Returns:
            Error response dictionary
            
        Raises:
            Exception: If reraise is True
        """
        try:
            # Find appropriate handler
            handler = self._find_handler(type(error))
            
            # Create error response
            error_response = handler(error, context)
            
            # Log the error
            self._log_error(error, error_response, context)
            
            # Reraise if requested
            if reraise:
                raise error
            
            return error_response
            
        except Exception as handler_error:
            # Fallback error handling
            self.logger.critical(
                "Error handler failed",
                original_error=str(error),
                handler_error=str(handler_error),
                traceback=traceback.format_exc()
            )
            
            fallback_response = {
                'error_code': 'HANDLER_FAILED',
                'message': 'Internal error handling failed',
                'timestamp': datetime.now().isoformat(),
                'severity': 'critical'
            }
            
            if reraise:
                raise error
            
            return fallback_response
    
    def _find_handler(self, error_type: Type[Exception]) -> Callable:
        """Find the most specific handler for an error type."""
        # Check for exact match first
        if error_type in self._error_handlers:
            return self._error_handlers[error_type]
        
        # Check for parent class matches
        for registered_type, handler in self._error_handlers.items():
            if issubclass(error_type, registered_type):
                return handler
        
        # Fallback to generic handler
        return self._error_handlers[Exception]
    
    def _log_error(
        self,
        error: Exception,
        error_response: Dict[str, Any],
        context: Optional[ErrorContext]
    ) -> None:
        """Log error with appropriate level and context."""
        severity = error_response.get('severity', 'error')
        
        log_data = {
            'error_code': error_response.get('error_code'),
            'error_type': type(error).__name__,
            'message': str(error),
            'severity': severity
        }
        
        # Add context information
        if context:
            log_data.update({
                'operation': context.operation,
                'user_id': context.user_id,
                'session_id': context.session_id,
                'request_id': context.request_id,
                'context': context.additional_context
            })
        
        # Add framework-specific context for QuantFrameworkError
        if isinstance(error, QuantFrameworkError):
            log_data.update({
                'framework_error_code': error.error_code,
                'framework_context': error.context
            })
        
        # Add traceback for critical errors
        if severity in ['critical', 'error']:
            log_data['traceback'] = traceback.format_exc()
        
        # Log with appropriate level
        if severity == 'critical':
            self.logger.critical("Critical error occurred", **log_data)
        elif severity == 'error':
            self.logger.error("Error occurred", **log_data)
        elif severity == 'warning':
            self.logger.warning("Warning occurred", **log_data)
        else:
            self.logger.info("Information", **log_data)
    
    def register_handler(
        self,
        error_type: Type[Exception],
        handler: Callable[[Exception, Optional[ErrorContext]], Dict[str, Any]]
    ) -> None:
        """Register a custom error handler."""
        self._error_handlers[error_type] = handler
    
    # Specific error handlers
    
    def _handle_data_error(
        self,
        error: DataError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle data-related errors."""
        return {
            'error_code': error.error_code or 'DATA_ERROR',
            'message': str(error),
            'category': 'data',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Check data source connectivity and data quality'
        }
    
    def _handle_validation_error(
        self,
        error: ValidationError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle validation errors."""
        return {
            'error_code': error.error_code or 'VALIDATION_ERROR',
            'message': str(error),
            'category': 'validation',
            'severity': 'warning',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Verify input parameters and data format'
        }
    
    def _handle_optimization_error(
        self,
        error: OptimizationError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle optimization errors."""
        return {
            'error_code': error.error_code or 'OPTIMIZATION_ERROR',
            'message': str(error),
            'category': 'optimization',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Review optimization constraints and parameters'
        }
    
    def _handle_backtest_error(
        self,
        error: BacktestError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle backtesting errors."""
        return {
            'error_code': error.error_code or 'BACKTEST_ERROR',
            'message': str(error),
            'category': 'backtesting',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Check strategy parameters and historical data'
        }
    
    def _handle_risk_error(
        self,
        error: RiskError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle risk management errors."""
        return {
            'error_code': error.error_code or 'RISK_ERROR',
            'message': str(error),
            'category': 'risk',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Review risk limits and portfolio composition'
        }
    
    def _handle_ml_error(
        self,
        error: MLError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle machine learning errors."""
        return {
            'error_code': error.error_code or 'ML_ERROR',
            'message': str(error),
            'category': 'machine_learning',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Check model parameters and training data'
        }
    
    def _handle_configuration_error(
        self,
        error: ConfigurationError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle configuration errors."""
        return {
            'error_code': error.error_code or 'CONFIGURATION_ERROR',
            'message': str(error),
            'category': 'configuration',
            'severity': 'critical',
            'timestamp': datetime.now().isoformat(),
            'recoverable': False,
            'suggested_action': 'Review configuration files and environment variables'
        }
    
    def _handle_infrastructure_error(
        self,
        error: InfrastructureError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle infrastructure errors."""
        return {
            'error_code': error.error_code or 'INFRASTRUCTURE_ERROR',
            'message': str(error),
            'category': 'infrastructure',
            'severity': 'critical',
            'timestamp': datetime.now().isoformat(),
            'recoverable': False,
            'suggested_action': 'Check system resources and external services'
        }
    
    def _handle_resource_error(
        self,
        error: ResourceError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle resource errors."""
        return {
            'error_code': error.error_code or 'RESOURCE_ERROR',
            'message': str(error),
            'category': 'resource',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Check resource availability and limits'
        }
    
    def _handle_business_logic_error(
        self,
        error: BusinessLogicError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle business logic errors."""
        return {
            'error_code': error.error_code or 'BUSINESS_LOGIC_ERROR',
            'message': str(error),
            'category': 'business_logic',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Review business rules and operation sequence'
        }
    
    def _handle_concurrency_error(
        self,
        error: ConcurrencyError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle concurrency errors."""
        return {
            'error_code': error.error_code or 'CONCURRENCY_ERROR',
            'message': str(error),
            'category': 'concurrency',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Retry operation or check for resource conflicts'
        }
    
    def _handle_framework_error(
        self,
        error: QuantFrameworkError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle generic framework errors."""
        return {
            'error_code': error.error_code or 'FRAMEWORK_ERROR',
            'message': str(error),
            'category': 'framework',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'context': error.context
        }
    
    def _handle_value_error(
        self,
        error: ValueError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle ValueError."""
        return {
            'error_code': 'VALUE_ERROR',
            'message': str(error),
            'category': 'validation',
            'severity': 'warning',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Check input values and types'
        }
    
    def _handle_type_error(
        self,
        error: TypeError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle TypeError."""
        return {
            'error_code': 'TYPE_ERROR',
            'message': str(error),
            'category': 'validation',
            'severity': 'warning',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Check parameter types and method signatures'
        }
    
    def _handle_key_error(
        self,
        error: KeyError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle KeyError."""
        return {
            'error_code': 'KEY_ERROR',
            'message': f"Missing key: {str(error)}",
            'category': 'data',
            'severity': 'warning',
            'timestamp': datetime.now().isoformat(),
            'recoverable': True,
            'suggested_action': 'Check data structure and required fields'
        }
    
    def _handle_attribute_error(
        self,
        error: AttributeError,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle AttributeError."""
        return {
            'error_code': 'ATTRIBUTE_ERROR',
            'message': str(error),
            'category': 'programming',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': False,
            'suggested_action': 'Check object attributes and method names'
        }
    
    def _handle_generic_error(
        self,
        error: Exception,
        context: Optional[ErrorContext]
    ) -> Dict[str, Any]:
        """Handle generic exceptions."""
        return {
            'error_code': 'GENERIC_ERROR',
            'message': str(error),
            'category': 'unknown',
            'severity': 'error',
            'timestamp': datetime.now().isoformat(),
            'recoverable': False,
            'error_type': type(error).__name__
        }


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def set_error_handler(handler: ErrorHandler) -> None:
    """Set global error handler instance."""
    global _global_error_handler
    _global_error_handler = handler


# Decorators for error handling

def handle_errors(
    operation: str,
    reraise: bool = False,
    context_factory: Optional[Callable[..., ErrorContext]] = None
):
    """
    Decorator for automatic error handling.
    
    Args:
        operation: Operation name for context
        reraise: Whether to reraise exceptions after handling
        context_factory: Function to create error context from function args
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create error context
                context = None
                if context_factory:
                    try:
                        context = context_factory(*args, **kwargs)
                    except Exception:
                        pass  # Ignore context creation errors
                
                if context is None:
                    context = ErrorContext(operation=operation)
                
                # Handle the error
                error_handler = get_error_handler()
                error_response = error_handler.handle_error(e, context, reraise)
                
                if not reraise:
                    return error_response
        
        return wrapper
    return decorator


def handle_async_errors(
    operation: str,
    reraise: bool = False,
    context_factory: Optional[Callable[..., ErrorContext]] = None
):
    """
    Decorator for automatic error handling in async functions.
    
    Args:
        operation: Operation name for context
        reraise: Whether to reraise exceptions after handling
        context_factory: Function to create error context from function args
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Create error context
                context = None
                if context_factory:
                    try:
                        context = context_factory(*args, **kwargs)
                    except Exception:
                        pass  # Ignore context creation errors
                
                if context is None:
                    context = ErrorContext(operation=operation)
                
                # Handle the error
                error_handler = get_error_handler()
                error_response = error_handler.handle_error(e, context, reraise)
                
                if not reraise:
                    return error_response
        
        return wrapper
    return decorator


# Context manager for error handling
class ErrorHandlingContext:
    """Context manager for error handling within a block."""
    
    def __init__(
        self,
        operation: str,
        reraise: bool = True,
        suppress: bool = False,
        context: Optional[ErrorContext] = None
    ):
        self.operation = operation
        self.reraise = reraise
        self.suppress = suppress
        self.context = context or ErrorContext(operation=operation)
        self.error_response = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_handler = get_error_handler()
            self.error_response = error_handler.handle_error(
                exc_val, self.context, reraise=False
            )
            
            if self.suppress:
                return True  # Suppress the exception
            elif not self.reraise:
                return True  # Suppress and return error response
        
        return False  # Let exception propagate