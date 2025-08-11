"""
Structured logging framework with multiple handlers and formatters.
Provides comprehensive logging capabilities for the quantitative framework.
"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager

from ...domain.interfaces import ILogger
from ...domain.exceptions import ConfigurationError


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from the log record
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                        'filename', 'module', 'lineno', 'funcName', 'created',
                        'msecs', 'relativeCreated', 'thread', 'threadName',
                        'processName', 'process', 'getMessage', 'exc_info',
                        'exc_text', 'stack_info']
        }
        
        if extra_fields:
            log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str)


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with color coding for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors for console output."""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the message
        formatted = super().format(record)
        
        # Add color to the level name
        formatted = formatted.replace(
            record.levelname,
            f"{color}{record.levelname}{reset}"
        )
        
        return formatted


class QuantFrameworkLogger(ILogger):
    """
    Main logger implementation for the quantitative framework.
    Provides structured logging with multiple handlers and formatters.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize logger with configuration.
        
        Args:
            name: Logger name
            config: Logger configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self._logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger with handlers and formatters."""
        # Clear existing handlers
        self._logger.handlers.clear()
        
        # Set log level
        level = self.config.get('level', 'INFO')
        self._logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent propagation to root logger
        self._logger.propagate = False
        
        # Setup console handler
        if self.config.get('console_output', True):
            self._setup_console_handler()
        
        # Setup file handler
        file_path = self.config.get('file_path')
        if file_path:
            self._setup_file_handler(file_path)
        
        # Setup structured log handler (JSON file)
        structured_path = self.config.get('structured_file_path')
        if structured_path:
            self._setup_structured_handler(structured_path)
    
    def _setup_console_handler(self) -> None:
        """Setup console handler with colored output."""
        handler = logging.StreamHandler(sys.stdout)
        
        # Use colored formatter for console
        formatter = ColoredConsoleFormatter(
            fmt=self.config.get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
    
    def _setup_file_handler(self, file_path: str) -> None:
        """Setup rotating file handler."""
        # Create log directory if it doesn't exist
        log_file = Path(file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=self.config.get('max_file_size', 10485760),  # 10MB
            backupCount=self.config.get('backup_count', 5)
        )
        
        # Use standard formatter for file
        formatter = logging.Formatter(
            fmt=self.config.get('format',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
    
    def _setup_structured_handler(self, file_path: str) -> None:
        """Setup structured JSON log handler."""
        # Create log directory if it doesn't exist
        log_file = Path(file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup rotating file handler for structured logs
        handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=self.config.get('max_file_size', 10485760),  # 10MB
            backupCount=self.config.get('backup_count', 5)
        )
        
        # Use structured formatter
        formatter = StructuredFormatter()
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, extra=kwargs)
    
    @contextmanager
    def context(self, **context_vars):
        """Context manager for adding context to all log messages."""
        # Store original extra
        original_extra = getattr(self._logger, '_context_extra', {})
        
        # Add new context
        new_extra = {**original_extra, **context_vars}
        self._logger._context_extra = new_extra
        
        try:
            yield
        finally:
            # Restore original context
            self._logger._context_extra = original_extra
    
    def bind(self, **kwargs) -> 'BoundLogger':
        """Create a bound logger with additional context."""
        return BoundLogger(self, kwargs)


class BoundLogger:
    """Logger bound with additional context variables."""
    
    def __init__(self, logger: QuantFrameworkLogger, context: Dict[str, Any]):
        self._logger = logger
        self._context = context
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with bound context."""
        self._logger.debug(message, **{**self._context, **kwargs})
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message with bound context."""
        self._logger.info(message, **{**self._context, **kwargs})
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with bound context."""
        self._logger.warning(message, **{**self._context, **kwargs})
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message with bound context."""
        self._logger.error(message, **{**self._context, **kwargs})
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message with bound context."""
        self._logger.critical(message, **{**self._context, **kwargs})
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with bound context."""
        self._logger.exception(message, **{**self._context, **kwargs})


class LoggerFactory:
    """Factory for creating and managing logger instances."""
    
    _loggers: Dict[str, QuantFrameworkLogger] = {}
    _default_config: Dict[str, Any] = {}
    
    @classmethod
    def configure(cls, config: Dict[str, Any]) -> None:
        """Configure default logger settings."""
        cls._default_config = config
    
    @classmethod
    def get_logger(cls, name: str, config: Optional[Dict[str, Any]] = None) -> QuantFrameworkLogger:
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name
            config: Optional logger-specific configuration
            
        Returns:
            Logger instance
        """
        if name not in cls._loggers:
            # Merge default config with specific config
            logger_config = {**cls._default_config}
            if config:
                logger_config.update(config)
            
            cls._loggers[name] = QuantFrameworkLogger(name, logger_config)
        
        return cls._loggers[name]
    
    @classmethod
    def shutdown(cls) -> None:
        """Shutdown all loggers and handlers."""
        for logger in cls._loggers.values():
            for handler in logger._logger.handlers:
                handler.close()
        cls._loggers.clear()


# Convenience functions
def get_logger(name: str, config: Optional[Dict[str, Any]] = None) -> QuantFrameworkLogger:
    """Get a logger instance."""
    return LoggerFactory.get_logger(name, config)


def configure_logging(config: Dict[str, Any]) -> None:
    """Configure default logging settings."""
    LoggerFactory.configure(config)


def shutdown_logging() -> None:
    """Shutdown all logging."""
    LoggerFactory.shutdown()


# Performance logging decorator
def log_performance(logger: Optional[QuantFrameworkLogger] = None):
    """Decorator to log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            log = logger or get_logger(func.__module__)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                log.info(
                    f"Function {func.__name__} executed successfully",
                    function=func.__name__,
                    execution_time=execution_time,
                    module=func.__module__
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                log.error(
                    f"Function {func.__name__} failed",
                    function=func.__name__,
                    execution_time=execution_time,
                    module=func.__module__,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator