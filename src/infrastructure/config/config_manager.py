"""
Configuration management system with environment-based configurations.
Provides centralized configuration loading and management.
"""

import os
import yaml
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from ...domain.exceptions import ConfigurationError, MissingConfigurationError, InvalidConfigurationError


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = "localhost"
    port: int = 5432
    name: str = "quantframework"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class DataSourceConfig:
    """Data source configuration settings."""
    yahoo_finance: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "rate_limit": 2000,
        "timeout": 30
    })
    alpha_vantage: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "api_key": "",
        "rate_limit": 500
    })
    iex_cloud: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "api_key": "",
        "base_url": "https://cloud.iexapis.com/stable"
    })
    quandl: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "api_key": "",
        "base_url": "https://www.quandl.com/api/v3"
    })


@dataclass
class RiskLimitsConfig:
    """Risk management configuration settings."""
    max_position_size: float = 0.05
    max_sector_exposure: float = 0.20
    var_limit: float = 0.02
    max_drawdown_limit: float = 0.15
    concentration_limit: float = 0.10
    leverage_limit: float = 1.0


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/quantframework.log"
    structured_file_path: str = "logs/quantframework.json"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    console_output: bool = True


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    enabled: bool = True
    backend: str = "memory"  # memory, redis, file
    ttl: int = 3600  # seconds
    max_size: int = 1000
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0


@dataclass
class ApplicationConfig:
    """Main application configuration."""
    environment: str = "development"
    debug: bool = False
    secret_key: str = ""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
    risk_limits: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)


class ConfigManager:
    """
    Configuration manager that loads and manages application configuration
    from multiple sources with environment-based overrides.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self._config: Dict[str, Any] = {}
        self._app_config: Optional[ApplicationConfig] = None
        self._environment = os.getenv("ENVIRONMENT", "development")
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from files and environment variables."""
        try:
            # Load base configuration
            base_config = self._load_config_file("base.yaml")
            
            # Load environment-specific configuration
            env_config = self._load_config_file(f"{self._environment}.yaml")
            
            # Merge configurations (environment overrides base)
            self._config = self._deep_merge(base_config, env_config)
            
            # Override with environment variables
            self._apply_environment_overrides()
            
            # Create application config object
            self._app_config = self._create_app_config()
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {str(e)}",
                error_code="CONFIG_LOAD_FAILED"
            )
    
    def _load_config_file(self, filename: str) -> Dict[str, Any]:
        """Load configuration from a YAML or JSON file."""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            if filename.startswith("base."):
                # Base config is required
                raise MissingConfigurationError(
                    f"Base configuration file not found: {file_path}",
                    error_code="BASE_CONFIG_MISSING"
                )
            else:
                # Environment-specific config is optional
                return {}
        
        try:
            with open(file_path, 'r') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    return yaml.safe_load(f) or {}
                elif filename.endswith('.json'):
                    return json.load(f) or {}
                else:
                    raise InvalidConfigurationError(
                        f"Unsupported configuration file format: {filename}",
                        error_code="UNSUPPORTED_CONFIG_FORMAT"
                    )
        except yaml.YAMLError as e:
            raise InvalidConfigurationError(
                f"Invalid YAML in configuration file {filename}: {str(e)}",
                error_code="INVALID_YAML"
            )
        except json.JSONDecodeError as e:
            raise InvalidConfigurationError(
                f"Invalid JSON in configuration file {filename}: {str(e)}",
                error_code="INVALID_JSON"
            )
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Database overrides
        if os.getenv("DB_HOST"):
            self._config.setdefault("database", {})["host"] = os.getenv("DB_HOST")
        if os.getenv("DB_PORT"):
            self._config.setdefault("database", {})["port"] = int(os.getenv("DB_PORT"))
        if os.getenv("DB_NAME"):
            self._config.setdefault("database", {})["name"] = os.getenv("DB_NAME")
        if os.getenv("DB_USER"):
            self._config.setdefault("database", {})["user"] = os.getenv("DB_USER")
        if os.getenv("DB_PASSWORD"):
            self._config.setdefault("database", {})["password"] = os.getenv("DB_PASSWORD")
        
        # API Keys
        if os.getenv("ALPHA_VANTAGE_KEY"):
            self._config.setdefault("data_sources", {}).setdefault("alpha_vantage", {})["api_key"] = os.getenv("ALPHA_VANTAGE_KEY")
        if os.getenv("IEX_CLOUD_KEY"):
            self._config.setdefault("data_sources", {}).setdefault("iex_cloud", {})["api_key"] = os.getenv("IEX_CLOUD_KEY")
        
        # Application settings
        if os.getenv("SECRET_KEY"):
            self._config["secret_key"] = os.getenv("SECRET_KEY")
        if os.getenv("DEBUG"):
            self._config["debug"] = os.getenv("DEBUG").lower() in ("true", "1", "yes")
        if os.getenv("LOG_LEVEL"):
            self._config.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")
    
    def _create_app_config(self) -> ApplicationConfig:
        """Create ApplicationConfig object from loaded configuration."""
        try:
            # Database config
            db_config = DatabaseConfig(**self._config.get("database", {}))
            
            # Data sources config
            ds_config = DataSourceConfig(**self._config.get("data_sources", {}))
            
            # Risk limits config
            risk_config = RiskLimitsConfig(**self._config.get("risk_limits", {}))
            
            # Logging config
            log_config = LoggingConfig(**self._config.get("logging", {}))
            
            # Cache config
            cache_config = CacheConfig(**self._config.get("cache", {}))
            
            # Main application config
            app_config = ApplicationConfig(
                environment=self._environment,
                debug=self._config.get("debug", False),
                secret_key=self._config.get("secret_key", ""),
                database=db_config,
                data_sources=ds_config,
                risk_limits=risk_config,
                logging=log_config,
                cache=cache_config
            )
            
            return app_config
            
        except Exception as e:
            raise InvalidConfigurationError(
                f"Failed to create application configuration: {str(e)}",
                error_code="CONFIG_CREATION_FAILED"
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key using dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key using dot notation.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        
        # Recreate application config
        self._app_config = self._create_app_config()
    
    def get_app_config(self) -> ApplicationConfig:
        """Get the application configuration object."""
        if self._app_config is None:
            raise ConfigurationError(
                "Application configuration not initialized",
                error_code="CONFIG_NOT_INITIALIZED"
            )
        return self._app_config
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._load_configuration()
    
    def validate(self) -> None:
        """Validate configuration settings."""
        config = self.get_app_config()
        
        # Validate required settings
        if not config.secret_key and config.environment == "production":
            raise InvalidConfigurationError(
                "Secret key is required in production environment",
                error_code="MISSING_SECRET_KEY"
            )
        
        # Validate database settings
        if not config.database.host:
            raise InvalidConfigurationError(
                "Database host is required",
                error_code="MISSING_DB_HOST"
            )
        
        # Validate risk limits
        if config.risk_limits.max_position_size <= 0 or config.risk_limits.max_position_size > 1:
            raise InvalidConfigurationError(
                "Max position size must be between 0 and 1",
                error_code="INVALID_POSITION_SIZE"
            )
        
        # Validate data source API keys if enabled
        if config.data_sources.alpha_vantage.get("enabled") and not config.data_sources.alpha_vantage.get("api_key"):
            raise InvalidConfigurationError(
                "Alpha Vantage API key is required when service is enabled",
                error_code="MISSING_ALPHA_VANTAGE_KEY"
            )
        
        if config.data_sources.iex_cloud.get("enabled") and not config.data_sources.iex_cloud.get("api_key"):
            raise InvalidConfigurationError(
                "IEX Cloud API key is required when service is enabled",
                error_code="MISSING_IEX_CLOUD_KEY"
            )


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def initialize_config(config_dir: Optional[str] = None) -> ConfigManager:
    """Initialize global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager(config_dir)
    return _config_manager