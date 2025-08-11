"""
Abstract base classes and interfaces for the quantitative framework.
These define contracts that implementations must follow.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pandas import DataFrame, Series
from .entities import Portfolio, Strategy, Signal, Position
from .value_objects import Price, Return, RiskMetrics, PerformanceMetrics


class IDataManager(ABC):
    """Interface for data management operations"""
    
    @abstractmethod
    async def fetch_market_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> DataFrame:
        """Fetch market data for given symbols and date range"""
        pass
    
    @abstractmethod
    async def get_cached_data(self, cache_key: str) -> Optional[DataFrame]:
        """Retrieve cached data by key"""
        pass
    
    @abstractmethod
    async def store_data(self, data: DataFrame, metadata: Dict[str, Any]) -> None:
        """Store data with associated metadata"""
        pass
    
    @abstractmethod
    async def validate_data_quality(self, data: DataFrame) -> Dict[str, Any]:
        """Validate data quality and return quality report"""
        pass


class IFeatureEngine(ABC):
    """Interface for feature engineering operations"""
    
    @abstractmethod
    def register_indicator(self, name: str, indicator: 'IIndicator') -> None:
        """Register a new technical indicator"""
        pass
    
    @abstractmethod
    def compute_features(
        self, 
        data: DataFrame, 
        feature_config: Dict[str, Any]
    ) -> DataFrame:
        """Compute features based on configuration"""
        pass
    
    @abstractmethod
    def validate_features(self, features: DataFrame) -> Dict[str, Any]:
        """Validate computed features"""
        pass


class IIndicator(ABC):
    """Interface for technical indicators"""
    
    @abstractmethod
    def calculate(self, data: DataFrame, **kwargs) -> Series:
        """Calculate indicator values"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get indicator parameters"""
        pass


class IStrategy(ABC):
    """Interface for trading strategies"""
    
    @abstractmethod
    def generate_signals(
        self, 
        data: DataFrame, 
        context: Dict[str, Any]
    ) -> List[Signal]:
        """Generate trading signals based on data and context"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        pass
    
    @abstractmethod
    def validate_signals(self, signals: List[Signal]) -> Dict[str, Any]:
        """Validate generated signals"""
        pass


class IBacktestEngine(ABC):
    """Interface for backtesting operations"""
    
    @abstractmethod
    async def run_backtest(
        self, 
        strategy: IStrategy, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run backtest for given strategy and configuration"""
        pass
    
    @abstractmethod
    def simulate_trading(
        self, 
        signals: List[Signal], 
        market_data: DataFrame
    ) -> Dict[str, Any]:
        """Simulate trading based on signals"""
        pass
    
    @abstractmethod
    def calculate_performance_metrics(
        self, 
        returns: Series
    ) -> PerformanceMetrics:
        """Calculate performance metrics from returns"""
        pass


class IPortfolioOptimizer(ABC):
    """Interface for portfolio optimization"""
    
    @abstractmethod
    def optimize(
        self, 
        expected_returns: Series, 
        covariance_matrix: DataFrame,
        constraints: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize portfolio weights"""
        pass
    
    @abstractmethod
    def rebalance(
        self, 
        current_portfolio: Portfolio, 
        target_weights: Series
    ) -> Dict[str, Any]:
        """Calculate rebalancing trades"""
        pass


class IRiskManager(ABC):
    """Interface for risk management operations"""
    
    @abstractmethod
    def calculate_portfolio_risk(self, portfolio: Portfolio) -> RiskMetrics:
        """Calculate portfolio risk metrics"""
        pass
    
    @abstractmethod
    def monitor_risk_limits(self, portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Monitor portfolio against risk limits"""
        pass
    
    @abstractmethod
    def stress_test(
        self, 
        portfolio: Portfolio, 
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform stress testing on portfolio"""
        pass


class IMLFramework(ABC):
    """Interface for machine learning operations"""
    
    @abstractmethod
    def train_model(
        self, 
        features: DataFrame, 
        targets: Series, 
        config: Dict[str, Any]
    ) -> 'IMLModel':
        """Train a machine learning model"""
        pass
    
    @abstractmethod
    def predict(
        self, 
        model: 'IMLModel', 
        features: DataFrame
    ) -> Dict[str, Any]:
        """Make predictions using trained model"""
        pass
    
    @abstractmethod
    def validate_model(
        self, 
        model: 'IMLModel', 
        test_data: DataFrame
    ) -> Dict[str, Any]:
        """Validate model performance"""
        pass


class IMLModel(ABC):
    """Interface for machine learning models"""
    
    @abstractmethod
    def predict(self, features: DataFrame) -> Series:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk"""
        pass


class IRepository(ABC):
    """Base interface for repository pattern"""
    
    @abstractmethod
    async def save(self, entity: Any) -> None:
        """Save entity to storage"""
        pass
    
    @abstractmethod
    async def find_by_id(self, entity_id: str) -> Optional[Any]:
        """Find entity by ID"""
        pass
    
    @abstractmethod
    async def find_all(self) -> List[Any]:
        """Find all entities"""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> None:
        """Delete entity by ID"""
        pass


class IPortfolioRepository(IRepository):
    """Interface for portfolio repository"""
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Portfolio]:
        """Find portfolio by name"""
        pass
    
    @abstractmethod
    async def find_active_portfolios(self) -> List[Portfolio]:
        """Find all active portfolios"""
        pass


class IStrategyRepository(IRepository):
    """Interface for strategy repository"""
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Strategy]:
        """Find strategy by name"""
        pass
    
    @abstractmethod
    async def find_active_strategies(self) -> List[Strategy]:
        """Find all active strategies"""
        pass


class IEventPublisher(ABC):
    """Interface for event publishing"""
    
    @abstractmethod
    async def publish(self, event: Dict[str, Any]) -> None:
        """Publish an event"""
        pass
    
    @abstractmethod
    def subscribe(self, event_type: str, handler: callable) -> None:
        """Subscribe to event type"""
        pass


class ILogger(ABC):
    """Interface for logging operations"""
    
    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        pass
    
    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        pass
    
    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """Log error message"""
        pass
    
    @abstractmethod
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        pass


class IConfigManager(ABC):
    """Interface for configuration management"""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        pass
    
    @abstractmethod
    def load_from_file(self, file_path: str) -> None:
        """Load configuration from file"""
        pass
    
    @abstractmethod
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to file"""
        pass