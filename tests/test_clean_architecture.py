"""
Tests for the clean architecture foundation.
"""
import pytest
from src.domain import (
    Portfolio, Position, Strategy, Price, Return, Signal,
    QuantFrameworkError, ValidationError
)
from src.domain.interfaces import (
    IDataManager, IFeatureEngine, IStrategy, IBacktestEngine,
    IPortfolioOptimizer, IRiskManager, IMLFramework
)
from src.infrastructure import DIContainer, get_container, configure_container
from decimal import Decimal
from datetime import datetime


def test_domain_entities():
    """Test that domain entities can be created and used correctly."""
    # Create a position
    position = Position(
        symbol="AAPL",
        quantity=Decimal("10"),
        average_cost=Decimal("150.00"),
        current_price=Decimal("155.00")
    )
    
    assert position.symbol == "AAPL"
    assert position.quantity == Decimal("10")
    assert position.market_value == Decimal("1550.00")
    assert position.unrealized_pnl == Decimal("50.00")
    
    # Create a portfolio
    portfolio = Portfolio(
        id="test-portfolio",
        name="Test Portfolio",
        cash=Decimal("1000.00")
    )
    
    portfolio.add_position(position)
    assert "AAPL" in portfolio.positions
    assert portfolio.total_value == Decimal("2550.00")  # 1550 + 1000
    
    # Test weights
    weights = portfolio.get_weights()
    assert weights["AAPL"] == pytest.approx(0.6078, 0.001)  # 1550/2550
    assert weights["CASH"] == pytest.approx(0.3922, 0.001)  # 1000/2550


def test_value_objects():
    """Test that value objects can be created and used correctly."""
    # Create a price
    price = Price(
        value=Decimal("155.00"),
        currency="USD",
        timestamp=datetime.now()
    )
    
    # Test immutability
    with pytest.raises(AttributeError):
        price.value = Decimal("160.00")
    
    # Test validation
    with pytest.raises(ValidationError):
        Price(value=Decimal("-10.00"), currency="USD", timestamp=datetime.now())


def test_exception_hierarchy():
    """Test that the exception hierarchy works correctly."""
    # Base exception
    base_error = QuantFrameworkError("Base error")
    assert isinstance(base_error, Exception)
    
    # Derived exception
    validation_error = ValidationError("Validation error")
    assert isinstance(validation_error, QuantFrameworkError)
    assert isinstance(validation_error, Exception)


def test_dependency_injection():
    """Test that the dependency injection container works correctly."""
    container = DIContainer()
    
    # Register a mock implementation
    class MockDataManager:
        def __init__(self):
            self.initialized = True
    
    container.register_singleton(IDataManager, MockDataManager)
    
    # Resolve the implementation
    data_manager = container.resolve(IDataManager)
    assert isinstance(data_manager, MockDataManager)
    assert data_manager.initialized
    
    # Test singleton behavior
    data_manager2 = container.resolve(IDataManager)
    assert data_manager is data_manager2  # Same instance


def test_interfaces():
    """Test that interfaces are properly defined."""
    # Check that interfaces have abstract methods
    assert hasattr(IDataManager, "fetch_market_data")
    assert hasattr(IFeatureEngine, "compute_features")
    assert hasattr(IStrategy, "generate_signals")
    assert hasattr(IBacktestEngine, "run_backtest")
    assert hasattr(IPortfolioOptimizer, "optimize")
    assert hasattr(IRiskManager, "calculate_portfolio_risk")
    assert hasattr(IMLFramework, "train_model")