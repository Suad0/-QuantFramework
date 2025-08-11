"""
Tests for the advanced backtesting engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from src.infrastructure.backtesting import (
    BacktestEngine, OrderManager, Order, OrderType, OrderSide,
    TradingCostSimulator, CorporateActionHandler, BiasDetector,
    PerformanceAnalyzer
)
from src.domain.value_objects import Signal, SignalType
from src.domain.interfaces import IStrategy


class MockStrategy(IStrategy):
    """Mock strategy for testing."""
    
    def __init__(self):
        self.parameters = {'lookback': 20, 'threshold': 0.02}
    
    def generate_signals(self, data: pd.DataFrame, context: dict) -> list:
        """Generate simple momentum signals."""
        signals = []
        
        if len(data) < self.parameters['lookback']:
            return signals
        
        # Simple momentum strategy
        for symbol in data.columns:
            if symbol == 'date':
                continue
                
            prices = data[symbol].dropna()
            if len(prices) < self.parameters['lookback']:
                continue
            
            # Calculate momentum
            current_price = prices.iloc[-1]
            past_price = prices.iloc[-self.parameters['lookback']]
            momentum = (current_price - past_price) / past_price
            
            # Generate signal
            if momentum > self.parameters['threshold']:
                signal = Signal(
                    symbol=symbol,
                    timestamp=context['current_date'],
                    signal_type=SignalType.BUY,
                    strength=min(momentum * 5, 1.0),  # Scale momentum to strength
                    confidence=0.7,
                    source='momentum_strategy',
                    metadata={}
                )
                signals.append(signal)
            elif momentum < -self.parameters['threshold']:
                signal = Signal(
                    symbol=symbol,
                    timestamp=context['current_date'],
                    signal_type=SignalType.SELL,
                    strength=min(abs(momentum) * 5, 1.0),
                    confidence=0.7,
                    source='momentum_strategy',
                    metadata={}
                )
                signals.append(signal)
        
        return signals
    
    def get_parameters(self) -> dict:
        return self.parameters.copy()
    
    def validate_signals(self, signals: list) -> dict:
        return {'valid': True, 'warnings': []}
    
    def update_parameters(self, new_parameters: dict) -> None:
        self.parameters.update(new_parameters)


def create_sample_market_data():
    """Create sample market data for testing."""
    
    # Create 100 days of sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Create price data for 3 symbols with different patterns
    np.random.seed(42)  # For reproducible tests
    
    # Symbol A: trending up
    prices_a = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
    
    # Symbol B: trending down
    prices_b = 100 * np.cumprod(1 + np.random.normal(-0.001, 0.02, 100))
    
    # Symbol C: sideways
    prices_c = 100 + np.cumsum(np.random.normal(0, 1, 100))
    
    market_data = pd.DataFrame({
        'AAPL': prices_a,
        'MSFT': prices_b,
        'GOOGL': prices_c
    }, index=dates)
    
    return market_data


class TestOrderManager:
    """Test the order management system."""
    
    def test_order_creation(self):
        """Test basic order creation."""
        order = Order(
            id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('100')
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal('100')
        assert order.remaining_quantity == Decimal('100')
        assert order.is_active
        assert not order.is_filled
    
    def test_order_execution(self):
        """Test order execution."""
        order = Order(
            id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('100')
        )
        
        # Fill the order
        order.fill(Decimal('100'), Decimal('150.00'), Decimal('1.00'))
        
        assert order.is_filled
        assert order.filled_quantity == Decimal('100')
        assert order.filled_price == Decimal('150.00')
        assert order.commission == Decimal('1.00')
    
    def test_order_manager(self):
        """Test order manager functionality."""
        manager = OrderManager()
        
        order = Order(
            id="",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('100')
        )
        
        order_id = manager.submit_order(order)
        assert order_id is not None
        assert order_id in manager.orders
        
        # Test market data processing
        market_data = pd.Series({'AAPL': 150.00})
        executed_orders = manager.process_market_data(market_data, datetime.now())
        
        assert len(executed_orders) == 1
        assert executed_orders[0].is_filled


class TestTradingCostSimulator:
    """Test the trading cost simulation."""
    
    def test_commission_calculation(self):
        """Test commission calculation."""
        simulator = TradingCostSimulator()
        
        order = Order(
            id="TEST_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('100')
        )
        order.fill(Decimal('100'), Decimal('150.00'))
        
        market_data = create_sample_market_data()
        timestamp = market_data.index[0]
        
        costs = simulator.calculate_trading_costs(
            order, Decimal('150.00'), market_data, timestamp
        )
        
        assert costs.commission > 0
        assert costs.total_cost > 0


class TestPerformanceAnalyzer:
    """Test the performance analysis."""
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        analyzer = PerformanceAnalyzer()
        
        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year
        
        metrics = analyzer.calculate_performance_metrics(returns)
        
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'annualized_return')
        assert hasattr(metrics, 'volatility')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        
        # Check that metrics are reasonable
        assert -1 < metrics.total_return < 5  # Total return between -100% and 500%
        assert 0 < metrics.volatility < 2     # Volatility between 0% and 200%


class TestBiasDetector:
    """Test the bias detection system."""
    
    def test_bias_detection(self):
        """Test bias detection functionality."""
        detector = BiasDetector()
        
        # Create mock backtest results
        returns = pd.Series(np.random.normal(0.002, 0.02, 252))  # Slightly positive returns
        
        backtest_results = {
            'returns': returns,
            'trades': [
                {'timestamp': datetime.now(), 'symbol': 'AAPL', 'pnl': 100},
                {'timestamp': datetime.now(), 'symbol': 'MSFT', 'pnl': -50}
            ],
            'start_date': datetime.now() - timedelta(days=365),
            'end_date': datetime.now(),
            'symbols': ['AAPL', 'MSFT'],
            'performance_metrics': {
                'sharpe_ratio': 1.5,
                'win_rate': 0.6
            }
        }
        
        market_data = create_sample_market_data()
        strategy_config = {'parameters': {'lookback': 20}}
        
        analysis = detector.analyze_backtest_for_bias(
            backtest_results, market_data, strategy_config
        )
        
        assert hasattr(analysis, 'warnings')
        assert hasattr(analysis, 'overall_risk_score')
        assert hasattr(analysis, 'is_reliable')
        assert 0 <= analysis.overall_risk_score <= 1


class TestBacktestEngine:
    """Test the main backtesting engine."""
    
    @pytest.mark.asyncio
    async def test_basic_backtest(self):
        """Test basic backtesting functionality."""
        engine = BacktestEngine()
        strategy = MockStrategy()
        
        market_data = create_sample_market_data()
        
        config = {
            'start_date': market_data.index[20],  # Start after lookback period
            'end_date': market_data.index[-1],
            'initial_capital': 100000,
            'market_data': market_data,
            'enable_corporate_actions': False,
            'enable_bias_detection': False
        }
        
        result = await engine.run_backtest(strategy, config)
        
        # Check basic result structure
        assert 'config' in result
        assert 'performance' in result
        assert 'trading' in result
        assert 'portfolio_values' in result
        assert 'returns' in result
        
        # Check that some trading occurred
        assert result['trading']['total_trades'] >= 0
        
        # Check performance metrics exist
        perf = result['performance']
        assert 'total_return' in perf
        assert 'sharpe_ratio' in perf
        assert 'max_drawdown' in perf
    
    def test_signal_simulation(self):
        """Test signal-based trading simulation."""
        engine = BacktestEngine()
        
        # Create test signals
        signals = [
            Signal(
                symbol='AAPL',
                timestamp=datetime.now(),
                signal_type=SignalType.BUY,
                strength=0.8,
                confidence=0.9,
                source='test',
                metadata={}
            ),
            Signal(
                symbol='MSFT',
                timestamp=datetime.now(),
                signal_type=SignalType.SELL,
                strength=0.6,
                confidence=0.7,
                source='test',
                metadata={}
            )
        ]
        
        market_data = create_sample_market_data()
        
        # Set up basic config for simulation
        engine.config = engine._parse_config({
            'start_date': datetime.now(),
            'end_date': datetime.now(),
            'initial_capital': 100000
        })
        
        result = engine.simulate_trading(signals, market_data)
        
        assert 'trades' in result
        assert 'final_portfolio_value' in result
        assert 'total_return' in result
        assert len(result['trades']) == len(signals)


if __name__ == "__main__":
    # Run a simple test
    import asyncio
    
    async def run_test():
        engine = BacktestEngine()
        strategy = MockStrategy()
        market_data = create_sample_market_data()
        
        config = {
            'start_date': market_data.index[20],
            'end_date': market_data.index[-1],
            'initial_capital': 100000,
            'market_data': market_data,
            'enable_corporate_actions': False,
            'enable_bias_detection': False
        }
        
        result = await engine.run_backtest(strategy, config)
        print("Backtest completed successfully!")
        print(f"Total Return: {result['performance']['total_return']:.2%}")
        print(f"Sharpe Ratio: {result['performance']['sharpe_ratio']:.2f}")
        print(f"Total Trades: {result['trading']['total_trades']}")
    
    asyncio.run(run_test())