"""
Advanced Backtesting Engine.

Main backtesting engine that orchestrates all backtesting components:
- Order management and execution
- Trading cost simulation
- Corporate action handling
- Performance analysis
- Bias detection
"""

from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ...domain.interfaces import IBacktestEngine, IStrategy
from ...domain.entities import Portfolio, Position
from ...domain.exceptions import BacktestError, ValidationError
from ...domain.value_objects import Signal, SignalType

from .order_manager import OrderManager, Order, OrderType, OrderSide
from .cost_simulator import TradingCostSimulator, CostModel
from .corporate_actions import CorporateActionHandler
from .performance_analyzer import PerformanceAnalyzer, DetailedPerformanceMetrics
from .bias_detector import BiasDetector


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Basic configuration
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Decimal('100000')
    
    # Trading configuration
    commission_per_share: Decimal = Decimal('0.005')
    commission_percentage: Decimal = Decimal('0.001')
    minimum_commission: Decimal = Decimal('1.0')
    maximum_commission: Decimal = Decimal('50.0')
    
    # Risk management
    max_position_size: float = 0.1  # 10% max position size
    max_leverage: float = 1.0       # No leverage by default
    
    # Execution settings
    slippage_bps: int = 5           # 5 basis points slippage
    market_impact_model: str = "sqrt"  # Square root market impact model
    
    # Analysis settings
    benchmark_symbol: Optional[str] = None
    risk_free_rate: float = 0.02    # 2% risk-free rate
    
    # Advanced settings
    enable_corporate_actions: bool = True
    enable_bias_detection: bool = True
    enable_walk_forward: bool = False
    
    # Output settings
    save_trades: bool = True
    save_positions: bool = True
    detailed_logging: bool = False


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    
    # Basic results
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Portfolio results
    initial_capital: Decimal
    final_capital: Decimal
    total_return: float
    
    # Performance metrics
    performance_metrics: DetailedPerformanceMetrics
    
    # Trading results
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades: List[Dict[str, Any]]
    
    # Time series data
    portfolio_values: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    
    # Analysis results
    bias_analysis: Optional[Dict[str, Any]] = None
    corporate_actions: List[Dict[str, Any]] = None
    
    # Metadata
    execution_time: float = 0.0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class BacktestEngine(IBacktestEngine):
    """Advanced backtesting engine with comprehensive features."""
    
    def __init__(self):
        self.order_manager = OrderManager()
        self.cost_simulator = TradingCostSimulator()
        self.corporate_action_handler = CorporateActionHandler()
        self.performance_analyzer = PerformanceAnalyzer()
        self.bias_detector = BiasDetector()
        
        # State tracking
        self.current_portfolio: Optional[Portfolio] = None
        self.market_data: Optional[pd.DataFrame] = None
        self.config: Optional[BacktestConfig] = None
        
        # Results tracking
        self.portfolio_history: List[Dict[str, Any]] = []
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_values: List[Dict[str, Any]] = []
    
    async def run_backtest(
        self,
        strategy: IStrategy,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a complete backtest."""
        
        start_time = datetime.now()
        
        try:
            # Parse configuration
            self.config = self._parse_config(config)
            
            # Initialize backtest
            self._initialize_backtest()
            
            # Load market data
            market_data = config.get('market_data')
            if market_data is None:
                raise BacktestError("Market data is required for backtesting")
            
            self.market_data = market_data
            
            # Load corporate actions if enabled
            if self.config.enable_corporate_actions:
                corporate_actions_data = config.get('corporate_actions')
                if corporate_actions_data is not None:
                    self.corporate_action_handler.load_corporate_actions_from_data(corporate_actions_data)
            
            # Run the main backtest loop
            await self._run_backtest_loop(strategy)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics()
            
            # Run bias detection if enabled
            bias_analysis = None
            if self.config.enable_bias_detection:
                bias_analysis = self._run_bias_detection(strategy)
            
            # Create result object
            result = self._create_backtest_result(performance_metrics, bias_analysis)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            return self._result_to_dict(result)
            
        except Exception as e:
            raise BacktestError(f"Backtest execution failed: {str(e)}")
    
    def simulate_trading(
        self,
        signals: List[Signal],
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Simulate trading based on signals."""
        
        if not signals:
            return {'error': 'No signals provided'}
        
        # Simple simulation without full backtest infrastructure
        trades = []
        portfolio_value = float(self.config.initial_capital) if self.config else 100000.0
        
        for signal in signals:
            # Simple trade simulation
            trade = {
                'timestamp': signal.timestamp,
                'symbol': signal.symbol,
                'signal_type': signal.signal_type.value,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'estimated_pnl': portfolio_value * 0.01 * signal.strength  # Simplified P&L
            }
            trades.append(trade)
            portfolio_value += trade['estimated_pnl']
        
        return {
            'trades': trades,
            'final_portfolio_value': portfolio_value,
            'total_return': (portfolio_value - float(self.config.initial_capital)) / float(self.config.initial_capital) if self.config else 0.0
        }
    
    def calculate_performance_metrics(self, returns: pd.Series) -> DetailedPerformanceMetrics:
        """Calculate performance metrics from returns series."""
        return self.performance_analyzer.calculate_performance_metrics(returns)
    
    def _parse_config(self, config: Dict[str, Any]) -> BacktestConfig:
        """Parse configuration dictionary into BacktestConfig object."""
        
        required_fields = ['start_date', 'end_date']
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Required configuration field missing: {field}")
        
        return BacktestConfig(
            start_date=pd.to_datetime(config['start_date']),
            end_date=pd.to_datetime(config['end_date']),
            initial_capital=Decimal(str(config.get('initial_capital', 100000))),
            commission_per_share=Decimal(str(config.get('commission_per_share', 0.005))),
            commission_percentage=Decimal(str(config.get('commission_percentage', 0.001))),
            minimum_commission=Decimal(str(config.get('minimum_commission', 1.0))),
            maximum_commission=Decimal(str(config.get('maximum_commission', 50.0))),
            max_position_size=config.get('max_position_size', 0.1),
            max_leverage=config.get('max_leverage', 1.0),
            slippage_bps=config.get('slippage_bps', 5),
            market_impact_model=config.get('market_impact_model', 'sqrt'),
            benchmark_symbol=config.get('benchmark_symbol'),
            risk_free_rate=config.get('risk_free_rate', 0.02),
            enable_corporate_actions=config.get('enable_corporate_actions', True),
            enable_bias_detection=config.get('enable_bias_detection', True),
            enable_walk_forward=config.get('enable_walk_forward', False),
            save_trades=config.get('save_trades', True),
            save_positions=config.get('save_positions', True),
            detailed_logging=config.get('detailed_logging', False)
        )
    
    def _initialize_backtest(self) -> None:
        """Initialize backtest state."""
        
        # Create initial portfolio
        self.current_portfolio = Portfolio(
            id="backtest_portfolio",
            name="Backtest Portfolio",
            cash=self.config.initial_capital
        )
        
        # Configure cost simulator
        cost_model = CostModel(
            commission_per_share=self.config.commission_per_share,
            commission_percentage=self.config.commission_percentage,
            minimum_commission=self.config.minimum_commission,
            maximum_commission=self.config.maximum_commission,
            base_slippage_bps=self.config.slippage_bps
        )
        self.cost_simulator = TradingCostSimulator(cost_model)
        
        # Clear previous state
        self.order_manager.clear_orders()
        self.portfolio_history.clear()
        self.trade_history.clear()
        self.daily_values.clear()
    
    async def _run_backtest_loop(self, strategy: IStrategy) -> None:
        """Run the main backtesting loop."""
        
        # Get date range for backtesting
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='D'
        )
        
        for current_date in date_range:
            # Skip weekends (assuming market data is only for trading days)
            if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                continue
            
            # Check if we have market data for this date
            if current_date not in self.market_data.index:
                continue
            
            current_market_data = self.market_data.loc[current_date]
            
            # Process corporate actions
            if self.config.enable_corporate_actions:
                self._process_corporate_actions(current_date.date())
            
            # Update portfolio positions with current prices
            self._update_portfolio_prices(current_market_data)
            
            # Generate signals from strategy
            context = {
                'current_date': current_date,
                'portfolio': self.current_portfolio,
                'market_data': current_market_data
            }
            
            # Get historical data up to current date for strategy
            historical_data = self.market_data[self.market_data.index <= current_date]
            
            signals = strategy.generate_signals(historical_data, context)
            
            # Process signals and create orders
            self._process_signals(signals, current_market_data, current_date)
            
            # Execute orders
            executed_orders = self.order_manager.process_market_data(current_market_data, current_date)
            
            # Process executed orders
            for order in executed_orders:
                self._process_executed_order(order, current_market_data, current_date)
            
            # Record daily portfolio state
            self._record_daily_state(current_date)
    
    def _process_corporate_actions(self, current_date: date) -> None:
        """Process corporate actions for the current date."""
        
        results = self.corporate_action_handler.process_corporate_actions(
            self.current_portfolio,
            current_date,
            self.market_data
        )
        
        # Add cash flows from dividends
        if results['cash_flows'] > 0:
            self.current_portfolio.cash += results['cash_flows']
        
        # Add new positions from spin-offs
        for symbol, position in results['new_positions'].items():
            self.current_portfolio.add_position(position)
    
    def _update_portfolio_prices(self, market_data: pd.Series) -> None:
        """Update portfolio positions with current market prices."""
        
        for symbol, position in self.current_portfolio.positions.items():
            if symbol in market_data.index:
                new_price = Decimal(str(market_data[symbol]))
                position.update_price(new_price)
    
    def _process_signals(
        self,
        signals: List[Signal],
        market_data: pd.Series,
        current_date: datetime
    ) -> None:
        """Process trading signals and create orders."""
        
        for signal in signals:
            if signal.symbol not in market_data.index:
                continue
            
            current_price = Decimal(str(market_data[signal.symbol]))
            
            # Calculate position size based on signal strength and risk management
            position_size = self._calculate_position_size(signal, current_price)
            
            if position_size == 0:
                continue
            
            # Determine order side
            if signal.signal_type == SignalType.BUY and position_size > 0:
                order_side = OrderSide.BUY
                quantity = Decimal(str(position_size))
            elif signal.signal_type == SignalType.SELL and position_size > 0:
                order_side = OrderSide.SELL
                quantity = Decimal(str(position_size))
            else:
                continue  # HOLD or invalid signal
            
            # Create market order (can be extended to support other order types)
            order = Order(
                id="",  # Will be generated by order manager
                symbol=signal.symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=current_date,
                metadata={
                    'signal_strength': signal.strength,
                    'signal_confidence': signal.confidence,
                    'signal_source': signal.source
                }
            )
            
            self.order_manager.submit_order(order)
    
    def _calculate_position_size(self, signal: Signal, current_price: Decimal) -> float:
        """Calculate position size based on signal and risk management rules."""
        
        # Get current portfolio value
        portfolio_value = self.current_portfolio.total_value
        
        if portfolio_value == 0:
            return 0
        
        # Calculate base position size based on signal strength
        base_position_value = float(portfolio_value) * abs(signal.strength) * signal.confidence
        
        # Apply maximum position size constraint
        max_position_value = float(portfolio_value) * self.config.max_position_size
        position_value = min(base_position_value, max_position_value)
        
        # Convert to number of shares
        shares = position_value / float(current_price)
        
        # Check if we have enough cash for buy orders
        if signal.signal_type == SignalType.BUY:
            available_cash = float(self.current_portfolio.cash)
            max_affordable_shares = available_cash / float(current_price)
            shares = min(shares, max_affordable_shares)
        
        # Check if we have enough shares for sell orders
        elif signal.signal_type == SignalType.SELL:
            current_position = self.current_portfolio.positions.get(signal.symbol)
            if current_position:
                max_sellable_shares = float(current_position.quantity)
                shares = min(shares, max_sellable_shares)
            else:
                shares = 0  # Can't sell what we don't own
        
        return max(0, shares)
    
    def _process_executed_order(
        self,
        order: Order,
        market_data: pd.Series,
        current_date: datetime
    ) -> None:
        """Process an executed order and update portfolio."""
        
        # Calculate trading costs
        trading_costs = self.cost_simulator.calculate_trading_costs(
            order,
            order.filled_price,
            self.market_data,
            current_date
        )
        
        # Update portfolio based on order
        if order.side == OrderSide.BUY:
            # Buy order: reduce cash, increase position
            total_cost = order.filled_quantity * order.filled_price + trading_costs.total_cost
            self.current_portfolio.cash -= total_cost
            
            self.current_portfolio.update_position(
                order.symbol,
                order.filled_quantity,
                order.filled_price
            )
        
        elif order.side == OrderSide.SELL:
            # Sell order: increase cash, reduce position
            total_proceeds = order.filled_quantity * order.filled_price - trading_costs.total_cost
            self.current_portfolio.cash += total_proceeds
            
            self.current_portfolio.update_position(
                order.symbol,
                -order.filled_quantity,
                order.filled_price
            )
        
        # Record trade
        if self.config.save_trades:
            trade_record = {
                'timestamp': current_date,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': float(order.filled_quantity),
                'price': float(order.filled_price),
                'commission': float(trading_costs.commission),
                'slippage': float(trading_costs.slippage),
                'market_impact': float(trading_costs.market_impact),
                'total_cost': float(trading_costs.total_cost),
                'signal_strength': order.metadata.get('signal_strength', 0),
                'signal_confidence': order.metadata.get('signal_confidence', 0)
            }
            self.trade_history.append(trade_record)
    
    def _record_daily_state(self, current_date: datetime) -> None:
        """Record daily portfolio state."""
        
        portfolio_value = float(self.current_portfolio.total_value)
        cash_value = float(self.current_portfolio.cash)
        positions_value = float(self.current_portfolio.positions_value)
        
        daily_record = {
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': cash_value,
            'positions_value': positions_value,
            'num_positions': len(self.current_portfolio.positions)
        }
        
        self.daily_values.append(daily_record)
        
        # Record positions if enabled
        if self.config.save_positions:
            positions_record = {
                'date': current_date,
                'positions': {
                    symbol: {
                        'quantity': float(pos.quantity),
                        'price': float(pos.current_price),
                        'value': float(pos.market_value)
                    }
                    for symbol, pos in self.current_portfolio.positions.items()
                }
            }
            self.portfolio_history.append(positions_record)
    
    def _calculate_performance_metrics(self) -> DetailedPerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if not self.daily_values:
            raise BacktestError("No daily values recorded for performance calculation")
        
        # Create portfolio values series
        dates = [record['date'] for record in self.daily_values]
        values = [record['portfolio_value'] for record in self.daily_values]
        
        portfolio_series = pd.Series(values, index=dates)
        
        # Calculate returns
        returns = portfolio_series.pct_change().dropna()
        
        # Get benchmark returns if specified
        benchmark_returns = None
        if self.config.benchmark_symbol and self.config.benchmark_symbol in self.market_data.columns:
            benchmark_data = self.market_data[self.config.benchmark_symbol]
            benchmark_returns = benchmark_data.pct_change().dropna()
        
        # Set risk-free rate for performance analyzer
        self.performance_analyzer.risk_free_rate = self.config.risk_free_rate
        
        # Calculate metrics
        return self.performance_analyzer.calculate_performance_metrics(
            returns,
            self.trade_history,
            benchmark_returns
        )
    
    def _run_bias_detection(self, strategy: IStrategy) -> Dict[str, Any]:
        """Run bias detection analysis."""
        
        # Prepare backtest results for bias analysis
        backtest_results = {
            'returns': pd.Series([record['portfolio_value'] for record in self.daily_values]).pct_change().dropna(),
            'trades': self.trade_history,
            'start_date': self.config.start_date,
            'end_date': self.config.end_date,
            'symbols': list(set(trade['symbol'] for trade in self.trade_history)),
            'performance_metrics': {
                'sharpe_ratio': 2.0,  # Placeholder - would use actual calculated metrics
                'win_rate': 0.6       # Placeholder
            }
        }
        
        strategy_config = {
            'parameters': strategy.get_parameters()
        }
        
        # Run bias analysis
        bias_result = self.bias_detector.analyze_backtest_for_bias(
            backtest_results,
            self.market_data,
            strategy_config
        )
        
        return {
            'warnings': [
                {
                    'bias_type': w.bias_type,
                    'severity': w.severity,
                    'description': w.description,
                    'recommendation': w.recommendation,
                    'confidence': w.confidence
                }
                for w in bias_result.warnings
            ],
            'overall_risk_score': bias_result.overall_risk_score,
            'is_reliable': bias_result.is_reliable,
            'summary': bias_result.summary
        }
    
    def _create_backtest_result(
        self,
        performance_metrics: DetailedPerformanceMetrics,
        bias_analysis: Optional[Dict[str, Any]]
    ) -> BacktestResult:
        """Create backtest result object."""
        
        # Create time series data
        dates = [record['date'] for record in self.daily_values]
        values = [record['portfolio_value'] for record in self.daily_values]
        
        portfolio_values = pd.Series(values, index=dates)
        returns = portfolio_values.pct_change().dropna()
        
        # Create positions DataFrame
        positions_df = pd.DataFrame()
        if self.portfolio_history:
            # This would be more complex in practice
            positions_df = pd.DataFrame(self.portfolio_history)
        
        # Count winning/losing trades
        winning_trades = len([t for t in self.trade_history if t.get('pnl', 0) > 0])
        losing_trades = len([t for t in self.trade_history if t.get('pnl', 0) < 0])
        
        return BacktestResult(
            config=self.config,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            duration_days=(self.config.end_date - self.config.start_date).days,
            initial_capital=self.config.initial_capital,
            final_capital=Decimal(str(portfolio_values.iloc[-1])) if len(portfolio_values) > 0 else self.config.initial_capital,
            total_return=performance_metrics.total_return,
            performance_metrics=performance_metrics,
            total_trades=len(self.trade_history),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            trades=self.trade_history,
            portfolio_values=portfolio_values,
            returns=returns,
            positions=positions_df,
            bias_analysis=bias_analysis,
            corporate_actions=[]  # Would be populated with actual corporate actions
        )
    
    def _result_to_dict(self, result: BacktestResult) -> Dict[str, Any]:
        """Convert BacktestResult to dictionary for return."""
        
        return {
            'config': {
                'start_date': result.start_date,
                'end_date': result.end_date,
                'initial_capital': float(result.initial_capital),
                'duration_days': result.duration_days
            },
            'performance': {
                'total_return': result.total_return,
                'annualized_return': result.performance_metrics.annualized_return,
                'volatility': result.performance_metrics.volatility,
                'sharpe_ratio': result.performance_metrics.sharpe_ratio,
                'max_drawdown': result.performance_metrics.max_drawdown,
                'win_rate': result.performance_metrics.win_rate
            },
            'trading': {
                'total_trades': result.total_trades,
                'winning_trades': result.winning_trades,
                'losing_trades': result.losing_trades
            },
            'portfolio_values': result.portfolio_values.to_dict(),
            'returns': result.returns.to_dict(),
            'trades': result.trades,
            'bias_analysis': result.bias_analysis,
            'execution_time': result.execution_time,
            'warnings': result.warnings
        }