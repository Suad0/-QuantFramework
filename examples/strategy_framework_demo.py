"""
Demonstration of the sophisticated strategy framework.

This example shows how to use the strategy registry, optimization,
signal aggregation, validation, and composition features.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import framework components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.interfaces import IStrategy
from src.domain.value_objects import Signal, SignalType
from src.application.services.strategy_service import StrategyService
from src.infrastructure.strategies.optimizer import ParameterRange
from src.infrastructure.strategies.aggregator import AggregationMethod, WeightingScheme
from src.infrastructure.strategies.validator import ValidationLevel
from src.infrastructure.strategies.composer import CompositionMethod, RiskBudgetingMethod, CompositionConstraints


# Example Strategy Implementations
class MovingAverageStrategy(IStrategy):
    """Simple moving average crossover strategy."""
    
    def __init__(self, short_window: int = 10, long_window: int = 30, threshold: float = 0.02):
        self.short_window = short_window
        self.long_window = long_window
        self.threshold = threshold
    
    def generate_signals(self, data: pd.DataFrame, context: Dict[str, Any]) -> List[Signal]:
        """Generate signals based on moving average crossover."""
        signals = []
        
        # Assume data has 'close' column
        if 'close' not in data.columns:
            return signals
        
        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        for i in range(len(data)):
            if i < self.long_window:
                continue
            
            short_val = short_ma.iloc[i]
            long_val = long_ma.iloc[i]
            
            if pd.isna(short_val) or pd.isna(long_val):
                continue
            
            # Calculate signal strength
            price_diff = (short_val - long_val) / long_val
            
            if price_diff > self.threshold:
                signal_type = SignalType.BUY
                strength = min(price_diff / self.threshold, 1.0)
            elif price_diff < -self.threshold:
                signal_type = SignalType.SELL
                strength = max(price_diff / self.threshold, -1.0)
            else:
                signal_type = SignalType.HOLD
                strength = 0.0
            
            # Calculate confidence based on trend consistency
            recent_diffs = [(short_ma.iloc[j] - long_ma.iloc[j]) / long_ma.iloc[j] 
                           for j in range(max(0, i-5), i+1)]
            trend_consistency = len([d for d in recent_diffs if d * price_diff > 0]) / len(recent_diffs)
            
            signal = Signal(
                symbol=context.get('symbol', 'UNKNOWN'),
                timestamp=data.index[i] if hasattr(data.index, '__getitem__') else datetime.now(),
                signal_type=signal_type,
                strength=strength,
                confidence=trend_consistency,
                source="MovingAverageStrategy",
                metadata={
                    "short_window": self.short_window,
                    "long_window": self.long_window,
                    "short_ma": short_val,
                    "long_ma": long_val,
                    "price_diff": price_diff
                }
            )
            signals.append(signal)
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "threshold": self.threshold
        }
    
    def validate_signals(self, signals: List[Signal]) -> Dict[str, Any]:
        """Validate generated signals."""
        return {"valid": True, "message": "Signals validated successfully"}


class RSIStrategy(IStrategy):
    """RSI-based mean reversion strategy."""
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data: pd.DataFrame, context: Dict[str, Any]) -> List[Signal]:
        """Generate signals based on RSI levels."""
        signals = []
        
        if 'close' not in data.columns:
            return signals
        
        # Calculate RSI
        rsi = self._calculate_rsi(data['close'], self.rsi_period)
        
        # Generate signals
        for i in range(len(data)):
            if i < self.rsi_period:
                continue
            
            rsi_val = rsi.iloc[i]
            if pd.isna(rsi_val):
                continue
            
            # Determine signal based on RSI levels
            if rsi_val < self.oversold:
                signal_type = SignalType.BUY
                strength = (self.oversold - rsi_val) / self.oversold
            elif rsi_val > self.overbought:
                signal_type = SignalType.SELL
                strength = -(rsi_val - self.overbought) / (100 - self.overbought)
            else:
                signal_type = SignalType.HOLD
                strength = 0.0
            
            # Calculate confidence based on RSI extremity
            if rsi_val < self.oversold:
                confidence = min((self.oversold - rsi_val) / 20, 1.0)
            elif rsi_val > self.overbought:
                confidence = min((rsi_val - self.overbought) / 20, 1.0)
            else:
                confidence = 0.5  # Neutral confidence for hold signals
            
            signal = Signal(
                symbol=context.get('symbol', 'UNKNOWN'),
                timestamp=data.index[i] if hasattr(data.index, '__getitem__') else datetime.now(),
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                source="RSIStrategy",
                metadata={
                    "rsi_period": self.rsi_period,
                    "rsi_value": rsi_val,
                    "oversold": self.oversold,
                    "overbought": self.overbought
                }
            )
            signals.append(signal)
        
        return signals
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            "rsi_period": self.rsi_period,
            "oversold": self.oversold,
            "overbought": self.overbought
        }
    
    def validate_signals(self, signals: List[Signal]) -> Dict[str, Any]:
        """Validate generated signals."""
        return {"valid": True, "message": "Signals validated successfully"}


class MomentumStrategy(IStrategy):
    """Price momentum strategy."""
    
    def __init__(self, lookback_period: int = 20, momentum_threshold: float = 0.05):
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
    
    def generate_signals(self, data: pd.DataFrame, context: Dict[str, Any]) -> List[Signal]:
        """Generate signals based on price momentum."""
        signals = []
        
        if 'close' not in data.columns:
            return signals
        
        # Calculate momentum
        momentum = data['close'].pct_change(self.lookback_period)
        
        # Generate signals
        for i in range(len(data)):
            if i < self.lookback_period:
                continue
            
            momentum_val = momentum.iloc[i]
            if pd.isna(momentum_val):
                continue
            
            # Determine signal based on momentum
            if momentum_val > self.momentum_threshold:
                signal_type = SignalType.BUY
                strength = min(momentum_val / self.momentum_threshold, 1.0)
            elif momentum_val < -self.momentum_threshold:
                signal_type = SignalType.SELL
                strength = max(momentum_val / self.momentum_threshold, -1.0)
            else:
                signal_type = SignalType.HOLD
                strength = 0.0
            
            # Calculate confidence based on momentum consistency
            recent_momentum = momentum.iloc[max(0, i-5):i+1]
            momentum_consistency = len([m for m in recent_momentum if m * momentum_val > 0]) / len(recent_momentum)
            
            signal = Signal(
                symbol=context.get('symbol', 'UNKNOWN'),
                timestamp=data.index[i] if hasattr(data.index, '__getitem__') else datetime.now(),
                signal_type=signal_type,
                strength=strength,
                confidence=momentum_consistency,
                source="MomentumStrategy",
                metadata={
                    "lookback_period": self.lookback_period,
                    "momentum_value": momentum_val,
                    "momentum_threshold": self.momentum_threshold
                }
            )
            signals.append(signal)
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            "lookback_period": self.lookback_period,
            "momentum_threshold": self.momentum_threshold
        }
    
    def validate_signals(self, signals: List[Signal]) -> Dict[str, Any]:
        """Validate generated signals."""
        return {"valid": True, "message": "Signals validated successfully"}


def create_sample_data(days: int = 252) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Generate synthetic price data with some trends and noise
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)  # Daily returns
    
    # Add some trend periods
    trend_periods = [(50, 100), (150, 200)]
    for start, end in trend_periods:
        returns[start:end] += 0.002  # Positive trend
    
    # Calculate prices from returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add some volume data
    volume = np.random.lognormal(10, 0.5, days)
    
    data = pd.DataFrame({
        'close': prices,
        'volume': volume
    }, index=dates)
    
    return data


def demonstrate_strategy_framework():
    """Demonstrate the complete strategy framework."""
    print("=== Strategy Framework Demonstration ===\n")
    
    # Initialize strategy service
    strategy_service = StrategyService()
    
    # Step 1: Register strategies
    print("1. Registering strategies...")
    strategy_service.register_strategy(
        MovingAverageStrategy, 
        "MovingAverage",
        {"description": "Moving average crossover strategy", "type": "trend_following"}
    )
    strategy_service.register_strategy(
        RSIStrategy,
        "RSI", 
        {"description": "RSI mean reversion strategy", "type": "mean_reversion"}
    )
    strategy_service.register_strategy(
        MomentumStrategy,
        "Momentum",
        {"description": "Price momentum strategy", "type": "momentum"}
    )
    
    print(f"Registered strategies: {strategy_service.list_available_strategies()}")
    print()
    
    # Step 2: Create sample data
    print("2. Creating sample market data...")
    market_data = create_sample_data(days=100)
    print(f"Created {len(market_data)} days of market data")
    print(f"Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}")
    print()
    
    # Step 3: Optimize strategy parameters (demonstration)
    print("3. Optimizing strategy parameters...")
    
    def fitness_function(strategy: IStrategy) -> float:
        """Simple fitness function for demonstration."""
        # Generate signals and calculate a simple fitness score
        signals = strategy.generate_signals(market_data, {"symbol": "DEMO"})
        if not signals:
            return 0.0
        
        # Simple fitness: number of signals with high confidence
        high_confidence_signals = [s for s in signals if s.confidence > 0.7]
        return len(high_confidence_signals) / len(signals)
    
    # Optimize MovingAverage strategy
    ma_param_ranges = [
        ParameterRange("short_window", 5, 20, "int"),
        ParameterRange("long_window", 25, 50, "int"),
        ParameterRange("threshold", 0.01, 0.05, "float")
    ]
    
    try:
        ma_optimization = strategy_service.optimize_strategy_parameters(
            strategy_name="MovingAverage",
            parameter_ranges=ma_param_ranges,
            fitness_function=fitness_function,
            method="genetic_algorithm",
            population_size=20,
            generations=10
        )
        print(f"MovingAverage optimization completed:")
        print(f"  Best parameters: {ma_optimization['best_parameters']}")
        print(f"  Best fitness: {ma_optimization['best_fitness']:.4f}")
    except Exception as e:
        print(f"Optimization failed: {e}")
    print()
    
    # Step 4: Generate signals from multiple strategies
    print("4. Generating signals from strategies...")
    strategy_names = ["MovingAverage", "RSI", "Momentum"]
    strategy_signals = {}
    
    for strategy_name in strategy_names:
        signals = strategy_service.generate_strategy_signals(
            strategy_name=strategy_name,
            market_data=market_data,
            context={"symbol": "DEMO"}
        )
        strategy_signals[strategy_name] = signals
        print(f"  {strategy_name}: {len(signals)} signals generated")
    
    print()
    
    # Step 5: Aggregate signals
    print("5. Aggregating signals from multiple strategies...")
    aggregated_signals = strategy_service.aggregate_multi_strategy_signals(
        strategy_signals=strategy_signals,
        method=AggregationMethod.WEIGHTED_AVERAGE,
        weighting_scheme=WeightingScheme.EQUAL
    )
    print(f"Aggregated {len(aggregated_signals)} signals")
    
    # Show sample aggregated signals
    if aggregated_signals:
        print("Sample aggregated signals:")
        for i, signal in enumerate(aggregated_signals[:5]):
            print(f"  {i+1}. {signal}")
    print()
    
    # Step 6: Validate signals
    print("6. Validating aggregated signals...")
    validation_report = strategy_service.validate_signals(
        signals=aggregated_signals,
        validation_level=ValidationLevel.STANDARD,
        historical_data=market_data
    )
    
    print(f"Validation results:")
    print(f"  Quality score: {validation_report['quality_score']:.3f}")
    print(f"  Critical failures: {validation_report['critical_failures']}")
    print(f"  Warnings: {validation_report['warnings']}")
    print(f"  Recommendations: {len(validation_report['recommendations'])}")
    
    if validation_report['recommendations']:
        print("  Top recommendations:")
        for rec in validation_report['recommendations'][:3]:
            print(f"    - {rec}")
    print()
    
    # Step 7: Compose strategies with risk budgeting
    print("7. Composing strategies with risk budgeting...")
    composition_constraints = CompositionConstraints(
        max_single_weight=0.6,
        min_single_weight=0.1,
        target_volatility=0.15
    )
    
    composition_result = strategy_service.compose_strategies(
        strategy_names=strategy_names,
        method=CompositionMethod.RISK_PARITY,
        risk_budgeting=RiskBudgetingMethod.VOLATILITY_BASED,
        constraints=composition_constraints,
        historical_data=market_data
    )
    
    print("Strategy composition results:")
    print(f"  Total expected return: {composition_result['total_expected_return']:.4f}")
    print(f"  Total expected risk: {composition_result['total_expected_risk']:.4f}")
    print("  Strategy allocations:")
    
    for allocation in composition_result['allocations']:
        print(f"    {allocation['strategy_name']}: {allocation['weight']:.3f} "
              f"(risk budget: {allocation['risk_budget']:.3f})")
    
    print()
    
    # Step 8: Run complete workflow
    print("8. Running complete integrated workflow...")
    workflow_config = {
        "aggregation_config": {
            "method": AggregationMethod.CONFIDENCE_WEIGHTED,
            "weighting_scheme": WeightingScheme.PERFORMANCE_BASED
        },
        "validation_config": {
            "validation_level": ValidationLevel.COMPREHENSIVE
        },
        "composition_config": {
            "method": CompositionMethod.SHARPE_OPTIMAL,
            "risk_budgeting": RiskBudgetingMethod.EQUAL_RISK
        }
    }
    
    workflow_results = strategy_service.run_complete_strategy_workflow(
        strategy_names=strategy_names,
        market_data=market_data,
        **workflow_config
    )
    
    print("Complete workflow results:")
    print(f"  Strategies processed: {len(workflow_results['strategies'])}")
    print(f"  Total signals generated: {sum(workflow_results['signals'].values())}")
    print(f"  Aggregated signals: {workflow_results['aggregated_signals']}")
    print(f"  Final quality score: {workflow_results['validation_report']['quality_score']:.3f}")
    print(f"  Portfolio expected return: {workflow_results['composition_result']['total_expected_return']:.4f}")
    print()
    
    # Step 9: Service statistics
    print("9. Service statistics...")
    stats = strategy_service.get_service_statistics()
    print(f"Service statistics:")
    for key, value in stats.items():
        if key != 'timestamp':
            print(f"  {key}: {value}")
    
    print("\n=== Strategy Framework Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_strategy_framework()