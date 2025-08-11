"""
Signal aggregation and weighting for multi-strategy frameworks.

This module provides functionality to aggregate signals from multiple strategies
with various weighting schemes and combination methods.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum

from ...domain.value_objects import Signal, SignalType
from ...domain.exceptions import ValidationError, StrategyError
from ...infrastructure.logging.logger import get_logger


class AggregationMethod(Enum):
    """Signal aggregation methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    RANK_BASED = "rank_based"
    ENSEMBLE = "ensemble"


class WeightingScheme(Enum):
    """Weighting schemes for signal aggregation."""
    EQUAL = "equal"
    PERFORMANCE_BASED = "performance_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    SHARPE_BASED = "sharpe_based"
    DYNAMIC = "dynamic"


class SignalAggregator:
    """
    Aggregates signals from multiple strategies using various weighting schemes.
    
    Provides sophisticated signal combination methods including weighted averaging,
    majority voting, confidence-based weighting, and ensemble methods.
    """
    
    def __init__(self, logger=None):
        """
        Initialize signal aggregator.
        
        Args:
            logger: Logger instance for logging operations
        """
        self.logger = logger or get_logger(__name__)
        self._strategy_weights: Dict[str, float] = {}
        self._performance_history: Dict[str, List[float]] = {}
        self._aggregation_history: List[Dict[str, Any]] = []
        
    def set_strategy_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for strategies in aggregation.
        
        Args:
            weights: Dictionary mapping strategy names to weights
            
        Raises:
            ValidationError: If weights are invalid
        """
        # Validate weights
        if not weights:
            raise ValidationError("Strategy weights cannot be empty")
        
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValidationError(f"Strategy weights must sum to 1.0, got {total_weight}")
        
        for strategy, weight in weights.items():
            if weight < 0:
                raise ValidationError(f"Strategy weight cannot be negative: {strategy}={weight}")
        
        self._strategy_weights = weights.copy()
        self.logger.info(f"Updated strategy weights: {weights}")
    
    def aggregate_signals(
        self,
        strategy_signals: Dict[str, List[Signal]],
        method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        weighting_scheme: WeightingScheme = WeightingScheme.EQUAL,
        **kwargs
    ) -> List[Signal]:
        """
        Aggregate signals from multiple strategies.
        
        Args:
            strategy_signals: Dictionary mapping strategy names to their signals
            method: Aggregation method to use
            weighting_scheme: Weighting scheme for strategies
            **kwargs: Additional parameters for specific methods
            
        Returns:
            List of aggregated signals
            
        Raises:
            StrategyError: If aggregation fails
            ValidationError: If inputs are invalid
        """
        try:
            self.logger.info(
                f"Aggregating signals from {len(strategy_signals)} strategies",
                method=method.value,
                weighting_scheme=weighting_scheme.value
            )
            
            # Validate inputs
            self._validate_strategy_signals(strategy_signals)
            
            # Calculate strategy weights if needed
            if weighting_scheme != WeightingScheme.EQUAL:
                weights = self._calculate_dynamic_weights(
                    strategy_signals, weighting_scheme, **kwargs
                )
            else:
                weights = self._get_equal_weights(list(strategy_signals.keys()))
            
            # Aggregate signals based on method
            if method == AggregationMethod.WEIGHTED_AVERAGE:
                aggregated_signals = self._weighted_average_aggregation(
                    strategy_signals, weights
                )
            elif method == AggregationMethod.MAJORITY_VOTE:
                aggregated_signals = self._majority_vote_aggregation(
                    strategy_signals, weights
                )
            elif method == AggregationMethod.CONFIDENCE_WEIGHTED:
                aggregated_signals = self._confidence_weighted_aggregation(
                    strategy_signals, weights
                )
            elif method == AggregationMethod.RANK_BASED:
                aggregated_signals = self._rank_based_aggregation(
                    strategy_signals, weights
                )
            elif method == AggregationMethod.ENSEMBLE:
                aggregated_signals = self._ensemble_aggregation(
                    strategy_signals, weights, **kwargs
                )
            else:
                raise ValidationError(f"Unsupported aggregation method: {method}")
            
            # Record aggregation history
            self._record_aggregation(strategy_signals, aggregated_signals, method, weights)
            
            self.logger.info(f"Successfully aggregated {len(aggregated_signals)} signals")
            return aggregated_signals
            
        except Exception as e:
            self.logger.error(f"Signal aggregation failed: {str(e)}")
            raise StrategyError(
                f"Signal aggregation failed: {str(e)}",
                error_code="AGGREGATION_FAILED"
            ) from e
    
    def _validate_strategy_signals(self, strategy_signals: Dict[str, List[Signal]]) -> None:
        """Validate strategy signals input."""
        if not strategy_signals:
            raise ValidationError("Strategy signals cannot be empty")
        
        # Check that all strategies have signals
        for strategy_name, signals in strategy_signals.items():
            if not signals:
                raise ValidationError(f"Strategy {strategy_name} has no signals")
            
            # Validate signal types
            for signal in signals:
                if not isinstance(signal, Signal):
                    raise ValidationError(f"Invalid signal type in strategy {strategy_name}")
    
    def _get_equal_weights(self, strategy_names: List[str]) -> Dict[str, float]:
        """Calculate equal weights for strategies."""
        weight = 1.0 / len(strategy_names)
        return {name: weight for name in strategy_names}
    
    def _calculate_dynamic_weights(
        self,
        strategy_signals: Dict[str, List[Signal]],
        weighting_scheme: WeightingScheme,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate dynamic weights based on weighting scheme.
        
        Args:
            strategy_signals: Strategy signals
            weighting_scheme: Weighting scheme to use
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of strategy weights
        """
        if weighting_scheme == WeightingScheme.PERFORMANCE_BASED:
            return self._calculate_performance_weights(strategy_signals, **kwargs)
        elif weighting_scheme == WeightingScheme.VOLATILITY_ADJUSTED:
            return self._calculate_volatility_weights(strategy_signals, **kwargs)
        elif weighting_scheme == WeightingScheme.SHARPE_BASED:
            return self._calculate_sharpe_weights(strategy_signals, **kwargs)
        elif weighting_scheme == WeightingScheme.DYNAMIC:
            return self._calculate_adaptive_weights(strategy_signals, **kwargs)
        else:
            return self._get_equal_weights(list(strategy_signals.keys()))
    
    def _calculate_performance_weights(
        self,
        strategy_signals: Dict[str, List[Signal]],
        lookback_period: int = 30,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate weights based on recent performance."""
        weights = {}
        total_performance = 0
        
        for strategy_name in strategy_signals.keys():
            # Get recent performance (placeholder - would use actual performance data)
            recent_performance = self._get_strategy_performance(strategy_name, lookback_period)
            weights[strategy_name] = max(recent_performance, 0.01)  # Minimum weight
            total_performance += weights[strategy_name]
        
        # Normalize weights
        if total_performance > 0:
            for strategy_name in weights:
                weights[strategy_name] /= total_performance
        else:
            weights = self._get_equal_weights(list(strategy_signals.keys()))
        
        return weights
    
    def _calculate_volatility_weights(
        self,
        strategy_signals: Dict[str, List[Signal]],
        **kwargs
    ) -> Dict[str, float]:
        """Calculate weights inversely proportional to volatility."""
        weights = {}
        total_inv_vol = 0
        
        for strategy_name in strategy_signals.keys():
            # Calculate signal volatility (placeholder implementation)
            volatility = self._calculate_signal_volatility(strategy_signals[strategy_name])
            inv_volatility = 1.0 / max(volatility, 0.01)  # Avoid division by zero
            weights[strategy_name] = inv_volatility
            total_inv_vol += inv_volatility
        
        # Normalize weights
        if total_inv_vol > 0:
            for strategy_name in weights:
                weights[strategy_name] /= total_inv_vol
        else:
            weights = self._get_equal_weights(list(strategy_signals.keys()))
        
        return weights
    
    def _calculate_sharpe_weights(
        self,
        strategy_signals: Dict[str, List[Signal]],
        **kwargs
    ) -> Dict[str, float]:
        """Calculate weights based on Sharpe ratios."""
        weights = {}
        total_sharpe = 0
        
        for strategy_name in strategy_signals.keys():
            # Calculate Sharpe ratio (placeholder - would use actual returns data)
            sharpe_ratio = self._get_strategy_sharpe(strategy_name)
            weights[strategy_name] = max(sharpe_ratio, 0.01)  # Minimum weight
            total_sharpe += weights[strategy_name]
        
        # Normalize weights
        if total_sharpe > 0:
            for strategy_name in weights:
                weights[strategy_name] /= total_sharpe
        else:
            weights = self._get_equal_weights(list(strategy_signals.keys()))
        
        return weights
    
    def _calculate_adaptive_weights(
        self,
        strategy_signals: Dict[str, List[Signal]],
        adaptation_rate: float = 0.1,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate adaptive weights that adjust based on recent performance."""
        if not self._strategy_weights:
            return self._get_equal_weights(list(strategy_signals.keys()))
        
        # Adjust weights based on recent performance
        adjusted_weights = {}
        for strategy_name in strategy_signals.keys():
            current_weight = self._strategy_weights.get(strategy_name, 0.0)
            performance_adjustment = self._get_recent_performance_adjustment(strategy_name)
            
            new_weight = current_weight + adaptation_rate * performance_adjustment
            adjusted_weights[strategy_name] = max(new_weight, 0.01)  # Minimum weight
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for strategy_name in adjusted_weights:
                adjusted_weights[strategy_name] /= total_weight
        
        return adjusted_weights
    
    def _weighted_average_aggregation(
        self,
        strategy_signals: Dict[str, List[Signal]],
        weights: Dict[str, float]
    ) -> List[Signal]:
        """Aggregate signals using weighted average."""
        # Group signals by symbol and timestamp
        signal_groups = self._group_signals_by_symbol_time(strategy_signals)
        aggregated_signals = []
        
        for (symbol, timestamp), signals_group in signal_groups.items():
            weighted_strength = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for strategy_name, signal in signals_group.items():
                weight = weights.get(strategy_name, 0.0)
                weighted_strength += signal.strength * weight
                weighted_confidence += signal.confidence * weight
                total_weight += weight
            
            if total_weight > 0:
                # Determine signal type based on weighted strength
                if weighted_strength > 0.1:
                    signal_type = SignalType.BUY
                elif weighted_strength < -0.1:
                    signal_type = SignalType.SELL
                else:
                    signal_type = SignalType.HOLD
                
                aggregated_signal = Signal(
                    symbol=symbol,
                    timestamp=timestamp,
                    signal_type=signal_type,
                    strength=weighted_strength,
                    confidence=weighted_confidence,
                    source="aggregated",
                    metadata={
                        "aggregation_method": "weighted_average",
                        "contributing_strategies": list(signals_group.keys()),
                        "weights": {k: weights.get(k, 0.0) for k in signals_group.keys()}
                    }
                )
                aggregated_signals.append(aggregated_signal)
        
        return aggregated_signals
    
    def _majority_vote_aggregation(
        self,
        strategy_signals: Dict[str, List[Signal]],
        weights: Dict[str, float]
    ) -> List[Signal]:
        """Aggregate signals using majority voting."""
        from ...domain.value_objects import SignalType
        signal_groups = self._group_signals_by_symbol_time(strategy_signals)
        aggregated_signals = []
        
        for (symbol, timestamp), signals_group in signal_groups.items():
            vote_counts = {SignalType.BUY: 0.0, SignalType.SELL: 0.0, SignalType.HOLD: 0.0}
            total_confidence = 0.0
            
            for strategy_name, signal in signals_group.items():
                weight = weights.get(strategy_name, 0.0)
                vote_counts[signal.signal_type] += weight
                total_confidence += signal.confidence * weight
            
            # Determine winning vote
            winning_signal = max(vote_counts, key=vote_counts.get)
            winning_strength = vote_counts[winning_signal] - max(
                v for k, v in vote_counts.items() if k != winning_signal
            )
            
            aggregated_signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                signal_type=winning_signal,
                strength=winning_strength,
                confidence=total_confidence / len(signals_group),
                source="aggregated",
                metadata={
                    "aggregation_method": "majority_vote",
                    "vote_counts": {k.value: v for k, v in vote_counts.items()},
                    "contributing_strategies": list(signals_group.keys())
                }
            )
            aggregated_signals.append(aggregated_signal)
        
        return aggregated_signals
    
    def _confidence_weighted_aggregation(
        self,
        strategy_signals: Dict[str, List[Signal]],
        weights: Dict[str, float]
    ) -> List[Signal]:
        """Aggregate signals weighted by confidence levels."""
        signal_groups = self._group_signals_by_symbol_time(strategy_signals)
        aggregated_signals = []
        
        for (symbol, timestamp), signals_group in signal_groups.items():
            weighted_strength = 0.0
            total_confidence_weight = 0.0
            
            for strategy_name, signal in signals_group.items():
                strategy_weight = weights.get(strategy_name, 0.0)
                confidence_weight = signal.confidence * strategy_weight
                
                weighted_strength += signal.strength * confidence_weight
                total_confidence_weight += confidence_weight
            
            if total_confidence_weight > 0:
                final_strength = weighted_strength / total_confidence_weight
                
                # Determine signal type
                if final_strength > 0.1:
                    signal_type = SignalType.BUY
                elif final_strength < -0.1:
                    signal_type = SignalType.SELL
                else:
                    signal_type = SignalType.HOLD
                
                aggregated_signal = Signal(
                    symbol=symbol,
                    timestamp=timestamp,
                    signal_type=signal_type,
                    strength=final_strength,
                    confidence=total_confidence_weight / len(signals_group),
                    source="aggregated",
                    metadata={
                        "aggregation_method": "confidence_weighted",
                        "contributing_strategies": list(signals_group.keys())
                    }
                )
                aggregated_signals.append(aggregated_signal)
        
        return aggregated_signals
    
    def _rank_based_aggregation(
        self,
        strategy_signals: Dict[str, List[Signal]],
        weights: Dict[str, float]
    ) -> List[Signal]:
        """Aggregate signals using rank-based method."""
        signal_groups = self._group_signals_by_symbol_time(strategy_signals)
        aggregated_signals = []
        
        for (symbol, timestamp), signals_group in signal_groups.items():
            # Rank signals by strength
            ranked_signals = sorted(
                signals_group.items(),
                key=lambda x: abs(x[1].strength),
                reverse=True
            )
            
            # Weight by rank and strategy weight
            weighted_strength = 0.0
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for rank, (strategy_name, signal) in enumerate(ranked_signals):
                rank_weight = 1.0 / (rank + 1)  # Higher rank = higher weight
                strategy_weight = weights.get(strategy_name, 0.0)
                combined_weight = rank_weight * strategy_weight
                
                weighted_strength += signal.strength * combined_weight
                weighted_confidence += signal.confidence * combined_weight
                total_weight += combined_weight
            
            if total_weight > 0:
                final_strength = weighted_strength / total_weight
                final_confidence = weighted_confidence / total_weight
                
                # Determine signal type
                from ...domain.value_objects import SignalType
                if final_strength > 0.1:
                    signal_type = SignalType.BUY
                elif final_strength < -0.1:
                    signal_type = SignalType.SELL
                else:
                    signal_type = SignalType.HOLD
                
                aggregated_signal = Signal(
                    symbol=symbol,
                    timestamp=timestamp,
                    signal_type=signal_type,
                    strength=final_strength,
                    confidence=final_confidence,
                    source="aggregated",
                    metadata={
                        "aggregation_method": "rank_based",
                        "contributing_strategies": list(signals_group.keys())
                    }
                )
                aggregated_signals.append(aggregated_signal)
        
        return aggregated_signals
    
    def _ensemble_aggregation(
        self,
        strategy_signals: Dict[str, List[Signal]],
        weights: Dict[str, float],
        ensemble_methods: List[AggregationMethod] = None,
        **kwargs
    ) -> List[Signal]:
        """Aggregate signals using ensemble of multiple methods."""
        if not ensemble_methods:
            ensemble_methods = [
                AggregationMethod.WEIGHTED_AVERAGE,
                AggregationMethod.MAJORITY_VOTE,
                AggregationMethod.CONFIDENCE_WEIGHTED
            ]
        
        # Get results from each method
        method_results = {}
        for method in ensemble_methods:
            if method != AggregationMethod.ENSEMBLE:  # Avoid recursion
                method_results[method] = self._get_method_aggregation(
                    strategy_signals, weights, method
                )
        
        # Combine results from different methods
        return self._combine_ensemble_results(method_results)
    
    def _get_method_aggregation(
        self,
        strategy_signals: Dict[str, List[Signal]],
        weights: Dict[str, float],
        method: AggregationMethod
    ) -> List[Signal]:
        """Get aggregation result for a specific method."""
        if method == AggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_aggregation(strategy_signals, weights)
        elif method == AggregationMethod.MAJORITY_VOTE:
            return self._majority_vote_aggregation(strategy_signals, weights)
        elif method == AggregationMethod.CONFIDENCE_WEIGHTED:
            return self._confidence_weighted_aggregation(strategy_signals, weights)
        elif method == AggregationMethod.RANK_BASED:
            return self._rank_based_aggregation(strategy_signals, weights)
        else:
            return []
    
    def _combine_ensemble_results(
        self,
        method_results: Dict[AggregationMethod, List[Signal]]
    ) -> List[Signal]:
        """Combine results from multiple aggregation methods."""
        # Group signals by symbol and timestamp across methods
        combined_groups = {}
        
        for method, signals in method_results.items():
            for signal in signals:
                key = (signal.symbol, signal.timestamp)
                if key not in combined_groups:
                    combined_groups[key] = {}
                combined_groups[key][method] = signal
        
        # Aggregate across methods
        final_signals = []
        for (symbol, timestamp), method_signals in combined_groups.items():
            # Simple average across methods
            avg_strength = np.mean([s.strength for s in method_signals.values()])
            avg_confidence = np.mean([s.confidence for s in method_signals.values()])
            
            # Determine signal type
            from ...domain.value_objects import SignalType
            if avg_strength > 0.1:
                signal_type = SignalType.BUY
            elif avg_strength < -0.1:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            final_signal = Signal(
                symbol=symbol,
                timestamp=timestamp,
                signal_type=signal_type,
                strength=avg_strength,
                confidence=avg_confidence,
                source="aggregated",
                metadata={
                    "aggregation_method": "ensemble",
                    "ensemble_methods": [m.value for m in method_signals.keys()],
                    "method_count": len(method_signals)
                }
            )
            final_signals.append(final_signal)
        
        return final_signals
    
    def _group_signals_by_symbol_time(
        self,
        strategy_signals: Dict[str, List[Signal]]
    ) -> Dict[tuple, Dict[str, Signal]]:
        """Group signals by symbol and timestamp."""
        groups = {}
        
        for strategy_name, signals in strategy_signals.items():
            for signal in signals:
                key = (signal.symbol, signal.timestamp)
                if key not in groups:
                    groups[key] = {}
                groups[key][strategy_name] = signal
        
        return groups
    
    def _record_aggregation(
        self,
        strategy_signals: Dict[str, List[Signal]],
        aggregated_signals: List[Signal],
        method: AggregationMethod,
        weights: Dict[str, float]
    ) -> None:
        """Record aggregation history for analysis."""
        record = {
            "timestamp": datetime.now(),
            "method": method.value,
            "strategy_count": len(strategy_signals),
            "input_signal_count": sum(len(signals) for signals in strategy_signals.values()),
            "output_signal_count": len(aggregated_signals),
            "weights": weights.copy()
        }
        
        self._aggregation_history.append(record)
        
        # Keep only recent history (last 1000 records)
        if len(self._aggregation_history) > 1000:
            self._aggregation_history = self._aggregation_history[-1000:]
    
    # Placeholder methods for performance calculations
    def _get_strategy_performance(self, strategy_name: str, lookback_period: int) -> float:
        """Get recent performance for a strategy (placeholder)."""
        # This would be implemented with actual performance tracking
        return np.random.uniform(0.5, 1.5)  # Placeholder
    
    def _calculate_signal_volatility(self, signals: List[Signal]) -> float:
        """Calculate volatility of signal strengths."""
        if len(signals) < 2:
            return 0.1  # Default volatility
        
        strengths = [signal.strength for signal in signals]
        return np.std(strengths)
    
    def _get_strategy_sharpe(self, strategy_name: str) -> float:
        """Get Sharpe ratio for a strategy (placeholder)."""
        # This would be implemented with actual performance tracking
        return np.random.uniform(0.5, 2.0)  # Placeholder
    
    def _get_recent_performance_adjustment(self, strategy_name: str) -> float:
        """Get recent performance adjustment for adaptive weighting."""
        # This would be implemented with actual performance tracking
        return np.random.uniform(-0.1, 0.1)  # Placeholder
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        if not self._aggregation_history:
            return {}
        
        recent_records = self._aggregation_history[-10:]  # Last 10 aggregations
        
        return {
            "total_aggregations": len(self._aggregation_history),
            "recent_methods": [r["method"] for r in recent_records],
            "average_input_signals": np.mean([r["input_signal_count"] for r in recent_records]),
            "average_output_signals": np.mean([r["output_signal_count"] for r in recent_records]),
            "current_weights": self._strategy_weights.copy()
        }