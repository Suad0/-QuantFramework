"""
Strategy composition framework with risk budgeting.

This module provides sophisticated strategy composition capabilities,
allowing multiple strategies to be combined with risk budgeting,
allocation constraints, and dynamic rebalancing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings

from ...domain.interfaces import IStrategy
from ...domain.entities import Strategy, Portfolio
from ...domain.value_objects import Signal
from ...domain.exceptions import StrategyError, ValidationError, RiskError
from ...infrastructure.logging.logger import get_logger


class CompositionMethod(Enum):
    """Strategy composition methods."""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    VOLATILITY_TARGET = "volatility_target"
    SHARPE_OPTIMAL = "sharpe_optimal"
    HIERARCHICAL = "hierarchical"
    DYNAMIC = "dynamic"


class RiskBudgetingMethod(Enum):
    """Risk budgeting methods."""
    EQUAL_RISK = "equal_risk"
    VOLATILITY_BASED = "volatility_based"
    VAR_BASED = "var_based"
    EXPECTED_SHORTFALL = "expected_shortfall"
    CUSTOM = "custom"


@dataclass
class StrategyAllocation:
    """Represents allocation to a strategy within a composition."""
    strategy_id: str
    strategy_name: str
    weight: float
    risk_budget: float
    expected_return: float
    expected_volatility: float
    max_weight: float = 1.0
    min_weight: float = 0.0
    is_active: bool = True


@dataclass
class CompositionConstraints:
    """Constraints for strategy composition."""
    max_strategies: Optional[int] = None
    min_strategies: Optional[int] = None
    max_single_weight: float = 0.5
    min_single_weight: float = 0.01
    max_total_risk: Optional[float] = None
    target_volatility: Optional[float] = None
    rebalance_threshold: float = 0.05
    turnover_limit: Optional[float] = None


@dataclass
class CompositionResult:
    """Result of strategy composition optimization."""
    allocations: List[StrategyAllocation]
    total_expected_return: float
    total_expected_risk: float
    risk_budget_utilization: Dict[str, float]
    composition_metrics: Dict[str, Any]
    rebalancing_trades: Optional[Dict[str, float]] = None
    optimization_details: Optional[Dict[str, Any]] = None


class StrategyComposer:
    """
    Advanced strategy composition framework with risk budgeting.
    
    Provides sophisticated methods for combining multiple strategies
    with risk budgeting, allocation optimization, and dynamic rebalancing.
    """
    
    def __init__(self, logger=None):
        """
        Initialize strategy composer.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or get_logger(__name__)
        self.composition_history = []
        self.current_composition = None
        self.performance_tracking = {}
        
    def compose_strategies(
        self,
        strategies: Dict[str, IStrategy],
        method: CompositionMethod = CompositionMethod.RISK_PARITY,
        risk_budgeting: RiskBudgetingMethod = RiskBudgetingMethod.EQUAL_RISK,
        constraints: Optional[CompositionConstraints] = None,
        historical_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> CompositionResult:
        """
        Compose multiple strategies with risk budgeting.
        
        Args:
            strategies: Dictionary of strategy ID to strategy instance
            method: Composition method to use
            risk_budgeting: Risk budgeting method
            constraints: Composition constraints
            historical_data: Historical data for optimization
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Composition result with allocations and metrics
            
        Raises:
            StrategyError: If composition fails
            ValidationError: If inputs are invalid
        """
        try:
            self.logger.info(
                f"Starting strategy composition",
                num_strategies=len(strategies),
                method=method.value,
                risk_budgeting=risk_budgeting.value
            )
            
            # Validate inputs
            self._validate_composition_inputs(strategies, constraints)
            
            # Set default constraints if not provided
            if constraints is None:
                constraints = CompositionConstraints()
            
            # Calculate strategy statistics
            strategy_stats = self._calculate_strategy_statistics(
                strategies, historical_data, **kwargs
            )
            
            # Calculate risk budgets
            risk_budgets = self._calculate_risk_budgets(
                strategy_stats, risk_budgeting, **kwargs
            )
            
            # Optimize composition based on method
            if method == CompositionMethod.EQUAL_WEIGHT:
                allocations = self._equal_weight_composition(
                    strategies, strategy_stats, risk_budgets, constraints
                )
            elif method == CompositionMethod.RISK_PARITY:
                allocations = self._risk_parity_composition(
                    strategies, strategy_stats, risk_budgets, constraints
                )
            elif method == CompositionMethod.VOLATILITY_TARGET:
                allocations = self._volatility_target_composition(
                    strategies, strategy_stats, risk_budgets, constraints, **kwargs
                )
            elif method == CompositionMethod.SHARPE_OPTIMAL:
                allocations = self._sharpe_optimal_composition(
                    strategies, strategy_stats, risk_budgets, constraints
                )
            elif method == CompositionMethod.HIERARCHICAL:
                allocations = self._hierarchical_composition(
                    strategies, strategy_stats, risk_budgets, constraints, **kwargs
                )
            elif method == CompositionMethod.DYNAMIC:
                allocations = self._dynamic_composition(
                    strategies, strategy_stats, risk_budgets, constraints, **kwargs
                )
            else:
                raise ValidationError(f"Unsupported composition method: {method}")
            
            # Calculate composition metrics
            composition_metrics = self._calculate_composition_metrics(
                allocations, strategy_stats
            )
            
            # Create result
            result = CompositionResult(
                allocations=allocations,
                total_expected_return=composition_metrics["expected_return"],
                total_expected_risk=composition_metrics["expected_risk"],
                risk_budget_utilization=self._calculate_risk_utilization(
                    allocations, risk_budgets
                ),
                composition_metrics=composition_metrics
            )
            
            # Store composition
            self.current_composition = result
            self.composition_history.append({
                "timestamp": datetime.now(),
                "method": method.value,
                "result": result
            })
            
            self.logger.info(
                f"Strategy composition completed",
                num_allocations=len(allocations),
                total_expected_return=result.total_expected_return,
                total_expected_risk=result.total_expected_risk
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Strategy composition failed: {str(e)}")
            raise StrategyError(
                f"Strategy composition failed: {str(e)}",
                error_code="COMPOSITION_FAILED"
            ) from e
    
    def rebalance_composition(
        self,
        current_allocations: List[StrategyAllocation],
        new_strategy_stats: Dict[str, Dict[str, float]],
        constraints: Optional[CompositionConstraints] = None
    ) -> CompositionResult:
        """
        Rebalance existing strategy composition.
        
        Args:
            current_allocations: Current strategy allocations
            new_strategy_stats: Updated strategy statistics
            constraints: Rebalancing constraints
            
        Returns:
            Rebalanced composition result
        """
        try:
            self.logger.info(
                f"Starting composition rebalancing",
                current_strategies=len(current_allocations)
            )
            
            if constraints is None:
                constraints = CompositionConstraints()
            
            # Check if rebalancing is needed
            rebalancing_needed = self._check_rebalancing_trigger(
                current_allocations, new_strategy_stats, constraints
            )
            
            if not rebalancing_needed:
                self.logger.info("No rebalancing needed")
                return self.current_composition
            
            # Calculate new optimal allocations
            # For now, using risk parity as default rebalancing method
            new_allocations = self._rebalance_risk_parity(
                current_allocations, new_strategy_stats, constraints
            )
            
            # Calculate rebalancing trades
            rebalancing_trades = self._calculate_rebalancing_trades(
                current_allocations, new_allocations
            )
            
            # Create rebalanced result
            composition_metrics = self._calculate_composition_metrics(
                new_allocations, new_strategy_stats
            )
            
            result = CompositionResult(
                allocations=new_allocations,
                total_expected_return=composition_metrics["expected_return"],
                total_expected_risk=composition_metrics["expected_risk"],
                risk_budget_utilization=self._calculate_risk_utilization(
                    new_allocations, {}  # Risk budgets would need to be recalculated
                ),
                composition_metrics=composition_metrics,
                rebalancing_trades=rebalancing_trades
            )
            
            # Update current composition
            self.current_composition = result
            
            self.logger.info(
                f"Composition rebalancing completed",
                trades_count=len(rebalancing_trades),
                total_turnover=sum(abs(trade) for trade in rebalancing_trades.values())
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Composition rebalancing failed: {str(e)}")
            raise StrategyError(
                f"Composition rebalancing failed: {str(e)}",
                error_code="REBALANCING_FAILED"
            ) from e
    
    def _validate_composition_inputs(
        self,
        strategies: Dict[str, IStrategy],
        constraints: Optional[CompositionConstraints]
    ) -> None:
        """Validate composition inputs."""
        if not strategies:
            raise ValidationError("Strategies dictionary cannot be empty")
        
        if len(strategies) < 2:
            raise ValidationError("At least 2 strategies required for composition")
        
        if constraints:
            if constraints.min_strategies and len(strategies) < constraints.min_strategies:
                raise ValidationError(
                    f"Number of strategies ({len(strategies)}) below minimum ({constraints.min_strategies})"
                )
            
            if constraints.max_strategies and len(strategies) > constraints.max_strategies:
                raise ValidationError(
                    f"Number of strategies ({len(strategies)}) above maximum ({constraints.max_strategies})"
                )
            
            if constraints.max_single_weight <= constraints.min_single_weight:
                raise ValidationError(
                    "max_single_weight must be greater than min_single_weight"
                )
    
    def _calculate_strategy_statistics(
        self,
        strategies: Dict[str, IStrategy],
        historical_data: Optional[pd.DataFrame],
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each strategy."""
        strategy_stats = {}
        
        for strategy_id, strategy in strategies.items():
            # This is a placeholder implementation
            # In practice, would calculate actual statistics from historical performance
            stats = {
                "expected_return": np.random.uniform(0.05, 0.15),  # Placeholder
                "volatility": np.random.uniform(0.10, 0.25),      # Placeholder
                "sharpe_ratio": np.random.uniform(0.5, 2.0),     # Placeholder
                "max_drawdown": np.random.uniform(0.05, 0.20),   # Placeholder
                "correlation_to_market": np.random.uniform(-0.5, 0.8)  # Placeholder
            }
            
            strategy_stats[strategy_id] = stats
        
        return strategy_stats
    
    def _calculate_risk_budgets(
        self,
        strategy_stats: Dict[str, Dict[str, float]],
        method: RiskBudgetingMethod,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate risk budgets for strategies."""
        if method == RiskBudgetingMethod.EQUAL_RISK:
            return self._equal_risk_budgets(strategy_stats)
        elif method == RiskBudgetingMethod.VOLATILITY_BASED:
            return self._volatility_based_budgets(strategy_stats)
        elif method == RiskBudgetingMethod.VAR_BASED:
            return self._var_based_budgets(strategy_stats, **kwargs)
        elif method == RiskBudgetingMethod.EXPECTED_SHORTFALL:
            return self._expected_shortfall_budgets(strategy_stats, **kwargs)
        elif method == RiskBudgetingMethod.CUSTOM:
            return kwargs.get("custom_budgets", self._equal_risk_budgets(strategy_stats))
        else:
            return self._equal_risk_budgets(strategy_stats)
    
    def _equal_risk_budgets(self, strategy_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate equal risk budgets."""
        num_strategies = len(strategy_stats)
        equal_budget = 1.0 / num_strategies
        return {strategy_id: equal_budget for strategy_id in strategy_stats.keys()}
    
    def _volatility_based_budgets(self, strategy_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate volatility-based risk budgets (inverse volatility weighting)."""
        inv_volatilities = {}
        total_inv_vol = 0
        
        for strategy_id, stats in strategy_stats.items():
            inv_vol = 1.0 / max(stats["volatility"], 0.01)  # Avoid division by zero
            inv_volatilities[strategy_id] = inv_vol
            total_inv_vol += inv_vol
        
        # Normalize to sum to 1
        return {
            strategy_id: inv_vol / total_inv_vol
            for strategy_id, inv_vol in inv_volatilities.items()
        }
    
    def _var_based_budgets(
        self,
        strategy_stats: Dict[str, Dict[str, float]],
        confidence_level: float = 0.05,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate VaR-based risk budgets (placeholder implementation)."""
        # This would implement actual VaR calculation
        # For now, using volatility as proxy
        return self._volatility_based_budgets(strategy_stats)
    
    def _expected_shortfall_budgets(
        self,
        strategy_stats: Dict[str, Dict[str, float]],
        confidence_level: float = 0.05,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate Expected Shortfall-based risk budgets (placeholder implementation)."""
        # This would implement actual Expected Shortfall calculation
        # For now, using volatility as proxy
        return self._volatility_based_budgets(strategy_stats)
    
    # Composition method implementations
    def _equal_weight_composition(
        self,
        strategies: Dict[str, IStrategy],
        strategy_stats: Dict[str, Dict[str, float]],
        risk_budgets: Dict[str, float],
        constraints: CompositionConstraints
    ) -> List[StrategyAllocation]:
        """Create equal weight composition."""
        num_strategies = len(strategies)
        equal_weight = 1.0 / num_strategies
        
        allocations = []
        for strategy_id in strategies.keys():
            stats = strategy_stats[strategy_id]
            allocation = StrategyAllocation(
                strategy_id=strategy_id,
                strategy_name=f"Strategy_{strategy_id}",
                weight=equal_weight,
                risk_budget=risk_budgets[strategy_id],
                expected_return=stats["expected_return"],
                expected_volatility=stats["volatility"],
                max_weight=constraints.max_single_weight,
                min_weight=constraints.min_single_weight
            )
            allocations.append(allocation)
        
        return allocations
    
    def _risk_parity_composition(
        self,
        strategies: Dict[str, IStrategy],
        strategy_stats: Dict[str, Dict[str, float]],
        risk_budgets: Dict[str, float],
        constraints: CompositionConstraints
    ) -> List[StrategyAllocation]:
        """Create risk parity composition."""
        # Simplified risk parity using inverse volatility
        inv_volatilities = {}
        total_inv_vol = 0
        
        for strategy_id, stats in strategy_stats.items():
            inv_vol = 1.0 / max(stats["volatility"], 0.01)
            inv_volatilities[strategy_id] = inv_vol
            total_inv_vol += inv_vol
        
        allocations = []
        for strategy_id in strategies.keys():
            stats = strategy_stats[strategy_id]
            weight = inv_volatilities[strategy_id] / total_inv_vol
            
            # Apply constraints
            weight = max(constraints.min_single_weight, 
                        min(constraints.max_single_weight, weight))
            
            allocation = StrategyAllocation(
                strategy_id=strategy_id,
                strategy_name=f"Strategy_{strategy_id}",
                weight=weight,
                risk_budget=risk_budgets[strategy_id],
                expected_return=stats["expected_return"],
                expected_volatility=stats["volatility"],
                max_weight=constraints.max_single_weight,
                min_weight=constraints.min_single_weight
            )
            allocations.append(allocation)
        
        # Normalize weights to sum to 1
        total_weight = sum(alloc.weight for alloc in allocations)
        for allocation in allocations:
            allocation.weight /= total_weight
        
        return allocations
    
    def _volatility_target_composition(
        self,
        strategies: Dict[str, IStrategy],
        strategy_stats: Dict[str, Dict[str, float]],
        risk_budgets: Dict[str, float],
        constraints: CompositionConstraints,
        target_volatility: float = 0.15,
        **kwargs
    ) -> List[StrategyAllocation]:
        """Create volatility-targeted composition."""
        target_vol = kwargs.get("target_volatility", target_volatility)
        if constraints.target_volatility:
            target_vol = constraints.target_volatility
        
        # Start with risk parity and scale to target volatility
        allocations = self._risk_parity_composition(
            strategies, strategy_stats, risk_budgets, constraints
        )
        
        # Calculate portfolio volatility (simplified - assumes zero correlation)
        portfolio_vol = np.sqrt(sum(
            (alloc.weight * alloc.expected_volatility) ** 2
            for alloc in allocations
        ))
        
        # Scale weights to achieve target volatility
        if portfolio_vol > 0:
            scale_factor = target_vol / portfolio_vol
            for allocation in allocations:
                allocation.weight *= scale_factor
        
        return allocations
    
    def _sharpe_optimal_composition(
        self,
        strategies: Dict[str, IStrategy],
        strategy_stats: Dict[str, Dict[str, float]],
        risk_budgets: Dict[str, float],
        constraints: CompositionConstraints
    ) -> List[StrategyAllocation]:
        """Create Sharpe ratio optimal composition."""
        # Weight by Sharpe ratio
        sharpe_ratios = {}
        total_sharpe = 0
        
        for strategy_id, stats in strategy_stats.items():
            sharpe = stats["sharpe_ratio"]
            sharpe_ratios[strategy_id] = max(sharpe, 0.01)  # Avoid negative weights
            total_sharpe += sharpe_ratios[strategy_id]
        
        allocations = []
        for strategy_id in strategies.keys():
            stats = strategy_stats[strategy_id]
            weight = sharpe_ratios[strategy_id] / total_sharpe
            
            # Apply constraints
            weight = max(constraints.min_single_weight, 
                        min(constraints.max_single_weight, weight))
            
            allocation = StrategyAllocation(
                strategy_id=strategy_id,
                strategy_name=f"Strategy_{strategy_id}",
                weight=weight,
                risk_budget=risk_budgets[strategy_id],
                expected_return=stats["expected_return"],
                expected_volatility=stats["volatility"],
                max_weight=constraints.max_single_weight,
                min_weight=constraints.min_single_weight
            )
            allocations.append(allocation)
        
        # Normalize weights
        total_weight = sum(alloc.weight for alloc in allocations)
        for allocation in allocations:
            allocation.weight /= total_weight
        
        return allocations
    
    def _hierarchical_composition(
        self,
        strategies: Dict[str, IStrategy],
        strategy_stats: Dict[str, Dict[str, float]],
        risk_budgets: Dict[str, float],
        constraints: CompositionConstraints,
        **kwargs
    ) -> List[StrategyAllocation]:
        """Create hierarchical composition (placeholder implementation)."""
        # This would implement hierarchical risk parity or similar
        # For now, falling back to risk parity
        self.logger.warning("Hierarchical composition not fully implemented, using risk parity")
        return self._risk_parity_composition(strategies, strategy_stats, risk_budgets, constraints)
    
    def _dynamic_composition(
        self,
        strategies: Dict[str, IStrategy],
        strategy_stats: Dict[str, Dict[str, float]],
        risk_budgets: Dict[str, float],
        constraints: CompositionConstraints,
        **kwargs
    ) -> List[StrategyAllocation]:
        """Create dynamic composition based on recent performance."""
        # This would implement dynamic allocation based on recent performance
        # For now, using performance-weighted allocation
        
        # Weight by recent Sharpe ratio with momentum adjustment
        allocations = []
        performance_weights = {}
        total_performance = 0
        
        for strategy_id, stats in strategy_stats.items():
            # Combine Sharpe ratio with momentum (placeholder)
            performance_score = stats["sharpe_ratio"] * (1 + stats["expected_return"])
            performance_weights[strategy_id] = max(performance_score, 0.01)
            total_performance += performance_weights[strategy_id]
        
        for strategy_id in strategies.keys():
            stats = strategy_stats[strategy_id]
            weight = performance_weights[strategy_id] / total_performance
            
            # Apply constraints
            weight = max(constraints.min_single_weight, 
                        min(constraints.max_single_weight, weight))
            
            allocation = StrategyAllocation(
                strategy_id=strategy_id,
                strategy_name=f"Strategy_{strategy_id}",
                weight=weight,
                risk_budget=risk_budgets[strategy_id],
                expected_return=stats["expected_return"],
                expected_volatility=stats["volatility"],
                max_weight=constraints.max_single_weight,
                min_weight=constraints.min_single_weight
            )
            allocations.append(allocation)
        
        # Normalize weights
        total_weight = sum(alloc.weight for alloc in allocations)
        for allocation in allocations:
            allocation.weight /= total_weight
        
        return allocations
    
    def _calculate_composition_metrics(
        self,
        allocations: List[StrategyAllocation],
        strategy_stats: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Calculate composition-level metrics."""
        # Portfolio expected return
        expected_return = sum(
            alloc.weight * alloc.expected_return for alloc in allocations
        )
        
        # Portfolio expected risk (simplified - assumes zero correlation)
        expected_risk = np.sqrt(sum(
            (alloc.weight * alloc.expected_volatility) ** 2
            for alloc in allocations
        ))
        
        # Portfolio Sharpe ratio
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0
        
        # Diversification metrics
        effective_strategies = 1 / sum(alloc.weight ** 2 for alloc in allocations)
        concentration = max(alloc.weight for alloc in allocations)
        
        return {
            "expected_return": expected_return,
            "expected_risk": expected_risk,
            "sharpe_ratio": sharpe_ratio,
            "effective_strategies": effective_strategies,
            "concentration": concentration,
            "num_strategies": len(allocations),
            "total_weight": sum(alloc.weight for alloc in allocations)
        }
    
    def _calculate_risk_utilization(
        self,
        allocations: List[StrategyAllocation],
        risk_budgets: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate risk budget utilization."""
        utilization = {}
        
        for allocation in allocations:
            if allocation.strategy_id in risk_budgets:
                budget = risk_budgets[allocation.strategy_id]
                # Simplified risk utilization calculation
                actual_risk_contribution = allocation.weight * allocation.expected_volatility
                total_risk = sum(
                    alloc.weight * alloc.expected_volatility for alloc in allocations
                )
                risk_contribution_pct = actual_risk_contribution / total_risk if total_risk > 0 else 0
                utilization[allocation.strategy_id] = risk_contribution_pct / budget if budget > 0 else 0
        
        return utilization
    
    def _check_rebalancing_trigger(
        self,
        current_allocations: List[StrategyAllocation],
        new_strategy_stats: Dict[str, Dict[str, float]],
        constraints: CompositionConstraints
    ) -> bool:
        """Check if rebalancing is needed."""
        # Check weight drift
        for allocation in current_allocations:
            # This is a simplified check - would need actual current weights
            # For now, assume some random drift
            weight_drift = np.random.uniform(-0.02, 0.02)  # Placeholder
            if abs(weight_drift) > constraints.rebalance_threshold:
                return True
        
        return False
    
    def _rebalance_risk_parity(
        self,
        current_allocations: List[StrategyAllocation],
        new_strategy_stats: Dict[str, Dict[str, float]],
        constraints: CompositionConstraints
    ) -> List[StrategyAllocation]:
        """Rebalance using risk parity method."""
        # Create strategy dictionary for rebalancing
        strategies = {alloc.strategy_id: None for alloc in current_allocations}
        risk_budgets = {alloc.strategy_id: alloc.risk_budget for alloc in current_allocations}
        
        return self._risk_parity_composition(
            strategies, new_strategy_stats, risk_budgets, constraints
        )
    
    def _calculate_rebalancing_trades(
        self,
        current_allocations: List[StrategyAllocation],
        new_allocations: List[StrategyAllocation]
    ) -> Dict[str, float]:
        """Calculate trades needed for rebalancing."""
        trades = {}
        
        # Create lookup for current weights
        current_weights = {alloc.strategy_id: alloc.weight for alloc in current_allocations}
        
        for new_alloc in new_allocations:
            current_weight = current_weights.get(new_alloc.strategy_id, 0.0)
            trade = new_alloc.weight - current_weight
            if abs(trade) > 1e-6:  # Only include meaningful trades
                trades[new_alloc.strategy_id] = trade
        
        return trades
    
    def get_composition_history(self) -> List[Dict[str, Any]]:
        """Get history of compositions."""
        return self.composition_history.copy()
    
    def get_current_composition(self) -> Optional[CompositionResult]:
        """Get current composition."""
        return self.current_composition
    
    def export_composition(self, result: CompositionResult, file_path: str) -> None:
        """Export composition result to file."""
        try:
            export_data = {
                "allocations": [
                    {
                        "strategy_id": alloc.strategy_id,
                        "strategy_name": alloc.strategy_name,
                        "weight": alloc.weight,
                        "risk_budget": alloc.risk_budget,
                        "expected_return": alloc.expected_return,
                        "expected_volatility": alloc.expected_volatility
                    }
                    for alloc in result.allocations
                ],
                "total_expected_return": result.total_expected_return,
                "total_expected_risk": result.total_expected_risk,
                "risk_budget_utilization": result.risk_budget_utilization,
                "composition_metrics": result.composition_metrics,
                "exported_at": datetime.now().isoformat()
            }
            
            import json
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported composition to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export composition: {str(e)}")
            raise StrategyError(
                f"Composition export failed: {str(e)}",
                error_code="EXPORT_FAILED"
            ) from e