"""
Comprehensive portfolio optimization constraints management system.

This module provides a sophisticated constraint management system for portfolio optimization
including position limits, sector constraints, turnover limits, ESG constraints, and
dynamic constraint adjustment based on market conditions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Enumeration of constraint types for categorization."""
    POSITION_LIMIT = "position_limit"
    SECTOR_LIMIT = "sector_limit"
    TURNOVER_LIMIT = "turnover_limit"
    ESG_CONSTRAINT = "esg_constraint"
    LEVERAGE_CONSTRAINT = "leverage_constraint"
    CONCENTRATION_CONSTRAINT = "concentration_constraint"
    TRANSACTION_COST = "transaction_cost"
    DYNAMIC_CONSTRAINT = "dynamic_constraint"


class ConstraintSeverity(Enum):
    """Constraint violation severity levels."""
    SOFT = "soft"  # Warning, can be violated with penalty
    HARD = "hard"  # Must be satisfied, optimization fails if violated
    CRITICAL = "critical"  # System-level constraint, cannot be overridden


@dataclass
class ConstraintViolation:
    """Information about a constraint violation."""
    constraint_name: str
    violation_type: str
    severity: ConstraintSeverity
    current_value: float
    limit_value: float
    violation_amount: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketCondition:
    """Market condition data for dynamic constraint adjustment."""
    volatility: float
    liquidity: float
    market_stress: float
    correlation_regime: str  # 'low', 'medium', 'high'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Constraint(ABC):
    """Enhanced abstract base class for portfolio optimization constraints."""
    
    name: str
    description: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    enabled: bool = True
    priority: int = 1  # Higher number = higher priority
    
    # Dynamic adjustment parameters
    is_dynamic: bool = False
    market_condition_sensitivity: float = 0.0  # 0 = no sensitivity, 1 = full sensitivity
    
    @abstractmethod
    def apply(self, weights: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Apply the constraint and return constraint parameters.
        
        Args:
            weights: Portfolio weights array
            **kwargs: Additional parameters needed for constraint
            
        Returns:
            Dictionary containing constraint parameters for optimization
        """
        pass
    
    @abstractmethod
    def validate(self, weights: np.ndarray, **kwargs) -> bool:
        """
        Validate if the given weights satisfy the constraint.
        
        Args:
            weights: Portfolio weights array
            **kwargs: Additional parameters needed for validation
            
        Returns:
            True if constraint is satisfied, False otherwise
        """
        pass
    
    def check_violation(self, weights: np.ndarray, **kwargs) -> Optional[ConstraintViolation]:
        """
        Check for constraint violations and return detailed information.
        
        Args:
            weights: Portfolio weights array
            **kwargs: Additional parameters needed for validation
            
        Returns:
            ConstraintViolation object if violated, None otherwise
        """
        if self.validate(weights, **kwargs):
            return None
        
        # Default implementation - subclasses should override for detailed info
        return ConstraintViolation(
            constraint_name=self.name,
            violation_type=self.constraint_type.value,
            severity=self.severity,
            current_value=0.0,
            limit_value=0.0,
            violation_amount=0.0,
            message=f"Constraint {self.name} violated"
        )
    
    def adjust_for_market_conditions(self, market_condition: MarketCondition) -> 'Constraint':
        """
        Adjust constraint parameters based on market conditions.
        
        Args:
            market_condition: Current market condition data
            
        Returns:
            Adjusted constraint (may return self if no adjustment needed)
        """
        if not self.is_dynamic or self.market_condition_sensitivity == 0.0:
            return self
        
        # Default implementation - subclasses should override
        return self


@dataclass
class PositionLimitConstraint(Constraint):
    """Enhanced constraint for individual position limits with dynamic adjustment."""
    
    min_weight: float = 0.0
    max_weight: float = 1.0
    
    # Asset-specific limits (optional)
    asset_specific_limits: Optional[Dict[str, tuple]] = None  # asset -> (min, max)
    
    # Dynamic adjustment parameters
    volatility_adjustment_factor: float = 0.0  # Reduce limits for high volatility assets
    liquidity_adjustment_factor: float = 0.0  # Reduce limits for low liquidity assets
    
    def __post_init__(self):
        if self.min_weight < 0:
            raise ValueError("Minimum weight cannot be negative")
        if self.max_weight > 1:
            raise ValueError("Maximum weight cannot exceed 1")
        if self.min_weight >= self.max_weight:
            raise ValueError("Minimum weight must be less than maximum weight")
    
    def apply(self, weights: np.ndarray, asset_names: Optional[List[str]] = None, 
              asset_volatilities: Optional[Dict[str, float]] = None,
              asset_liquidities: Optional[Dict[str, float]] = None, **kwargs) -> Dict[str, Any]:
        """Apply position limit constraints with dynamic adjustment."""
        n_assets = len(weights)
        bounds = []
        
        for i in range(n_assets):
            min_w, max_w = self.min_weight, self.max_weight
            
            # Apply asset-specific limits if available
            if self.asset_specific_limits and asset_names and i < len(asset_names):
                asset = asset_names[i]
                if asset in self.asset_specific_limits:
                    asset_min, asset_max = self.asset_specific_limits[asset]
                    min_w = max(min_w, asset_min)
                    max_w = min(max_w, asset_max)
            
            # Apply dynamic adjustments
            if asset_names and i < len(asset_names):
                asset = asset_names[i]
                
                # Volatility adjustment
                if asset_volatilities and asset in asset_volatilities and self.volatility_adjustment_factor > 0:
                    vol_adjustment = 1.0 - (asset_volatilities[asset] * self.volatility_adjustment_factor)
                    max_w *= max(0.1, vol_adjustment)  # Don't reduce below 10% of original limit
                
                # Liquidity adjustment
                if asset_liquidities and asset in asset_liquidities and self.liquidity_adjustment_factor > 0:
                    liq_adjustment = asset_liquidities[asset] * self.liquidity_adjustment_factor
                    max_w *= max(0.1, liq_adjustment)
            
            bounds.append((min_w, max_w))
        
        return {
            'type': 'bounds',
            'bounds': bounds
        }
    
    def validate(self, weights: np.ndarray, asset_names: Optional[List[str]] = None,
                 asset_volatilities: Optional[Dict[str, float]] = None,
                 asset_liquidities: Optional[Dict[str, float]] = None, **kwargs) -> bool:
        """Validate position limits with dynamic adjustment."""
        constraint_params = self.apply(weights, asset_names, asset_volatilities, asset_liquidities, **kwargs)
        bounds = constraint_params['bounds']
        
        for i, (min_w, max_w) in enumerate(bounds):
            if not (min_w <= weights[i] <= max_w):
                return False
        return True
    
    def check_violation(self, weights: np.ndarray, asset_names: Optional[List[str]] = None,
                       asset_volatilities: Optional[Dict[str, float]] = None,
                       asset_liquidities: Optional[Dict[str, float]] = None, **kwargs) -> Optional[ConstraintViolation]:
        """Check for position limit violations with detailed information."""
        constraint_params = self.apply(weights, asset_names, asset_volatilities, asset_liquidities, **kwargs)
        bounds = constraint_params['bounds']
        
        violations = []
        for i, (min_w, max_w) in enumerate(bounds):
            if weights[i] < min_w:
                asset_name = asset_names[i] if asset_names and i < len(asset_names) else f"Asset_{i}"
                violations.append(ConstraintViolation(
                    constraint_name=f"{self.name} - {asset_name}",
                    violation_type="position_limit_min",
                    severity=self.severity,
                    current_value=weights[i],
                    limit_value=min_w,
                    violation_amount=min_w - weights[i],
                    message=f"Position {asset_name} weight {weights[i]:.4f} below minimum {min_w:.4f}"
                ))
            elif weights[i] > max_w:
                asset_name = asset_names[i] if asset_names and i < len(asset_names) else f"Asset_{i}"
                violations.append(ConstraintViolation(
                    constraint_name=f"{self.name} - {asset_name}",
                    violation_type="position_limit_max",
                    severity=self.severity,
                    current_value=weights[i],
                    limit_value=max_w,
                    violation_amount=weights[i] - max_w,
                    message=f"Position {asset_name} weight {weights[i]:.4f} above maximum {max_w:.4f}"
                ))
        
        return violations[0] if violations else None
    
    def adjust_for_market_conditions(self, market_condition: MarketCondition) -> 'PositionLimitConstraint':
        """Adjust position limits based on market conditions."""
        if not self.is_dynamic or self.market_condition_sensitivity == 0.0:
            return self
        
        # Create adjusted constraint
        adjusted = PositionLimitConstraint(
            name=f"{self.name} (Market Adjusted)",
            description=f"{self.description} - Adjusted for market conditions",
            constraint_type=self.constraint_type,
            severity=self.severity,
            enabled=self.enabled,
            priority=self.priority,
            is_dynamic=self.is_dynamic,
            market_condition_sensitivity=self.market_condition_sensitivity,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            asset_specific_limits=self.asset_specific_limits,
            volatility_adjustment_factor=self.volatility_adjustment_factor,
            liquidity_adjustment_factor=self.liquidity_adjustment_factor
        )
        
        # Adjust limits based on market stress
        stress_factor = 1.0 - (market_condition.market_stress * self.market_condition_sensitivity)
        adjusted.max_weight *= max(0.5, stress_factor)  # Don't reduce below 50% of original
        
        # Adjust based on volatility regime
        if market_condition.volatility > 0.3:  # High volatility regime
            vol_factor = 1.0 - (market_condition.volatility * self.market_condition_sensitivity * 0.5)
            adjusted.max_weight *= max(0.7, vol_factor)
        
        return adjusted


@dataclass
class SectorConstraint(Constraint):
    """Enhanced constraint for sector exposure limits with dynamic adjustment."""
    
    # Enhanced features
    sector_neutrality: Optional[Dict[str, float]] = None  # sector -> target weight for neutrality
    correlation_adjustment: bool = False  # Adjust limits based on sector correlations
    
    # Required fields (must come after optional fields in dataclass)
    sector_mapping: Dict[str, str] = field(default_factory=dict)  # asset -> sector mapping
    sector_limits: Dict[str, tuple] = field(default_factory=dict)  # sector -> (min_weight, max_weight)
    
    def __post_init__(self):
        # Validate sector limits
        for sector, (min_w, max_w) in self.sector_limits.items():
            if min_w < 0 or max_w > 1 or min_w > max_w:
                raise ValueError(f"Invalid sector limits for {sector}: ({min_w}, {max_w})")
    
    def apply(self, weights: np.ndarray, asset_names: List[str], 
              sector_correlations: Optional[Dict[str, Dict[str, float]]] = None, **kwargs) -> Dict[str, Any]:
        """Apply sector constraints with enhanced features."""
        constraints = []
        
        for sector, (min_weight, max_weight) in self.sector_limits.items():
            # Find assets in this sector
            sector_indices = [
                i for i, asset in enumerate(asset_names)
                if self.sector_mapping.get(asset) == sector
            ]
            
            if sector_indices:
                # Adjust limits based on correlations if enabled
                adjusted_min, adjusted_max = min_weight, max_weight
                
                if self.correlation_adjustment and sector_correlations:
                    # Reduce limits for highly correlated sectors
                    avg_correlation = self._calculate_average_sector_correlation(
                        sector, sector_correlations
                    )
                    if avg_correlation > 0.7:  # High correlation threshold
                        correlation_factor = 1.0 - (avg_correlation - 0.7) * 0.5
                        adjusted_max *= correlation_factor
                
                # Create constraint matrix for this sector
                A_sector = np.zeros(len(asset_names))
                A_sector[sector_indices] = 1
                
                constraints.extend([
                    {'type': 'ineq', 'fun': lambda w, A=A_sector, limit=adjusted_max: limit - np.dot(A, w)},
                    {'type': 'ineq', 'fun': lambda w, A=A_sector, limit=adjusted_min: np.dot(A, w) - limit}
                ])
        
        # Add sector neutrality constraints if specified
        if self.sector_neutrality:
            for sector, target_weight in self.sector_neutrality.items():
                sector_indices = [
                    i for i, asset in enumerate(asset_names)
                    if self.sector_mapping.get(asset) == sector
                ]
                
                if sector_indices:
                    A_sector = np.zeros(len(asset_names))
                    A_sector[sector_indices] = 1
                    
                    # Soft constraint for neutrality (can be violated with penalty)
                    constraints.append({
                        'type': 'ineq', 
                        'fun': lambda w, A=A_sector, target=target_weight: 0.05 - abs(np.dot(A, w) - target)
                    })
        
        return {'constraints': constraints}
    
    def validate(self, weights: np.ndarray, asset_names: List[str], 
                 sector_correlations: Optional[Dict[str, Dict[str, float]]] = None, **kwargs) -> bool:
        """Validate sector constraints with enhanced features."""
        for sector, (min_weight, max_weight) in self.sector_limits.items():
            sector_weight = sum(
                weights[i] for i, asset in enumerate(asset_names)
                if self.sector_mapping.get(asset) == sector
            )
            
            # Adjust limits based on correlations if enabled
            adjusted_min, adjusted_max = min_weight, max_weight
            if self.correlation_adjustment and sector_correlations:
                avg_correlation = self._calculate_average_sector_correlation(
                    sector, sector_correlations
                )
                if avg_correlation > 0.7:
                    correlation_factor = 1.0 - (avg_correlation - 0.7) * 0.5
                    adjusted_max *= correlation_factor
            
            if not (adjusted_min <= sector_weight <= adjusted_max):
                return False
        
        return True
    
    def check_violation(self, weights: np.ndarray, asset_names: List[str],
                       sector_correlations: Optional[Dict[str, Dict[str, float]]] = None, **kwargs) -> Optional[ConstraintViolation]:
        """Check for sector constraint violations with detailed information."""
        for sector, (min_weight, max_weight) in self.sector_limits.items():
            sector_weight = sum(
                weights[i] for i, asset in enumerate(asset_names)
                if self.sector_mapping.get(asset) == sector
            )
            
            # Adjust limits based on correlations if enabled
            adjusted_min, adjusted_max = min_weight, max_weight
            if self.correlation_adjustment and sector_correlations:
                avg_correlation = self._calculate_average_sector_correlation(
                    sector, sector_correlations
                )
                if avg_correlation > 0.7:
                    correlation_factor = 1.0 - (avg_correlation - 0.7) * 0.5
                    adjusted_max *= correlation_factor
            
            if sector_weight < adjusted_min:
                return ConstraintViolation(
                    constraint_name=f"{self.name} - {sector}",
                    violation_type="sector_limit_min",
                    severity=self.severity,
                    current_value=sector_weight,
                    limit_value=adjusted_min,
                    violation_amount=adjusted_min - sector_weight,
                    message=f"Sector {sector} weight {sector_weight:.4f} below minimum {adjusted_min:.4f}"
                )
            elif sector_weight > adjusted_max:
                return ConstraintViolation(
                    constraint_name=f"{self.name} - {sector}",
                    violation_type="sector_limit_max",
                    severity=self.severity,
                    current_value=sector_weight,
                    limit_value=adjusted_max,
                    violation_amount=sector_weight - adjusted_max,
                    message=f"Sector {sector} weight {sector_weight:.4f} above maximum {adjusted_max:.4f}"
                )
        
        return None
    
    def _calculate_average_sector_correlation(self, sector: str, 
                                            sector_correlations: Dict[str, Dict[str, float]]) -> float:
        """Calculate average correlation of a sector with other sectors."""
        if sector not in sector_correlations:
            return 0.0
        
        correlations = [
            corr for other_sector, corr in sector_correlations[sector].items()
            if other_sector != sector
        ]
        
        return np.mean(correlations) if correlations else 0.0
    
    def get_sector_exposures(self, weights: np.ndarray, asset_names: List[str]) -> Dict[str, float]:
        """Calculate current sector exposures."""
        sector_exposures = {}
        
        for sector in self.sector_limits.keys():
            sector_weight = sum(
                weights[i] for i, asset in enumerate(asset_names)
                if self.sector_mapping.get(asset) == sector
            )
            sector_exposures[sector] = sector_weight
        
        return sector_exposures
    
    def adjust_for_market_conditions(self, market_condition: MarketCondition) -> 'SectorConstraint':
        """Adjust sector limits based on market conditions."""
        if not self.is_dynamic or self.market_condition_sensitivity == 0.0:
            return self
        
        # Create adjusted constraint
        adjusted_limits = {}
        
        for sector, (min_w, max_w) in self.sector_limits.items():
            # Reduce sector concentration during high stress periods
            if market_condition.market_stress > 0.5:
                stress_factor = 1.0 - (market_condition.market_stress * self.market_condition_sensitivity * 0.3)
                adjusted_max = max_w * max(0.7, stress_factor)
            else:
                adjusted_max = max_w
            
            # Adjust based on correlation regime
            if market_condition.correlation_regime == 'high':
                correlation_factor = 1.0 - (self.market_condition_sensitivity * 0.2)
                adjusted_max *= correlation_factor
            
            adjusted_limits[sector] = (min_w, adjusted_max)
        
        return SectorConstraint(
            name=f"{self.name} (Market Adjusted)",
            description=f"{self.description} - Adjusted for market conditions",
            constraint_type=self.constraint_type,
            severity=self.severity,
            enabled=self.enabled,
            priority=self.priority,
            is_dynamic=self.is_dynamic,
            market_condition_sensitivity=self.market_condition_sensitivity,
            sector_mapping=self.sector_mapping,
            sector_limits=adjusted_limits,
            sector_neutrality=self.sector_neutrality,
            correlation_adjustment=self.correlation_adjustment
        )


@dataclass
class TurnoverConstraint(Constraint):
    """Enhanced constraint for portfolio turnover limits with transaction cost integration."""
    
    # Transaction cost integration
    transaction_cost_model: Optional[Callable[[np.ndarray], float]] = None
    max_transaction_cost: Optional[float] = None
    
    # Enhanced turnover controls
    one_way_turnover: bool = False  # If True, limit one-way turnover instead of total
    asset_specific_turnover_limits: Optional[Dict[str, float]] = None
    current_weights: Optional[np.ndarray] = None
    
    # Required fields
    max_turnover: float = 1.0
    
    def __post_init__(self):
        if self.max_turnover < 0:
            raise ValueError("Maximum turnover cannot be negative")
        if self.max_turnover > 2:
            raise ValueError("Maximum turnover cannot exceed 2 (200%)")
    
    def apply(self, weights: np.ndarray, asset_names: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Apply turnover constraint with enhanced features."""
        if self.current_weights is None:
            return {}
        
        constraints = []
        
        # Main turnover constraint
        def turnover_constraint(w):
            if self.one_way_turnover:
                # One-way turnover (sum of positive trades only)
                trades = w - self.current_weights
                return self.max_turnover - np.sum(np.maximum(trades, 0))
            else:
                # Total turnover (sum of absolute trades)
                return self.max_turnover - np.sum(np.abs(w - self.current_weights))
        
        constraints.append({'type': 'ineq', 'fun': turnover_constraint})
        
        # Asset-specific turnover limits
        if self.asset_specific_turnover_limits and asset_names:
            for i, asset in enumerate(asset_names):
                if asset in self.asset_specific_turnover_limits:
                    asset_limit = self.asset_specific_turnover_limits[asset]
                    
                    def asset_turnover_constraint(w, idx=i, limit=asset_limit):
                        return limit - abs(w[idx] - self.current_weights[idx])
                    
                    constraints.append({'type': 'ineq', 'fun': asset_turnover_constraint})
        
        # Transaction cost constraint
        if self.transaction_cost_model and self.max_transaction_cost:
            def transaction_cost_constraint(w):
                trades = w - self.current_weights
                cost = self.transaction_cost_model(trades)
                return self.max_transaction_cost - cost
            
            constraints.append({'type': 'ineq', 'fun': transaction_cost_constraint})
        
        return {'constraints': constraints}
    
    def validate(self, weights: np.ndarray, asset_names: Optional[List[str]] = None, **kwargs) -> bool:
        """Validate turnover constraint with enhanced features."""
        if self.current_weights is None:
            return True
        
        # Check main turnover constraint
        if self.one_way_turnover:
            trades = weights - self.current_weights
            turnover = np.sum(np.maximum(trades, 0))
        else:
            turnover = np.sum(np.abs(weights - self.current_weights))
        
        if turnover > self.max_turnover:
            return False
        
        # Check asset-specific turnover limits
        if self.asset_specific_turnover_limits and asset_names:
            for i, asset in enumerate(asset_names):
                if asset in self.asset_specific_turnover_limits:
                    asset_turnover = abs(weights[i] - self.current_weights[i])
                    if asset_turnover > self.asset_specific_turnover_limits[asset]:
                        return False
        
        # Check transaction cost constraint
        if self.transaction_cost_model and self.max_transaction_cost:
            trades = weights - self.current_weights
            cost = self.transaction_cost_model(trades)
            if cost > self.max_transaction_cost:
                return False
        
        return True
    
    def check_violation(self, weights: np.ndarray, asset_names: Optional[List[str]] = None, **kwargs) -> Optional[ConstraintViolation]:
        """Check for turnover constraint violations with detailed information."""
        if self.current_weights is None:
            return None
        
        # Check main turnover constraint
        if self.one_way_turnover:
            trades = weights - self.current_weights
            turnover = np.sum(np.maximum(trades, 0))
            turnover_type = "one_way_turnover"
        else:
            turnover = np.sum(np.abs(weights - self.current_weights))
            turnover_type = "total_turnover"
        
        if turnover > self.max_turnover:
            return ConstraintViolation(
                constraint_name=self.name,
                violation_type=turnover_type,
                severity=self.severity,
                current_value=turnover,
                limit_value=self.max_turnover,
                violation_amount=turnover - self.max_turnover,
                message=f"Portfolio {turnover_type} {turnover:.4f} exceeds limit {self.max_turnover:.4f}"
            )
        
        # Check asset-specific turnover limits
        if self.asset_specific_turnover_limits and asset_names:
            for i, asset in enumerate(asset_names):
                if asset in self.asset_specific_turnover_limits:
                    asset_turnover = abs(weights[i] - self.current_weights[i])
                    asset_limit = self.asset_specific_turnover_limits[asset]
                    
                    if asset_turnover > asset_limit:
                        return ConstraintViolation(
                            constraint_name=f"{self.name} - {asset}",
                            violation_type="asset_turnover",
                            severity=self.severity,
                            current_value=asset_turnover,
                            limit_value=asset_limit,
                            violation_amount=asset_turnover - asset_limit,
                            message=f"Asset {asset} turnover {asset_turnover:.4f} exceeds limit {asset_limit:.4f}"
                        )
        
        # Check transaction cost constraint
        if self.transaction_cost_model and self.max_transaction_cost:
            trades = weights - self.current_weights
            cost = self.transaction_cost_model(trades)
            
            if cost > self.max_transaction_cost:
                return ConstraintViolation(
                    constraint_name=f"{self.name} - Transaction Cost",
                    violation_type="transaction_cost",
                    severity=self.severity,
                    current_value=cost,
                    limit_value=self.max_transaction_cost,
                    violation_amount=cost - self.max_transaction_cost,
                    message=f"Transaction cost {cost:.4f} exceeds limit {self.max_transaction_cost:.4f}"
                )
        
        return None
    
    def calculate_turnover_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate various turnover metrics."""
        if self.current_weights is None:
            return {}
        
        trades = weights - self.current_weights
        
        return {
            'total_turnover': float(np.sum(np.abs(trades))),
            'one_way_turnover': float(np.sum(np.maximum(trades, 0))),
            'buy_turnover': float(np.sum(np.maximum(trades, 0))),
            'sell_turnover': float(np.sum(np.maximum(-trades, 0))),
            'net_turnover': float(np.sum(trades)),
            'max_individual_trade': float(np.max(np.abs(trades))),
            'num_trades': int(np.sum(np.abs(trades) > 1e-6))
        }
    
    def adjust_for_market_conditions(self, market_condition: MarketCondition) -> 'TurnoverConstraint':
        """Adjust turnover limits based on market conditions."""
        if not self.is_dynamic or self.market_condition_sensitivity == 0.0:
            return self
        
        # Reduce turnover limits during high volatility or low liquidity
        volatility_factor = 1.0 - (market_condition.volatility * self.market_condition_sensitivity * 0.5)
        liquidity_factor = market_condition.liquidity  # Assume liquidity is normalized 0-1
        stress_factor = 1.0 - (market_condition.market_stress * self.market_condition_sensitivity * 0.3)
        
        adjustment_factor = min(volatility_factor, liquidity_factor, stress_factor)
        adjusted_max_turnover = self.max_turnover * max(0.3, adjustment_factor)
        
        return TurnoverConstraint(
            name=f"{self.name} (Market Adjusted)",
            description=f"{self.description} - Adjusted for market conditions",
            constraint_type=self.constraint_type,
            severity=self.severity,
            enabled=self.enabled,
            priority=self.priority,
            is_dynamic=self.is_dynamic,
            market_condition_sensitivity=self.market_condition_sensitivity,
            max_turnover=adjusted_max_turnover,
            current_weights=self.current_weights,
            transaction_cost_model=self.transaction_cost_model,
            max_transaction_cost=self.max_transaction_cost,
            one_way_turnover=self.one_way_turnover,
            asset_specific_turnover_limits=self.asset_specific_turnover_limits
        )


@dataclass
class ESGConstraint(Constraint):
    """Enhanced constraint for ESG (Environmental, Social, Governance) and sustainable investing criteria."""
    
    # Component-specific constraints
    environmental_scores: Optional[Dict[str, float]] = None
    social_scores: Optional[Dict[str, float]] = None
    governance_scores: Optional[Dict[str, float]] = None
    min_environmental_score: Optional[float] = None
    min_social_score: Optional[float] = None
    min_governance_score: Optional[float] = None
    
    # Exclusion criteria
    exclude_assets: Optional[List[str]] = None
    exclude_sectors: Optional[List[str]] = None
    exclude_countries: Optional[List[str]] = None
    
    # Sustainable investing themes
    green_assets: Optional[List[str]] = None  # Assets with green/sustainable focus
    min_green_allocation: Optional[float] = None
    
    # Carbon footprint constraints
    carbon_intensities: Optional[Dict[str, float]] = None  # asset -> carbon intensity
    max_portfolio_carbon_intensity: Optional[float] = None
    
    # Controversy screening
    controversy_scores: Optional[Dict[str, float]] = None  # asset -> controversy score (lower is better)
    max_controversy_score: Optional[float] = None
    
    # Basic ESG scoring (required fields)
    esg_scores: Dict[str, float] = field(default_factory=dict)  # asset -> ESG score mapping (0-100)
    min_portfolio_score: float = 0.0
    
    def __post_init__(self):
        if not 0 <= self.min_portfolio_score <= 100:
            raise ValueError("ESG score must be between 0 and 100")
    
    def apply(self, weights: np.ndarray, asset_names: List[str], 
              asset_sectors: Optional[Dict[str, str]] = None,
              asset_countries: Optional[Dict[str, str]] = None, **kwargs) -> Dict[str, Any]:
        """Apply comprehensive ESG constraints."""
        constraints = []
        
        # Asset exclusions
        if self.exclude_assets:
            for i, asset in enumerate(asset_names):
                if asset in self.exclude_assets:
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda w, idx=i: w[idx]  # Force weight to be 0
                    })
        
        # Sector exclusions
        if self.exclude_sectors and asset_sectors:
            for i, asset in enumerate(asset_names):
                if asset_sectors.get(asset) in self.exclude_sectors:
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda w, idx=i: w[idx]
                    })
        
        # Country exclusions
        if self.exclude_countries and asset_countries:
            for i, asset in enumerate(asset_names):
                if asset_countries.get(asset) in self.exclude_countries:
                    constraints.append({
                        'type': 'eq',
                        'fun': lambda w, idx=i: w[idx]
                    })
        
        # Overall ESG score constraint
        esg_scores_array = np.array([
            self.esg_scores.get(asset, 0) for asset in asset_names
        ])
        
        def esg_constraint(w):
            portfolio_score = np.dot(w, esg_scores_array)
            return portfolio_score - self.min_portfolio_score
        
        constraints.append({'type': 'ineq', 'fun': esg_constraint})
        
        # Component-specific ESG constraints
        if self.environmental_scores and self.min_environmental_score:
            env_scores_array = np.array([
                self.environmental_scores.get(asset, 0) for asset in asset_names
            ])
            
            def env_constraint(w):
                return np.dot(w, env_scores_array) - self.min_environmental_score
            
            constraints.append({'type': 'ineq', 'fun': env_constraint})
        
        if self.social_scores and self.min_social_score:
            social_scores_array = np.array([
                self.social_scores.get(asset, 0) for asset in asset_names
            ])
            
            def social_constraint(w):
                return np.dot(w, social_scores_array) - self.min_social_score
            
            constraints.append({'type': 'ineq', 'fun': social_constraint})
        
        if self.governance_scores and self.min_governance_score:
            gov_scores_array = np.array([
                self.governance_scores.get(asset, 0) for asset in asset_names
            ])
            
            def gov_constraint(w):
                return np.dot(w, gov_scores_array) - self.min_governance_score
            
            constraints.append({'type': 'ineq', 'fun': gov_constraint})
        
        # Green/sustainable allocation constraint
        if self.green_assets and self.min_green_allocation:
            green_indices = [
                i for i, asset in enumerate(asset_names)
                if asset in self.green_assets
            ]
            
            if green_indices:
                def green_constraint(w):
                    green_allocation = sum(w[i] for i in green_indices)
                    return green_allocation - self.min_green_allocation
                
                constraints.append({'type': 'ineq', 'fun': green_constraint})
        
        # Carbon intensity constraint
        if self.carbon_intensities and self.max_portfolio_carbon_intensity:
            carbon_array = np.array([
                self.carbon_intensities.get(asset, 0) for asset in asset_names
            ])
            
            def carbon_constraint(w):
                portfolio_carbon = np.dot(w, carbon_array)
                return self.max_portfolio_carbon_intensity - portfolio_carbon
            
            constraints.append({'type': 'ineq', 'fun': carbon_constraint})
        
        # Controversy screening constraint
        if self.controversy_scores and self.max_controversy_score:
            controversy_array = np.array([
                self.controversy_scores.get(asset, 0) for asset in asset_names
            ])
            
            def controversy_constraint(w):
                portfolio_controversy = np.dot(w, controversy_array)
                return self.max_controversy_score - portfolio_controversy
            
            constraints.append({'type': 'ineq', 'fun': controversy_constraint})
        
        return {'constraints': constraints}
    
    def validate(self, weights: np.ndarray, asset_names: List[str],
                 asset_sectors: Optional[Dict[str, str]] = None,
                 asset_countries: Optional[Dict[str, str]] = None, **kwargs) -> bool:
        """Validate comprehensive ESG constraints."""
        # Check asset exclusions
        if self.exclude_assets:
            for i, asset in enumerate(asset_names):
                if asset in self.exclude_assets and weights[i] > 1e-6:
                    return False
        
        # Check sector exclusions
        if self.exclude_sectors and asset_sectors:
            for i, asset in enumerate(asset_names):
                if asset_sectors.get(asset) in self.exclude_sectors and weights[i] > 1e-6:
                    return False
        
        # Check country exclusions
        if self.exclude_countries and asset_countries:
            for i, asset in enumerate(asset_names):
                if asset_countries.get(asset) in self.exclude_countries and weights[i] > 1e-6:
                    return False
        
        # Check overall ESG score
        portfolio_score = sum(
            weights[i] * self.esg_scores.get(asset, 0)
            for i, asset in enumerate(asset_names)
        )
        if portfolio_score < self.min_portfolio_score:
            return False
        
        # Check component-specific scores
        if self.environmental_scores and self.min_environmental_score:
            env_score = sum(
                weights[i] * self.environmental_scores.get(asset, 0)
                for i, asset in enumerate(asset_names)
            )
            if env_score < self.min_environmental_score:
                return False
        
        if self.social_scores and self.min_social_score:
            social_score = sum(
                weights[i] * self.social_scores.get(asset, 0)
                for i, asset in enumerate(asset_names)
            )
            if social_score < self.min_social_score:
                return False
        
        if self.governance_scores and self.min_governance_score:
            gov_score = sum(
                weights[i] * self.governance_scores.get(asset, 0)
                for i, asset in enumerate(asset_names)
            )
            if gov_score < self.min_governance_score:
                return False
        
        # Check green allocation
        if self.green_assets and self.min_green_allocation:
            green_allocation = sum(
                weights[i] for i, asset in enumerate(asset_names)
                if asset in self.green_assets
            )
            if green_allocation < self.min_green_allocation:
                return False
        
        # Check carbon intensity
        if self.carbon_intensities and self.max_portfolio_carbon_intensity:
            portfolio_carbon = sum(
                weights[i] * self.carbon_intensities.get(asset, 0)
                for i, asset in enumerate(asset_names)
            )
            if portfolio_carbon > self.max_portfolio_carbon_intensity:
                return False
        
        # Check controversy score
        if self.controversy_scores and self.max_controversy_score:
            portfolio_controversy = sum(
                weights[i] * self.controversy_scores.get(asset, 0)
                for i, asset in enumerate(asset_names)
            )
            if portfolio_controversy > self.max_controversy_score:
                return False
        
        return True
    
    def calculate_esg_metrics(self, weights: np.ndarray, asset_names: List[str]) -> Dict[str, float]:
        """Calculate comprehensive ESG metrics for the portfolio."""
        metrics = {}
        
        # Overall ESG score
        metrics['esg_score'] = sum(
            weights[i] * self.esg_scores.get(asset, 0)
            for i, asset in enumerate(asset_names)
        )
        
        # Component scores
        if self.environmental_scores:
            metrics['environmental_score'] = sum(
                weights[i] * self.environmental_scores.get(asset, 0)
                for i, asset in enumerate(asset_names)
            )
        
        if self.social_scores:
            metrics['social_score'] = sum(
                weights[i] * self.social_scores.get(asset, 0)
                for i, asset in enumerate(asset_names)
            )
        
        if self.governance_scores:
            metrics['governance_score'] = sum(
                weights[i] * self.governance_scores.get(asset, 0)
                for i, asset in enumerate(asset_names)
            )
        
        # Green allocation
        if self.green_assets:
            metrics['green_allocation'] = sum(
                weights[i] for i, asset in enumerate(asset_names)
                if asset in self.green_assets
            )
        
        # Carbon intensity
        if self.carbon_intensities:
            metrics['carbon_intensity'] = sum(
                weights[i] * self.carbon_intensities.get(asset, 0)
                for i, asset in enumerate(asset_names)
            )
        
        # Controversy score
        if self.controversy_scores:
            metrics['controversy_score'] = sum(
                weights[i] * self.controversy_scores.get(asset, 0)
                for i, asset in enumerate(asset_names)
            )
        
        return metrics
    
    def check_violation(self, weights: np.ndarray, asset_names: List[str],
                       asset_sectors: Optional[Dict[str, str]] = None,
                       asset_countries: Optional[Dict[str, str]] = None, **kwargs) -> Optional[ConstraintViolation]:
        """Check for ESG constraint violations with detailed information."""
        # Check overall ESG score first
        portfolio_score = sum(
            weights[i] * self.esg_scores.get(asset, 0)
            for i, asset in enumerate(asset_names)
        )
        
        if portfolio_score < self.min_portfolio_score:
            return ConstraintViolation(
                constraint_name=self.name,
                violation_type="esg_score",
                severity=self.severity,
                current_value=portfolio_score,
                limit_value=self.min_portfolio_score,
                violation_amount=self.min_portfolio_score - portfolio_score,
                message=f"Portfolio ESG score {portfolio_score:.2f} below minimum {self.min_portfolio_score:.2f}"
            )
        
        # Check other constraints...
        # (Similar pattern for other constraint types)
        
        return None


@dataclass
class LeverageConstraint(Constraint):
    """Constraint for portfolio leverage limits."""
    
    max_leverage: float = 1.0
    allow_short: bool = False
    
    def __post_init__(self):
        if self.max_leverage <= 0:
            raise ValueError("Maximum leverage must be positive")
    
    def apply(self, weights: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Apply leverage constraints."""
        constraints = []
        
        # Leverage constraint: sum of absolute weights <= max_leverage
        def leverage_constraint(w):
            return self.max_leverage - np.sum(np.abs(w))
        
        constraints.append({'type': 'ineq', 'fun': leverage_constraint})
        
        # Short selling constraint
        if not self.allow_short:
            bounds = [(0, None) for _ in range(len(weights))]
            return {'constraints': constraints, 'bounds': bounds}
        
        return {'constraints': constraints}
    
    def validate(self, weights: np.ndarray, **kwargs) -> bool:
        """Validate leverage constraints."""
        # Check leverage
        leverage = np.sum(np.abs(weights))
        if leverage > self.max_leverage + 1e-6:  # Small tolerance for numerical errors
            return False
        
        # Check short selling
        if not self.allow_short and np.any(weights < -1e-6):
            return False
        
        return True


@dataclass
class ConcentrationConstraint(Constraint):
    """Constraint for concentration limits (e.g., top N holdings)."""
    
    top_n: int = 5  # Number of top holdings to consider
    max_concentration: float = 1.0  # Maximum weight of top N holdings
    
    def __post_init__(self):
        if not 0 < self.max_concentration <= 1:
            raise ValueError("Maximum concentration must be between 0 and 1")
        if self.top_n <= 0:
            raise ValueError("Top N must be positive")
    
    def apply(self, weights: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Apply concentration constraint."""
        def concentration_constraint(w):
            # Sort weights in descending order and sum top N
            top_weights = np.sort(w)[-self.top_n:]
            return self.max_concentration - np.sum(top_weights)
        
        return {
            'constraints': [{'type': 'ineq', 'fun': concentration_constraint}]
        }
    
    def validate(self, weights: np.ndarray, **kwargs) -> bool:
        """Validate concentration constraint."""
        top_weights = np.sort(weights)[-self.top_n:]
        concentration = np.sum(top_weights)
        return concentration <= self.max_concentration + 1e-6  # Small tolerance


class ConstraintManager:
    """
    Comprehensive constraint management system with dynamic adjustment capabilities.
    
    This class manages multiple constraints, handles constraint conflicts,
    and provides dynamic adjustment based on market conditions.
    """
    
    def __init__(self):
        """Initialize the constraint manager."""
        self.constraints: List[Constraint] = []
        self.market_condition: Optional[MarketCondition] = None
        self.constraint_history: List[Dict[str, Any]] = []
        
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the manager."""
        if not constraint.enabled:
            logger.info(f"Constraint {constraint.name} is disabled, not adding")
            return
            
        self.constraints.append(constraint)
        logger.info(f"Added constraint: {constraint.name} (Type: {constraint.constraint_type.value})")
    
    def remove_constraint(self, constraint_name: str) -> bool:
        """Remove a constraint by name."""
        for i, constraint in enumerate(self.constraints):
            if constraint.name == constraint_name:
                removed = self.constraints.pop(i)
                logger.info(f"Removed constraint: {removed.name}")
                return True
        return False
    
    def get_constraint(self, constraint_name: str) -> Optional[Constraint]:
        """Get a constraint by name."""
        for constraint in self.constraints:
            if constraint.name == constraint_name:
                return constraint
        return None
    
    def update_market_condition(self, market_condition: MarketCondition) -> None:
        """Update market conditions for dynamic constraint adjustment."""
        self.market_condition = market_condition
        logger.info(f"Updated market conditions: volatility={market_condition.volatility:.3f}, "
                   f"stress={market_condition.market_stress:.3f}, "
                   f"correlation_regime={market_condition.correlation_regime}")
    
    def get_active_constraints(self) -> List[Constraint]:
        """Get all active (enabled) constraints, adjusted for market conditions."""
        active_constraints = []
        
        for constraint in self.constraints:
            if not constraint.enabled:
                continue
                
            # Apply dynamic adjustment if market conditions are available
            if constraint.is_dynamic and self.market_condition:
                adjusted_constraint = constraint.adjust_for_market_conditions(self.market_condition)
                active_constraints.append(adjusted_constraint)
            else:
                active_constraints.append(constraint)
        
        # Sort by priority (higher priority first)
        active_constraints.sort(key=lambda c: c.priority, reverse=True)
        
        return active_constraints
    
    def apply_all_constraints(self, weights: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Apply all active constraints and return combined constraint parameters."""
        active_constraints = self.get_active_constraints()
        
        combined_constraints = []
        combined_bounds = None
        constraint_metadata = {}
        
        for constraint in active_constraints:
            try:
                constraint_params = constraint.apply(weights, **kwargs)
                
                # Combine constraints
                if 'constraints' in constraint_params:
                    combined_constraints.extend(constraint_params['constraints'])
                
                # Handle bounds (take most restrictive)
                if 'bounds' in constraint_params:
                    new_bounds = constraint_params['bounds']
                    if combined_bounds is None:
                        combined_bounds = new_bounds
                    else:
                        # Take intersection of bounds (most restrictive)
                        combined_bounds = [
                            (max(old[0], new[0]), min(old[1], new[1]))
                            for old, new in zip(combined_bounds, new_bounds)
                        ]
                
                # Store metadata
                constraint_metadata[constraint.name] = {
                    'type': constraint.constraint_type.value,
                    'severity': constraint.severity.value,
                    'priority': constraint.priority,
                    'is_dynamic': constraint.is_dynamic
                }
                
            except Exception as e:
                logger.error(f"Error applying constraint {constraint.name}: {str(e)}")
                if constraint.severity == ConstraintSeverity.CRITICAL:
                    raise
        
        result = {
            'constraints': combined_constraints,
            'constraint_metadata': constraint_metadata,
            'num_active_constraints': len(active_constraints)
        }
        
        if combined_bounds:
            result['bounds'] = combined_bounds
        
        return result
    
    def validate_all_constraints(self, weights: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Validate weights against all active constraints."""
        active_constraints = self.get_active_constraints()
        
        violations = []
        validation_results = {}
        
        for constraint in active_constraints:
            try:
                is_valid = constraint.validate(weights, **kwargs)
                validation_results[constraint.name] = is_valid
                
                if not is_valid:
                    violation = constraint.check_violation(weights, **kwargs)
                    if violation:
                        violations.append(violation)
                
            except Exception as e:
                logger.error(f"Error validating constraint {constraint.name}: {str(e)}")
                validation_results[constraint.name] = False
                
                violations.append(ConstraintViolation(
                    constraint_name=constraint.name,
                    violation_type="validation_error",
                    severity=constraint.severity,
                    current_value=0.0,
                    limit_value=0.0,
                    violation_amount=0.0,
                    message=f"Validation error: {str(e)}"
                ))
        
        # Sort violations by severity
        violations.sort(key=lambda v: ['soft', 'hard', 'critical'].index(v.severity.value), reverse=True)
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'validation_results': validation_results,
            'num_violations': len(violations),
            'critical_violations': [v for v in violations if v.severity == ConstraintSeverity.CRITICAL],
            'hard_violations': [v for v in violations if v.severity == ConstraintSeverity.HARD],
            'soft_violations': [v for v in violations if v.severity == ConstraintSeverity.SOFT]
        }
    
    def resolve_constraint_conflicts(self) -> List[str]:
        """
        Identify and suggest resolutions for constraint conflicts.
        
        Returns:
            List of conflict resolution suggestions
        """
        conflicts = []
        active_constraints = self.get_active_constraints()
        
        # Check for conflicting position limits
        position_constraints = [c for c in active_constraints 
                              if c.constraint_type == ConstraintType.POSITION_LIMIT]
        
        if len(position_constraints) > 1:
            conflicts.append("Multiple position limit constraints detected. Consider consolidating.")
        
        # Check for conflicting sector constraints
        sector_constraints = [c for c in active_constraints 
                            if c.constraint_type == ConstraintType.SECTOR_LIMIT]
        
        if len(sector_constraints) > 1:
            # Check for overlapping sector mappings
            all_sectors = set()
            for constraint in sector_constraints:
                if hasattr(constraint, 'sector_limits'):
                    all_sectors.update(constraint.sector_limits.keys())
            
            if len(all_sectors) > sum(len(c.sector_limits) for c in sector_constraints):
                conflicts.append("Overlapping sector constraints detected. Consider merging sector limits.")
        
        # Check for impossible constraint combinations
        turnover_constraints = [c for c in active_constraints 
                              if c.constraint_type == ConstraintType.TURNOVER_LIMIT]
        
        if turnover_constraints:
            min_turnover = min(c.max_turnover for c in turnover_constraints)
            if min_turnover < 0.01:  # Very low turnover
                position_constraints_exist = any(c.constraint_type == ConstraintType.POSITION_LIMIT 
                                               for c in active_constraints)
                if position_constraints_exist:
                    conflicts.append("Very low turnover limit may conflict with position limits. "
                                   "Consider relaxing turnover constraint.")
        
        return conflicts
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get a summary of all constraints and their status."""
        active_constraints = self.get_active_constraints()
        
        summary = {
            'total_constraints': len(self.constraints),
            'active_constraints': len(active_constraints),
            'disabled_constraints': len(self.constraints) - len(active_constraints),
            'dynamic_constraints': len([c for c in active_constraints if c.is_dynamic]),
            'constraint_types': {},
            'severity_distribution': {},
            'market_condition_applied': self.market_condition is not None
        }
        
        # Count by type
        for constraint in active_constraints:
            constraint_type = constraint.constraint_type.value
            summary['constraint_types'][constraint_type] = summary['constraint_types'].get(constraint_type, 0) + 1
        
        # Count by severity
        for constraint in active_constraints:
            severity = constraint.severity.value
            summary['severity_distribution'][severity] = summary['severity_distribution'].get(severity, 0) + 1
        
        return summary
    
    def create_constraint_report(self, weights: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Create a comprehensive constraint report."""
        validation_results = self.validate_all_constraints(weights, **kwargs)
        conflicts = self.resolve_constraint_conflicts()
        summary = self.get_constraint_summary()
        
        return {
            'timestamp': datetime.now(),
            'summary': summary,
            'validation': validation_results,
            'conflicts': conflicts,
            'market_condition': self.market_condition,
            'recommendations': self._generate_recommendations(validation_results, conflicts)
        }
    
    def _generate_recommendations(self, validation_results: Dict[str, Any], 
                                 conflicts: List[str]) -> List[str]:
        """Generate recommendations based on validation results and conflicts."""
        recommendations = []
        
        if validation_results['critical_violations']:
            recommendations.append("CRITICAL: Address critical constraint violations immediately.")
        
        if validation_results['hard_violations']:
            recommendations.append("Address hard constraint violations before proceeding.")
        
        if validation_results['soft_violations']:
            recommendations.append("Consider addressing soft constraint violations to improve portfolio quality.")
        
        if conflicts:
            recommendations.append("Resolve constraint conflicts to avoid optimization issues.")
        
        if self.market_condition and self.market_condition.market_stress > 0.7:
            recommendations.append("High market stress detected. Consider relaxing dynamic constraints.")
        
        if not recommendations:
            recommendations.append("All constraints are satisfied. Portfolio is ready for optimization.")
        
        return recommendations