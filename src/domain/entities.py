"""
Domain entities for the quantitative framework.

Entities are objects with identity that encapsulate business logic and maintain state.
They represent the core business concepts in the quantitative finance domain.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, Optional, List
from enum import Enum
import pandas as pd

from .value_objects import Price, Return, Signal
from .exceptions import ValidationError


class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ReturnType(Enum):
    """Types of return calculations."""
    SIMPLE = "SIMPLE"
    LOG = "LOG"


@dataclass
class Position:
    """Represents a position in a financial instrument."""
    
    symbol: str
    quantity: Decimal
    average_cost: Decimal
    current_price: Decimal
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate position data after initialization."""
        if not self.symbol:
            raise ValidationError("Position symbol cannot be empty")
        if self.quantity < 0:
            raise ValidationError("Position quantity cannot be negative")
        if self.average_cost < 0:
            raise ValidationError("Average cost cannot be negative")
        if self.current_price < 0:
            raise ValidationError("Current price cannot be negative")
    
    @property
    def market_value(self) -> Decimal:
        """Calculate current market value of the position."""
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized profit/loss."""
        return (self.current_price - self.average_cost) * self.quantity
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.average_cost == 0:
            return 0.0
        return float((self.current_price - self.average_cost) / self.average_cost)
    
    def update_price(self, new_price: Decimal) -> None:
        """Update the current price of the position."""
        if new_price < 0:
            raise ValidationError("Price cannot be negative")
        self.current_price = new_price
        self.last_updated = datetime.now()


@dataclass
class Portfolio:
    """Represents a portfolio containing multiple positions."""
    
    id: str
    name: str
    positions: Dict[str, Position] = field(default_factory=dict)
    cash: Decimal = field(default=Decimal('0'))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate portfolio data after initialization."""
        if not self.id:
            raise ValidationError("Portfolio ID cannot be empty")
        if not self.name:
            raise ValidationError("Portfolio name cannot be empty")
        if self.cash < 0:
            raise ValidationError("Cash cannot be negative")
    
    @property
    def total_value(self) -> Decimal:
        """Calculate total portfolio value including cash."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return positions_value + self.cash
    
    @property
    def positions_value(self) -> Decimal:
        """Calculate total value of all positions excluding cash."""
        return sum(pos.market_value for pos in self.positions.values())
    
    def get_weights(self) -> Dict[str, float]:
        """Get position weights as percentage of total portfolio value."""
        total_val = self.total_value
        if total_val == 0:
            return {}
        
        weights = {}
        for symbol, position in self.positions.items():
            weights[symbol] = float(position.market_value / total_val)
        
        # Add cash weight
        if self.cash > 0:
            weights['CASH'] = float(self.cash / total_val)
        
        return weights
    
    def add_position(self, position: Position) -> None:
        """Add a new position to the portfolio."""
        if position.symbol in self.positions:
            raise ValidationError(f"Position for {position.symbol} already exists")
        
        self.positions[position.symbol] = position
        self.updated_at = datetime.now()
    
    def update_position(self, symbol: str, quantity: Decimal, price: Decimal) -> None:
        """Update an existing position or create a new one."""
        if symbol in self.positions:
            # Update existing position
            existing_pos = self.positions[symbol]
            total_cost = (existing_pos.quantity * existing_pos.average_cost + 
                         quantity * price)
            total_quantity = existing_pos.quantity + quantity
            
            if total_quantity == 0:
                # Remove position if quantity becomes zero
                del self.positions[symbol]
            else:
                existing_pos.quantity = total_quantity
                existing_pos.average_cost = total_cost / total_quantity
                existing_pos.current_price = price
                existing_pos.last_updated = datetime.now()
        else:
            # Create new position
            new_position = Position(
                symbol=symbol,
                quantity=quantity,
                average_cost=price,
                current_price=price
            )
            self.positions[symbol] = new_position
        
        self.updated_at = datetime.now()
    
    def remove_position(self, symbol: str) -> None:
        """Remove a position from the portfolio."""
        if symbol not in self.positions:
            raise ValidationError(f"Position for {symbol} does not exist")
        
        del self.positions[symbol]
        self.updated_at = datetime.now()
    
    def calculate_returns(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """Calculate portfolio returns over a time period."""
        # This is a placeholder - actual implementation would require historical data
        # Will be implemented in later tasks when data management is available
        raise NotImplementedError("Returns calculation requires historical data access")


@dataclass
class Strategy:
    """Represents a trading strategy with its configuration and metadata."""
    
    id: str
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    performance_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate strategy data after initialization."""
        if not self.id:
            raise ValidationError("Strategy ID cannot be empty")
        if not self.name:
            raise ValidationError("Strategy name cannot be empty")
        if not self.description:
            raise ValidationError("Strategy description cannot be empty")
    
    def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        self.parameters.update(new_parameters)
        self.updated_at = datetime.now()
    
    def activate(self) -> None:
        """Activate the strategy."""
        self.is_active = True
        self.updated_at = datetime.now()
    
    def deactivate(self) -> None:
        """Deactivate the strategy."""
        self.is_active = False
        self.updated_at = datetime.now()
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Update strategy performance metrics."""
        self.performance_metrics = metrics.copy()
        self.updated_at = datetime.now()