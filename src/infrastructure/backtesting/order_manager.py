"""
Order Management System for backtesting.

Handles different order types (market, limit, stop) with realistic execution logic.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from ...domain.exceptions import BacktestError, ValidationError
from ...domain.value_objects import Price


class OrderType(Enum):
    """Types of orders supported by the system."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderSide(Enum):
    """Order side (buy or sell)."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Represents a trading order."""
    
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None  # For limit orders
    stop_price: Optional[Decimal] = None  # For stop orders
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = field(default=Decimal('0'))
    filled_price: Optional[Decimal] = None
    commission: Decimal = field(default=Decimal('0'))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate order parameters."""
        if self.quantity <= 0:
            raise ValidationError("Order quantity must be positive")
        
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValidationError(f"{self.order_type.value} order requires a price")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValidationError(f"{self.order_type.value} order requires a stop price")
        
        if self.price is not None and self.price <= 0:
            raise ValidationError("Order price must be positive")
        
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValidationError("Stop price must be positive")
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining quantity to be filled."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active (can be executed)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
    
    def fill(self, quantity: Decimal, price: Decimal, commission: Decimal = Decimal('0')) -> None:
        """Fill the order (partially or completely)."""
        if quantity <= 0:
            raise ValidationError("Fill quantity must be positive")
        
        if quantity > self.remaining_quantity:
            raise ValidationError("Fill quantity exceeds remaining quantity")
        
        self.filled_quantity += quantity
        self.filled_price = price
        self.commission += commission
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self) -> None:
        """Cancel the order."""
        if not self.is_active:
            raise ValidationError("Cannot cancel non-active order")
        self.status = OrderStatus.CANCELLED
    
    def reject(self, reason: str = "") -> None:
        """Reject the order."""
        self.status = OrderStatus.REJECTED
        self.metadata['rejection_reason'] = reason


class OrderManager:
    """Manages order execution during backtesting."""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self._order_counter = 0
    
    def submit_order(self, order: Order) -> str:
        """Submit an order for execution."""
        if not order.id:
            order.id = self._generate_order_id()
        
        self.orders[order.id] = order
        return order.id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.is_active:
            order.cancel()
            self._move_to_history(order_id)
            return True
        return False
    
    def process_market_data(self, market_data: pd.Series, timestamp: datetime) -> List[Order]:
        """Process market data and execute eligible orders."""
        executed_orders = []
        
        for order_id, order in list(self.orders.items()):
            if not order.is_active:
                continue
            
            if order.symbol not in market_data.index:
                continue
            
            current_price = Decimal(str(market_data[order.symbol]))
            
            if self._should_execute_order(order, current_price):
                execution_price = self._calculate_execution_price(order, current_price)
                
                # Calculate fill quantity (could be partial for large orders)
                fill_quantity = self._calculate_fill_quantity(order, current_price, market_data)
                
                try:
                    if fill_quantity > 0:
                        order.fill(fill_quantity, execution_price)
                        executed_orders.append(order)
                        
                        if order.is_filled:
                            self._move_to_history(order_id)
                
                except ValidationError as e:
                    order.reject(str(e))
                    self._move_to_history(order_id)
        
        return executed_orders
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol."""
        active_orders = [order for order in self.orders.values() if order.is_active]
        
        if symbol:
            active_orders = [order for order in active_orders if order.symbol == symbol]
        
        return active_orders
    
    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get order history, optionally filtered by symbol."""
        if symbol:
            return [order for order in self.order_history if order.symbol == symbol]
        return self.order_history.copy()
    
    def get_filled_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all filled orders, optionally filtered by symbol."""
        filled_orders = [order for order in self.order_history if order.is_filled]
        
        if symbol:
            filled_orders = [order for order in filled_orders if order.symbol == symbol]
        
        return filled_orders
    
    def clear_orders(self) -> None:
        """Clear all orders and history."""
        self.orders.clear()
        self.order_history.clear()
        self._order_counter = 0
    
    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        self._order_counter += 1
        return f"ORDER_{self._order_counter:06d}"
    
    def _should_execute_order(self, order: Order, current_price: Decimal) -> bool:
        """Determine if an order should be executed based on current market price."""
        if order.order_type == OrderType.MARKET:
            return True
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                return current_price <= order.price
            else:  # SELL
                return current_price >= order.price
        
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                return current_price >= order.stop_price
            else:  # SELL
                return current_price <= order.stop_price
        
        elif order.order_type == OrderType.STOP_LIMIT:
            # First check if stop is triggered
            stop_triggered = False
            if order.side == OrderSide.BUY:
                stop_triggered = current_price >= order.stop_price
            else:  # SELL
                stop_triggered = current_price <= order.stop_price
            
            if not stop_triggered:
                return False
            
            # Convert to limit order once stop is triggered
            order.metadata['stop_triggered'] = True
            
            # Then check limit condition
            if order.side == OrderSide.BUY:
                return current_price <= order.price
            else:  # SELL
                return current_price >= order.price
        
        return False
    
    def _calculate_execution_price(self, order: Order, current_price: Decimal) -> Decimal:
        """Calculate the actual execution price for an order."""
        if order.order_type == OrderType.MARKET:
            return current_price
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders get filled at the limit price or better
            if order.side == OrderSide.BUY:
                return min(current_price, order.price)
            else:  # SELL
                return max(current_price, order.price)
        
        elif order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            # Stop orders become market orders when triggered
            # Stop-limit orders use the limit price if available
            if order.order_type == OrderType.STOP_LIMIT:
                if order.side == OrderSide.BUY:
                    return min(current_price, order.price)
                else:  # SELL
                    return max(current_price, order.price)
            else:
                return current_price
        
        return current_price
    
    def _calculate_fill_quantity(
        self, 
        order: Order, 
        current_price: Decimal, 
        market_data: pd.Series
    ) -> Decimal:
        """Calculate the quantity that can be filled for an order."""
        # For backtesting, we typically assume full fills
        # In reality, this would consider market depth, liquidity, etc.
        
        remaining_qty = order.remaining_quantity
        
        # For very large orders, we might simulate partial fills
        # This is a simplified model - real implementation would be more sophisticated
        if remaining_qty > Decimal('10000'):  # Large order threshold
            # Fill only a portion of large orders to simulate market impact
            max_fill = remaining_qty * Decimal('0.5')  # Fill up to 50% at once
            return min(remaining_qty, max_fill)
        
        return remaining_qty
    
    def _move_to_history(self, order_id: str) -> None:
        """Move an order from active orders to history."""
        if order_id in self.orders:
            order = self.orders.pop(order_id)
            self.order_history.append(order)