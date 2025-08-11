"""
Domain value objects for the quantitative framework.

Value objects are immutable objects that represent descriptive aspects of the domain
with no conceptual identity. They are defined by their attributes rather than identity.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any
from enum import Enum

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


@dataclass(frozen=True)
class Price:
    """Immutable value object representing a price at a specific time."""
    
    value: Decimal
    currency: str
    timestamp: datetime
    
    def __post_init__(self):
        """Validate price data after initialization."""
        if self.value < 0:
            raise ValidationError("Price cannot be negative")
        if not self.currency:
            raise ValidationError("Currency cannot be empty")
        if len(self.currency) != 3:
            raise ValidationError("Currency must be a 3-letter code (e.g., USD, EUR)")
    
    def to_currency(self, target_currency: str, exchange_rate: Decimal) -> 'Price':
        """Convert price to another currency."""
        if not target_currency or len(target_currency) != 3:
            raise ValidationError("Target currency must be a 3-letter code")
        if exchange_rate <= 0:
            raise ValidationError("Exchange rate must be positive")
        
        converted_value = self.value * exchange_rate
        return Price(
            value=converted_value,
            currency=target_currency,
            timestamp=self.timestamp
        )
    
    def __str__(self) -> str:
        return f"{self.value} {self.currency}"


@dataclass(frozen=True)
class Return:
    """Immutable value object representing a return over a specific period."""
    
    value: float
    period: str
    type: ReturnType
    timestamp: datetime
    
    def __post_init__(self):
        """Validate return data after initialization."""
        if not self.period:
            raise ValidationError("Period cannot be empty")
        if self.period not in ['1d', '1w', '1m', '3m', '6m', '1y', 'ytd', 'custom']:
            raise ValidationError("Invalid period format")
        if self.type not in ReturnType:
            raise ValidationError("Invalid return type")
    
    def annualize(self, periods_per_year: int = 252) -> 'Return':
        """Annualize the return based on the period."""
        if self.period == '1y':
            return self  # Already annualized
        
        # Convert period to number of periods per year
        period_multipliers = {
            '1d': 252,  # Trading days per year
            '1w': 52,   # Weeks per year
            '1m': 12,   # Months per year
            '3m': 4,    # Quarters per year
            '6m': 2,    # Half-years per year
        }
        
        multiplier = period_multipliers.get(self.period, periods_per_year)
        
        if self.type == ReturnType.SIMPLE:
            annualized_return = (1 + self.value) ** multiplier - 1
        else:  # LOG return
            annualized_return = self.value * multiplier
        
        return Return(
            value=annualized_return,
            period='1y',
            type=self.type,
            timestamp=self.timestamp
        )
    
    def to_simple(self) -> 'Return':
        """Convert log return to simple return."""
        if self.type == ReturnType.SIMPLE:
            return self
        
        import math
        simple_return = math.exp(self.value) - 1
        return Return(
            value=simple_return,
            period=self.period,
            type=ReturnType.SIMPLE,
            timestamp=self.timestamp
        )
    
    def to_log(self) -> 'Return':
        """Convert simple return to log return."""
        if self.type == ReturnType.LOG:
            return self
        
        import math
        log_return = math.log(1 + self.value)
        return Return(
            value=log_return,
            period=self.period,
            type=ReturnType.LOG,
            timestamp=self.timestamp
        )
    
    def __str__(self) -> str:
        return f"{self.value:.4f} ({self.type.value}, {self.period})"


@dataclass(frozen=True)
class Signal:
    """Immutable value object representing a trading signal."""
    
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate signal data after initialization."""
        if not self.symbol:
            raise ValidationError("Signal symbol cannot be empty")
        if not isinstance(self.signal_type, SignalType):
            raise ValidationError("Invalid signal type")
        if not -1.0 <= self.strength <= 1.0:
            raise ValidationError("Signal strength must be between -1.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("Signal confidence must be between 0.0 and 1.0")
        if not self.source:
            raise ValidationError("Signal source cannot be empty")
    
    def is_buy_signal(self) -> bool:
        """Check if this is a buy signal."""
        return self.signal_type == SignalType.BUY and self.strength > 0
    
    def is_sell_signal(self) -> bool:
        """Check if this is a sell signal."""
        return self.signal_type == SignalType.SELL and self.strength < 0
    
    def is_hold_signal(self) -> bool:
        """Check if this is a hold signal."""
        return self.signal_type == SignalType.HOLD or self.strength == 0
    
    def get_weighted_strength(self) -> float:
        """Get signal strength weighted by confidence."""
        return self.strength * self.confidence
    
    def __str__(self) -> str:
        return f"{self.symbol}: {self.signal_type.value} (strength: {self.strength:.2f}, confidence: {self.confidence:.2f})"


@dataclass(frozen=True)
class RiskMetrics:
    """Immutable value object representing risk metrics for a portfolio or strategy."""
    
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    cvar_95: float  # Conditional VaR at 95% confidence
    volatility: float  # Annualized volatility
    max_drawdown: float  # Maximum drawdown
    beta: float  # Beta relative to benchmark
    tracking_error: float  # Tracking error vs benchmark
    timestamp: datetime
    
    def __post_init__(self):
        """Validate risk metrics after initialization."""
        if self.var_95 > 0:
            raise ValidationError("VaR should be negative (representing loss)")
        if self.var_99 > 0:
            raise ValidationError("VaR should be negative (representing loss)")
        if self.cvar_95 > 0:
            raise ValidationError("CVaR should be negative (representing loss)")
        if self.volatility < 0:
            raise ValidationError("Volatility cannot be negative")
        if self.max_drawdown > 0:
            raise ValidationError("Max drawdown should be negative or zero")
        if self.tracking_error < 0:
            raise ValidationError("Tracking error cannot be negative")
    
    def __str__(self) -> str:
        return f"VaR95: {self.var_95:.4f}, Vol: {self.volatility:.4f}, MaxDD: {self.max_drawdown:.4f}"


@dataclass(frozen=True)
class PerformanceMetrics:
    """Immutable value object representing performance metrics."""
    
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    timestamp: datetime
    
    def __post_init__(self):
        """Validate performance metrics after initialization."""
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValidationError("Win rate must be between 0.0 and 1.0")
        if self.profit_factor < 0:
            raise ValidationError("Profit factor cannot be negative")
        if self.volatility < 0:
            raise ValidationError("Volatility cannot be negative")
        if self.max_drawdown > 0:
            raise ValidationError("Max drawdown should be negative or zero")
    
    def __str__(self) -> str:
        return f"Return: {self.annualized_return:.2%}, Sharpe: {self.sharpe_ratio:.2f}, MaxDD: {self.max_drawdown:.2%}"