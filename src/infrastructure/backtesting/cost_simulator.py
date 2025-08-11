"""
Trading Cost Simulator for realistic backtesting.

Simulates various trading costs including:
- Commission costs
- Bid-ask spread costs
- Market impact costs
- Slippage costs
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from ...domain.exceptions import ValidationError
from .order_manager import Order, OrderSide


@dataclass
class TradingCosts:
    """Container for different types of trading costs."""
    
    commission: Decimal = Decimal('0')
    spread_cost: Decimal = Decimal('0')
    market_impact: Decimal = Decimal('0')
    slippage: Decimal = Decimal('0')
    
    @property
    def total_cost(self) -> Decimal:
        """Calculate total trading cost."""
        return self.commission + self.spread_cost + self.market_impact + self.slippage


@dataclass
class CostModel:
    """Configuration for trading cost calculations."""
    
    # Commission structure
    commission_per_share: Decimal = Decimal('0.005')  # $0.005 per share
    commission_percentage: Decimal = Decimal('0.001')  # 0.1% of trade value
    minimum_commission: Decimal = Decimal('1.0')  # Minimum $1 commission
    maximum_commission: Decimal = Decimal('50.0')  # Maximum $50 commission
    
    # Spread costs
    typical_spread_bps: int = 5  # 5 basis points typical spread
    spread_multiplier: float = 0.5  # Pay half the spread on average
    
    # Market impact model parameters
    market_impact_coefficient: float = 0.1  # Impact coefficient
    market_impact_exponent: float = 0.6  # Impact exponent (square root-ish)
    
    # Slippage parameters
    base_slippage_bps: int = 2  # Base slippage in basis points
    volatility_multiplier: float = 1.0  # Multiplier for volatility-based slippage
    
    # Liquidity parameters
    average_daily_volume: Dict[str, float] = None  # ADV by symbol
    
    def __post_init__(self):
        if self.average_daily_volume is None:
            self.average_daily_volume = {}


class TradingCostSimulator:
    """Simulates realistic trading costs for backtesting."""
    
    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()
        self._volatility_cache: Dict[str, float] = {}
    
    def calculate_trading_costs(
        self,
        order: Order,
        execution_price: Decimal,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> TradingCosts:
        """Calculate comprehensive trading costs for an order execution."""
        
        trade_value = order.filled_quantity * execution_price
        
        # Calculate commission
        commission = self._calculate_commission(order, trade_value)
        
        # Calculate spread cost
        spread_cost = self._calculate_spread_cost(order, execution_price, market_data, timestamp)
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(order, execution_price, market_data, timestamp)
        
        # Calculate slippage
        slippage = self._calculate_slippage(order, execution_price, market_data, timestamp)
        
        return TradingCosts(
            commission=commission,
            spread_cost=spread_cost,
            market_impact=market_impact,
            slippage=slippage
        )
    
    def _calculate_commission(self, order: Order, trade_value: Decimal) -> Decimal:
        """Calculate commission costs."""
        # Per-share commission
        per_share_commission = order.filled_quantity * self.cost_model.commission_per_share
        
        # Percentage-based commission
        percentage_commission = trade_value * self.cost_model.commission_percentage
        
        # Use the higher of the two
        commission = max(per_share_commission, percentage_commission)
        
        # Apply minimum and maximum limits
        commission = max(commission, self.cost_model.minimum_commission)
        commission = min(commission, self.cost_model.maximum_commission)
        
        return commission
    
    def _calculate_spread_cost(
        self,
        order: Order,
        execution_price: Decimal,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Decimal:
        """Calculate bid-ask spread costs."""
        
        # Get typical spread for the symbol
        spread_bps = self.cost_model.typical_spread_bps
        
        # Adjust spread based on volatility if available
        if order.symbol in market_data.columns:
            volatility = self._get_volatility(order.symbol, market_data, timestamp)
            # Higher volatility typically means wider spreads
            spread_bps = int(spread_bps * (1 + volatility))
        
        # Calculate spread cost (typically pay half the spread)
        spread_cost = (
            execution_price * 
            order.filled_quantity * 
            Decimal(str(spread_bps / 10000)) *  # Convert bps to decimal
            Decimal(str(self.cost_model.spread_multiplier))
        )
        
        return spread_cost
    
    def _calculate_market_impact(
        self,
        order: Order,
        execution_price: Decimal,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Decimal:
        """Calculate market impact costs using square-root model."""
        
        # Get average daily volume for the symbol
        adv = self.cost_model.average_daily_volume.get(order.symbol, 1000000)  # Default 1M shares
        
        # Calculate participation rate (order size / ADV)
        participation_rate = float(order.filled_quantity) / adv
        
        # Market impact using square-root model: impact = coefficient * (participation_rate ^ exponent)
        impact_factor = (
            self.cost_model.market_impact_coefficient * 
            (participation_rate ** self.cost_model.market_impact_exponent)
        )
        
        # Adjust for volatility - higher volatility means higher impact
        volatility = self._get_volatility(order.symbol, market_data, timestamp)
        volatility_adjustment = 1 + (volatility * 0.5)  # Scale volatility impact
        impact_factor *= volatility_adjustment
        
        # Adjust for order side - sell orders typically have higher impact
        if order.side == OrderSide.SELL:
            impact_factor *= 1.1  # 10% higher impact for sells
        
        # Convert to basis points and apply to trade value
        impact_bps = impact_factor * 100  # Convert to basis points
        market_impact = execution_price * order.filled_quantity * Decimal(str(impact_bps / 10000))
        
        # Market impact is always a cost (positive)
        return abs(market_impact)
    
    def _calculate_slippage(
        self,
        order: Order,
        execution_price: Decimal,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> Decimal:
        """Calculate slippage costs."""
        
        base_slippage = self.cost_model.base_slippage_bps
        
        # Adjust slippage based on volatility
        if order.symbol in market_data.columns:
            volatility = self._get_volatility(order.symbol, market_data, timestamp)
            volatility_adjustment = volatility * self.cost_model.volatility_multiplier
            total_slippage_bps = base_slippage * (1 + volatility_adjustment)
        else:
            total_slippage_bps = base_slippage
        
        # Calculate slippage cost
        slippage = (
            execution_price * 
            order.filled_quantity * 
            Decimal(str(total_slippage_bps / 10000))
        )
        
        return slippage
    
    def _get_volatility(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        timestamp: pd.Timestamp,
        lookback_days: int = 20
    ) -> float:
        """Calculate rolling volatility for a symbol."""
        
        cache_key = f"{symbol}_{timestamp}"
        if cache_key in self._volatility_cache:
            return self._volatility_cache[cache_key]
        
        try:
            # Get price data for the symbol
            if symbol not in market_data.columns:
                return 0.2  # Default 20% annualized volatility
            
            # Get recent price data
            symbol_data = market_data[symbol].dropna()
            
            # Find the position of current timestamp
            if timestamp not in symbol_data.index:
                # Find the closest timestamp
                closest_idx = symbol_data.index.get_indexer([timestamp], method='nearest')[0]
                if closest_idx == -1:
                    return 0.2
            else:
                closest_idx = symbol_data.index.get_loc(timestamp)
            
            # Get lookback period data
            start_idx = max(0, closest_idx - lookback_days)
            end_idx = closest_idx + 1
            
            recent_prices = symbol_data.iloc[start_idx:end_idx]
            
            if len(recent_prices) < 2:
                return 0.2
            
            # Calculate returns and volatility
            returns = recent_prices.pct_change().dropna()
            
            if len(returns) == 0:
                return 0.2
            
            # Annualized volatility (assuming 252 trading days)
            volatility = float(returns.std() * np.sqrt(252))
            
            # Cache the result
            self._volatility_cache[cache_key] = volatility
            
            return volatility
            
        except Exception:
            # Return default volatility if calculation fails
            return 0.2
    
    def update_cost_model(self, **kwargs) -> None:
        """Update cost model parameters."""
        for key, value in kwargs.items():
            if hasattr(self.cost_model, key):
                setattr(self.cost_model, key, value)
            else:
                raise ValidationError(f"Unknown cost model parameter: {key}")
    
    def set_average_daily_volumes(self, adv_data: Dict[str, float]) -> None:
        """Set average daily volume data for symbols."""
        self.cost_model.average_daily_volume.update(adv_data)
    
    def clear_cache(self) -> None:
        """Clear the volatility cache."""
        self._volatility_cache.clear()