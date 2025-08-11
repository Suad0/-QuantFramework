"""
Corporate Action Handler for backtesting.

Handles various corporate actions that affect backtesting accuracy:
- Stock splits and stock dividends
- Cash dividends
- Spin-offs
- Mergers and acquisitions
- Rights offerings
"""

from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd

from ...domain.exceptions import ValidationError, BacktestError
from ...domain.entities import Position, Portfolio


class CorporateActionType(Enum):
    """Types of corporate actions."""
    DIVIDEND = "DIVIDEND"
    STOCK_SPLIT = "STOCK_SPLIT"
    STOCK_DIVIDEND = "STOCK_DIVIDEND"
    SPIN_OFF = "SPIN_OFF"
    MERGER = "MERGER"
    RIGHTS_OFFERING = "RIGHTS_OFFERING"
    SPECIAL_DIVIDEND = "SPECIAL_DIVIDEND"


@dataclass
class CorporateAction:
    """Represents a corporate action event."""
    
    symbol: str
    action_type: CorporateActionType
    ex_date: date  # Ex-dividend/ex-split date
    record_date: Optional[date] = None
    payment_date: Optional[date] = None
    
    # Action-specific parameters
    dividend_amount: Optional[Decimal] = None  # Per share dividend
    split_ratio: Optional[float] = None  # New shares / old shares
    spin_off_ratio: Optional[float] = None  # Spin-off shares per original share
    spin_off_symbol: Optional[str] = None  # Symbol of spun-off company
    merger_ratio: Optional[float] = None  # Exchange ratio for merger
    merger_symbol: Optional[str] = None  # Symbol of acquiring company
    
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Validate required parameters based on action type
        if self.action_type == CorporateActionType.DIVIDEND:
            if self.dividend_amount is None or self.dividend_amount <= 0:
                raise ValidationError("Dividend actions require positive dividend amount")
        
        elif self.action_type == CorporateActionType.STOCK_SPLIT:
            if self.split_ratio is None or self.split_ratio <= 0:
                raise ValidationError("Stock split actions require positive split ratio")
        
        elif self.action_type == CorporateActionType.SPIN_OFF:
            if self.spin_off_ratio is None or self.spin_off_ratio <= 0:
                raise ValidationError("Spin-off actions require positive spin-off ratio")
            if not self.spin_off_symbol:
                raise ValidationError("Spin-off actions require spin-off symbol")
        
        elif self.action_type == CorporateActionType.MERGER:
            if self.merger_ratio is None or self.merger_ratio <= 0:
                raise ValidationError("Merger actions require positive merger ratio")
            if not self.merger_symbol:
                raise ValidationError("Merger actions require merger symbol")


class CorporateActionHandler:
    """Handles corporate actions during backtesting."""
    
    def __init__(self):
        self.actions: Dict[str, List[CorporateAction]] = {}  # symbol -> list of actions
        self.processed_actions: List[CorporateAction] = []
    
    def add_corporate_action(self, action: CorporateAction) -> None:
        """Add a corporate action to the handler."""
        if action.symbol not in self.actions:
            self.actions[action.symbol] = []
        
        self.actions[action.symbol].append(action)
        
        # Sort actions by ex-date
        self.actions[action.symbol].sort(key=lambda x: x.ex_date)
    
    def load_corporate_actions_from_data(self, actions_data: pd.DataFrame) -> None:
        """Load corporate actions from a DataFrame."""
        required_columns = ['symbol', 'action_type', 'ex_date']
        
        if not all(col in actions_data.columns for col in required_columns):
            raise ValidationError(f"Corporate actions data must contain columns: {required_columns}")
        
        for _, row in actions_data.iterrows():
            try:
                action = CorporateAction(
                    symbol=row['symbol'],
                    action_type=CorporateActionType(row['action_type']),
                    ex_date=pd.to_datetime(row['ex_date']).date(),
                    record_date=pd.to_datetime(row.get('record_date')).date() if pd.notna(row.get('record_date')) else None,
                    payment_date=pd.to_datetime(row.get('payment_date')).date() if pd.notna(row.get('payment_date')) else None,
                    dividend_amount=Decimal(str(row['dividend_amount'])) if pd.notna(row.get('dividend_amount')) else None,
                    split_ratio=float(row['split_ratio']) if pd.notna(row.get('split_ratio')) else None,
                    spin_off_ratio=float(row['spin_off_ratio']) if pd.notna(row.get('spin_off_ratio')) else None,
                    spin_off_symbol=row.get('spin_off_symbol') if pd.notna(row.get('spin_off_symbol')) else None,
                    merger_ratio=float(row['merger_ratio']) if pd.notna(row.get('merger_ratio')) else None,
                    merger_symbol=row.get('merger_symbol') if pd.notna(row.get('merger_symbol')) else None
                )
                self.add_corporate_action(action)
            except Exception as e:
                raise BacktestError(f"Error loading corporate action for {row['symbol']}: {str(e)}")
    
    def process_corporate_actions(
        self,
        portfolio: Portfolio,
        current_date: date,
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Process corporate actions that occur on the current date."""
        
        results = {
            'cash_flows': Decimal('0'),
            'position_changes': {},
            'new_positions': {},
            'actions_processed': []
        }
        
        for symbol in list(portfolio.positions.keys()):
            if symbol not in self.actions:
                continue
            
            position = portfolio.positions[symbol]
            
            for action in self.actions[symbol]:
                if action.ex_date == current_date and action not in self.processed_actions:
                    
                    if action.action_type == CorporateActionType.DIVIDEND:
                        cash_flow = self._process_dividend(position, action)
                        results['cash_flows'] += cash_flow
                    
                    elif action.action_type == CorporateActionType.STOCK_SPLIT:
                        self._process_stock_split(position, action)
                        results['position_changes'][symbol] = {
                            'old_quantity': position.quantity / Decimal(str(action.split_ratio)),
                            'new_quantity': position.quantity,
                            'split_ratio': action.split_ratio
                        }
                    
                    elif action.action_type == CorporateActionType.SPIN_OFF:
                        new_position = self._process_spin_off(position, action, market_data)
                        if new_position:
                            results['new_positions'][action.spin_off_symbol] = new_position
                    
                    elif action.action_type == CorporateActionType.MERGER:
                        self._process_merger(portfolio, position, action)
                        results['position_changes'][symbol] = 'merged'
                    
                    self.processed_actions.append(action)
                    results['actions_processed'].append(action)
        
        return results
    
    def _process_dividend(self, position: Position, action: CorporateAction) -> Decimal:
        """Process a dividend payment."""
        dividend_payment = position.quantity * action.dividend_amount
        
        # Record dividend in metadata for tracking
        if 'dividends_received' not in action.metadata:
            action.metadata['dividends_received'] = []
        
        action.metadata['dividends_received'].append({
            'symbol': position.symbol,
            'quantity': float(position.quantity),
            'dividend_per_share': float(action.dividend_amount),
            'total_dividend': float(dividend_payment),
            'ex_date': action.ex_date,
            'payment_date': action.payment_date
        })
        
        return dividend_payment
    
    def _process_stock_split(self, position: Position, action: CorporateAction) -> None:
        """Process a stock split."""
        # Adjust quantity and average cost
        old_quantity = position.quantity
        old_avg_cost = position.average_cost
        
        position.quantity = old_quantity * Decimal(str(action.split_ratio))
        position.average_cost = old_avg_cost / Decimal(str(action.split_ratio))
        position.last_updated = datetime.now()
    
    def _process_spin_off(
        self,
        position: Position,
        action: CorporateAction,
        market_data: Optional[pd.DataFrame] = None
    ) -> Optional[Position]:
        """Process a spin-off."""
        
        # Calculate spin-off shares
        spin_off_shares = position.quantity * Decimal(str(action.spin_off_ratio))
        
        # Estimate spin-off share price (simplified approach)
        # In reality, this would require more sophisticated valuation
        spin_off_price = position.current_price * Decimal('0.1')  # Assume 10% of parent value
        
        if market_data is not None and action.spin_off_symbol in market_data.columns:
            # Use actual market price if available
            try:
                spin_off_price = Decimal(str(market_data[action.spin_off_symbol].iloc[-1]))
            except (KeyError, IndexError):
                pass
        
        # Create new position for spin-off
        spin_off_position = Position(
            symbol=action.spin_off_symbol,
            quantity=spin_off_shares,
            average_cost=spin_off_price,
            current_price=spin_off_price
        )
        
        # Adjust original position value (simplified)
        # In practice, this would require more sophisticated adjustment
        adjustment_factor = Decimal('0.9')  # Assume 10% value goes to spin-off
        position.current_price *= adjustment_factor
        position.last_updated = datetime.now()
        
        return spin_off_position
    
    def _process_merger(self, portfolio: Portfolio, position: Position, action: CorporateAction) -> None:
        """Process a merger."""
        
        # Calculate new shares in acquiring company
        new_shares = position.quantity * Decimal(str(action.merger_ratio))
        
        # Remove old position
        old_symbol = position.symbol
        portfolio.remove_position(old_symbol)
        
        # Add or update position in acquiring company
        if action.merger_symbol in portfolio.positions:
            # Update existing position
            existing_position = portfolio.positions[action.merger_symbol]
            total_cost = (existing_position.quantity * existing_position.average_cost + 
                         new_shares * position.average_cost)
            total_quantity = existing_position.quantity + new_shares
            
            existing_position.quantity = total_quantity
            existing_position.average_cost = total_cost / total_quantity
            existing_position.last_updated = datetime.now()
        else:
            # Create new position
            new_position = Position(
                symbol=action.merger_symbol,
                quantity=new_shares,
                average_cost=position.average_cost,
                current_price=position.current_price
            )
            portfolio.add_position(new_position)
    
    def adjust_historical_prices(
        self,
        price_data: pd.DataFrame,
        symbol: str,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """Adjust historical prices for corporate actions (split/dividend adjusted)."""
        
        if symbol not in self.actions:
            return price_data
        
        adjusted_data = price_data.copy()
        
        # Get actions for this symbol, sorted by date (most recent first)
        symbol_actions = sorted(
            [action for action in self.actions[symbol] if end_date is None or action.ex_date <= end_date],
            key=lambda x: x.ex_date,
            reverse=True
        )
        
        for action in symbol_actions:
            # Find the adjustment date in the data
            action_date = pd.Timestamp(action.ex_date)
            
            if action_date not in adjusted_data.index:
                # Find the closest date before the action
                before_action = adjusted_data.index[adjusted_data.index < action_date]
                if len(before_action) == 0:
                    continue
                action_date = before_action[-1]
            
            # Apply adjustments to data before the action date
            before_action_mask = adjusted_data.index <= action_date
            
            if action.action_type == CorporateActionType.STOCK_SPLIT:
                # Adjust prices for stock split
                adjustment_factor = Decimal(str(1.0 / action.split_ratio))
                
                for col in adjusted_data.columns:
                    if col.lower() in ['open', 'high', 'low', 'close', 'adj_close']:
                        adjusted_data.loc[before_action_mask, col] *= float(adjustment_factor)
                    elif col.lower() == 'volume':
                        adjusted_data.loc[before_action_mask, col] *= action.split_ratio
            
            elif action.action_type == CorporateActionType.DIVIDEND:
                # Adjust prices for dividend
                dividend_adjustment = float(action.dividend_amount)
                
                for col in adjusted_data.columns:
                    if col.lower() in ['open', 'high', 'low', 'close', 'adj_close']:
                        adjusted_data.loc[before_action_mask, col] -= dividend_adjustment
        
        return adjusted_data
    
    def get_actions_for_symbol(self, symbol: str) -> List[CorporateAction]:
        """Get all corporate actions for a symbol."""
        return self.actions.get(symbol, []).copy()
    
    def get_actions_in_period(self, start_date: date, end_date: date) -> List[CorporateAction]:
        """Get all corporate actions within a date range."""
        actions_in_period = []
        
        for symbol_actions in self.actions.values():
            for action in symbol_actions:
                if start_date <= action.ex_date <= end_date:
                    actions_in_period.append(action)
        
        return sorted(actions_in_period, key=lambda x: x.ex_date)
    
    def clear_processed_actions(self) -> None:
        """Clear the list of processed actions."""
        self.processed_actions.clear()
    
    def reset(self) -> None:
        """Reset the handler, clearing all actions and processed actions."""
        self.actions.clear()
        self.processed_actions.clear()