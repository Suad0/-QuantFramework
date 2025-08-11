"""
Backtesting infrastructure components.

This module provides comprehensive backtesting capabilities including:
- Realistic trading cost simulation
- Advanced order management system
- Corporate action handling
- Walk-forward analysis
- Bias detection and prevention
"""

from .engine import BacktestEngine
from .order_manager import OrderManager, Order, OrderType, OrderStatus, OrderSide
from .cost_simulator import TradingCostSimulator
from .corporate_actions import CorporateActionHandler, CorporateAction, CorporateActionType
from .walk_forward import WalkForwardAnalyzer, WalkForwardConfig
from .bias_detector import BiasDetector
from .performance_analyzer import PerformanceAnalyzer

__all__ = [
    'BacktestEngine',
    'OrderManager',
    'Order',
    'OrderType',
    'OrderStatus',
    'OrderSide',
    'TradingCostSimulator',
    'CorporateActionHandler',
    'CorporateAction',
    'CorporateActionType',
    'WalkForwardAnalyzer',
    'WalkForwardConfig',
    'BiasDetector',
    'PerformanceAnalyzer'
]