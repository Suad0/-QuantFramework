"""
Real-time risk monitoring system with configurable alerts.

This module provides comprehensive risk monitoring capabilities including:
- Real-time portfolio risk tracking
- Configurable risk limits and alerts
- Risk dashboard and reporting
- Automated risk notifications
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from threading import Thread, Event
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ...domain.entities import Portfolio, Position
from ...domain.value_objects import RiskMetrics
from ...domain.exceptions import ValidationError
from .var_calculator import VaRCalculator, VaRMethod


class AlertSeverity(Enum):
    """Risk alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of risk alerts."""
    VAR_BREACH = "var_breach"
    CONCENTRATION = "concentration"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    LIQUIDITY = "liquidity"
    CORRELATION = "correlation"
    LEVERAGE = "leverage"
    SECTOR_EXPOSURE = "sector_exposure"


@dataclass
class RiskLimit:
    """Risk limit configuration."""
    name: str
    limit_type: AlertType
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    lookback_days: int = 252
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAlert:
    """Risk alert notification."""
    alert_id: str
    portfolio_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'portfolio_id': self.portfolio_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'metadata': self.metadata
        }


class RiskMonitor:
    """
    Real-time risk monitoring system.
    
    This class provides continuous monitoring of portfolio risk metrics
    and generates alerts when risk limits are breached.
    """
    
    def __init__(
        self,
        var_calculator: Optional[VaRCalculator] = None,
        monitoring_interval: int = 60,  # seconds
        max_workers: int = 4
    ):
        """
        Initialize risk monitor.
        
        Args:
            var_calculator: VaR calculator instance
            monitoring_interval: Monitoring frequency in seconds
            max_workers: Maximum number of worker threads
        """
        self.var_calculator = var_calculator or VaRCalculator()
        self.monitoring_interval = monitoring_interval
        self.max_workers = max_workers
        
        # Risk limits and alerts
        self.risk_limits: Dict[str, List[RiskLimit]] = {}
        self.active_alerts: Dict[str, List[RiskAlert]] = {}
        self.alert_history: List[RiskAlert] = []
        
        # Monitoring control
        self.is_monitoring = False
        self.stop_event = Event()
        self.monitoring_thread: Optional[Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Alert handlers
        self.alert_handlers: List[Callable[[RiskAlert], None]] = []
        
        # Portfolio data cache
        self.portfolio_cache: Dict[str, Dict[str, Any]] = {}
        
    def add_risk_limit(self, portfolio_id: str, risk_limit: RiskLimit) -> None:
        """Add a risk limit for a portfolio."""
        if portfolio_id not in self.risk_limits:
            self.risk_limits[portfolio_id] = []
        
        self.risk_limits[portfolio_id].append(risk_limit)
    
    def remove_risk_limit(self, portfolio_id: str, limit_name: str) -> bool:
        """Remove a risk limit."""
        if portfolio_id not in self.risk_limits:
            return False
        
        original_count = len(self.risk_limits[portfolio_id])
        self.risk_limits[portfolio_id] = [
            limit for limit in self.risk_limits[portfolio_id]
            if limit.name != limit_name
        ]
        
        return len(self.risk_limits[portfolio_id]) < original_count
    
    def add_alert_handler(self, handler: Callable[[RiskAlert], None]) -> None:
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def start_monitoring(self) -> None:
        """Start real-time risk monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop risk monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.stop_event.is_set():
            try:
                # Monitor all portfolios with risk limits
                futures = []
                for portfolio_id in self.risk_limits.keys():
                    future = self.executor.submit(self._monitor_portfolio, portfolio_id)
                    futures.append(future)
                
                # Wait for all monitoring tasks to complete
                for future in futures:
                    try:
                        future.result(timeout=self.monitoring_interval / 2)
                    except Exception as e:
                        print(f"Error monitoring portfolio: {e}")
                
                # Wait for next monitoring cycle
                self.stop_event.wait(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _monitor_portfolio(self, portfolio_id: str) -> None:
        """Monitor a single portfolio for risk limit breaches."""
        try:
            # Get portfolio data (this would typically come from a data service)
            portfolio_data = self._get_portfolio_data(portfolio_id)
            if not portfolio_data:
                return
            
            # Check each risk limit
            for risk_limit in self.risk_limits.get(portfolio_id, []):
                if not risk_limit.enabled:
                    continue
                
                alert = self._check_risk_limit(portfolio_id, portfolio_data, risk_limit)
                if alert:
                    self._handle_alert(alert)
                    
        except Exception as e:
            print(f"Error monitoring portfolio {portfolio_id}: {e}")
    
    def _get_portfolio_data(self, portfolio_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio data for monitoring."""
        # This is a placeholder - in practice, this would fetch real portfolio data
        # from a data service or database
        
        # Check cache first
        if portfolio_id in self.portfolio_cache:
            cache_entry = self.portfolio_cache[portfolio_id]
            if (datetime.now() - cache_entry['timestamp']).seconds < 300:  # 5 min cache
                return cache_entry['data']
        
        # Generate sample data for demonstration
        np.random.seed(hash(portfolio_id) % 2**32)
        
        portfolio_data = {
            'portfolio_id': portfolio_id,
            'total_value': 1000000 + np.random.normal(0, 50000),
            'returns': pd.Series(np.random.normal(0.001, 0.02, 252)),  # Daily returns
            'positions': {
                'AAPL': {'weight': 0.15, 'value': 150000},
                'GOOGL': {'weight': 0.12, 'value': 120000},
                'MSFT': {'weight': 0.10, 'value': 100000},
                'TSLA': {'weight': 0.08, 'value': 80000},
                'AMZN': {'weight': 0.07, 'value': 70000},
            },
            'sector_exposures': {
                'Technology': 0.52,
                'Healthcare': 0.15,
                'Financials': 0.12,
                'Consumer': 0.21
            },
            'beta': 1.2 + np.random.normal(0, 0.1),
            'volatility': 0.15 + np.random.normal(0, 0.02),
            'max_drawdown': -0.08 + np.random.normal(0, 0.02),
            'timestamp': datetime.now()
        }
        
        # Cache the data
        self.portfolio_cache[portfolio_id] = {
            'data': portfolio_data,
            'timestamp': datetime.now()
        }
        
        return portfolio_data
    
    def _check_risk_limit(
        self,
        portfolio_id: str,
        portfolio_data: Dict[str, Any],
        risk_limit: RiskLimit
    ) -> Optional[RiskAlert]:
        """Check if a risk limit is breached."""
        try:
            current_value = self._calculate_risk_metric(portfolio_data, risk_limit)
            
            if self._is_limit_breached(current_value, risk_limit):
                alert_id = f"{portfolio_id}_{risk_limit.name}_{int(datetime.now().timestamp())}"
                
                return RiskAlert(
                    alert_id=alert_id,
                    portfolio_id=portfolio_id,
                    alert_type=risk_limit.limit_type,
                    severity=risk_limit.severity,
                    message=self._generate_alert_message(risk_limit, current_value),
                    current_value=current_value,
                    threshold=risk_limit.threshold,
                    timestamp=datetime.now(),
                    metadata=risk_limit.metadata.copy()
                )
            
            return None
            
        except Exception as e:
            print(f"Error checking risk limit {risk_limit.name}: {e}")
            return None
    
    def _calculate_risk_metric(
        self,
        portfolio_data: Dict[str, Any],
        risk_limit: RiskLimit
    ) -> float:
        """Calculate the risk metric value."""
        if risk_limit.limit_type == AlertType.VAR_BREACH:
            returns = portfolio_data['returns']
            var_result = self.var_calculator.calculate_var(
                returns,
                method=VaRMethod.HISTORICAL,
                portfolio_value=portfolio_data['total_value']
            )
            return var_result.var_95
            
        elif risk_limit.limit_type == AlertType.CONCENTRATION:
            positions = portfolio_data['positions']
            max_weight = max(pos['weight'] for pos in positions.values())
            return max_weight
            
        elif risk_limit.limit_type == AlertType.VOLATILITY:
            return portfolio_data['volatility']
            
        elif risk_limit.limit_type == AlertType.DRAWDOWN:
            return abs(portfolio_data['max_drawdown'])
            
        elif risk_limit.limit_type == AlertType.SECTOR_EXPOSURE:
            sector_exposures = portfolio_data['sector_exposures']
            max_sector_exposure = max(sector_exposures.values())
            return max_sector_exposure
            
        elif risk_limit.limit_type == AlertType.LEVERAGE:
            # Calculate leverage as sum of absolute weights
            positions = portfolio_data['positions']
            leverage = sum(abs(pos['weight']) for pos in positions.values())
            return leverage
            
        else:
            return 0.0
    
    def _is_limit_breached(self, current_value: float, risk_limit: RiskLimit) -> bool:
        """Check if the current value breaches the risk limit."""
        if risk_limit.limit_type in [AlertType.VAR_BREACH, AlertType.DRAWDOWN]:
            # For VaR and drawdown, breach occurs when value exceeds threshold
            return current_value > risk_limit.threshold
        else:
            # For other metrics, breach occurs when value exceeds threshold
            return current_value > risk_limit.threshold
    
    def _generate_alert_message(self, risk_limit: RiskLimit, current_value: float) -> str:
        """Generate alert message."""
        messages = {
            AlertType.VAR_BREACH: f"VaR limit breached: {current_value:.2f} > {risk_limit.threshold:.2f}",
            AlertType.CONCENTRATION: f"Concentration limit breached: {current_value:.2%} > {risk_limit.threshold:.2%}",
            AlertType.VOLATILITY: f"Volatility limit breached: {current_value:.2%} > {risk_limit.threshold:.2%}",
            AlertType.DRAWDOWN: f"Drawdown limit breached: {current_value:.2%} > {risk_limit.threshold:.2%}",
            AlertType.SECTOR_EXPOSURE: f"Sector exposure limit breached: {current_value:.2%} > {risk_limit.threshold:.2%}",
            AlertType.LEVERAGE: f"Leverage limit breached: {current_value:.2f} > {risk_limit.threshold:.2f}",
        }
        
        return messages.get(
            risk_limit.limit_type,
            f"Risk limit '{risk_limit.name}' breached: {current_value} > {risk_limit.threshold}"
        )
    
    def _handle_alert(self, alert: RiskAlert) -> None:
        """Handle a new risk alert."""
        # Add to active alerts
        if alert.portfolio_id not in self.active_alerts:
            self.active_alerts[alert.portfolio_id] = []
        
        self.active_alerts[alert.portfolio_id].append(alert)
        self.alert_history.append(alert)
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error in alert handler: {e}")
    
    def get_active_alerts(
        self,
        portfolio_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[RiskAlert]:
        """Get active alerts with optional filtering."""
        alerts = []
        
        if portfolio_id:
            alerts = self.active_alerts.get(portfolio_id, [])
        else:
            for portfolio_alerts in self.active_alerts.values():
                alerts.extend(portfolio_alerts)
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return [alert for alert in alerts if not alert.acknowledged]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for portfolio_alerts in self.active_alerts.values():
            for alert in portfolio_alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledge()
                    return True
        return False
    
    def clear_acknowledged_alerts(self, portfolio_id: Optional[str] = None) -> int:
        """Clear acknowledged alerts."""
        cleared_count = 0
        
        if portfolio_id:
            if portfolio_id in self.active_alerts:
                original_count = len(self.active_alerts[portfolio_id])
                self.active_alerts[portfolio_id] = [
                    alert for alert in self.active_alerts[portfolio_id]
                    if not alert.acknowledged
                ]
                cleared_count = original_count - len(self.active_alerts[portfolio_id])
        else:
            for pid in self.active_alerts:
                original_count = len(self.active_alerts[pid])
                self.active_alerts[pid] = [
                    alert for alert in self.active_alerts[pid]
                    if not alert.acknowledged
                ]
                cleared_count += original_count - len(self.active_alerts[pid])
        
        return cleared_count
    
    def get_risk_dashboard(self, portfolio_id: str) -> Dict[str, Any]:
        """Get risk dashboard data for a portfolio."""
        portfolio_data = self._get_portfolio_data(portfolio_id)
        if not portfolio_data:
            return {}
        
        # Calculate current risk metrics
        returns = portfolio_data['returns']
        var_result = self.var_calculator.calculate_var(
            returns,
            method=VaRMethod.HISTORICAL,
            portfolio_value=portfolio_data['total_value']
        )
        
        # Get active alerts
        active_alerts = self.get_active_alerts(portfolio_id)
        
        # Calculate risk utilization
        risk_utilization = {}
        for risk_limit in self.risk_limits.get(portfolio_id, []):
            if risk_limit.enabled:
                current_value = self._calculate_risk_metric(portfolio_data, risk_limit)
                utilization = (current_value / risk_limit.threshold) * 100
                risk_utilization[risk_limit.name] = {
                    'current': current_value,
                    'limit': risk_limit.threshold,
                    'utilization': utilization,
                    'status': 'breach' if utilization > 100 else 'warning' if utilization > 80 else 'ok'
                }
        
        return {
            'portfolio_id': portfolio_id,
            'timestamp': datetime.now(),
            'total_value': portfolio_data['total_value'],
            'var_95': var_result.var_95,
            'var_99': var_result.var_99,
            'cvar_95': var_result.cvar_95,
            'volatility': portfolio_data['volatility'],
            'max_drawdown': portfolio_data['max_drawdown'],
            'beta': portfolio_data['beta'],
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'risk_utilization': risk_utilization,
            'sector_exposures': portfolio_data['sector_exposures'],
            'top_positions': dict(list(portfolio_data['positions'].items())[:5])
        }
    
    def export_alerts(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        portfolio_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Export alert history to DataFrame."""
        alerts = self.alert_history
        
        # Filter by date range
        if start_date:
            alerts = [a for a in alerts if a.timestamp >= start_date]
        if end_date:
            alerts = [a for a in alerts if a.timestamp <= end_date]
        if portfolio_id:
            alerts = [a for a in alerts if a.portfolio_id == portfolio_id]
        
        if not alerts:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = [alert.to_dict() for alert in alerts]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df.sort_values('timestamp', ascending=False)


# Example alert handlers
def console_alert_handler(alert: RiskAlert) -> None:
    """Simple console alert handler."""
    print(f"[{alert.severity.value.upper()}] {alert.timestamp}: {alert.message}")


def email_alert_handler(alert: RiskAlert) -> None:
    """Email alert handler (placeholder)."""
    # In practice, this would send an email
    print(f"EMAIL ALERT: {alert.message}")


def slack_alert_handler(alert: RiskAlert) -> None:
    """Slack alert handler (placeholder)."""
    # In practice, this would send a Slack message
    print(f"SLACK ALERT: {alert.message}")