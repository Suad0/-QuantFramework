"""
Comprehensive dashboard system for real-time portfolio monitoring,
risk metrics, and strategy performance comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QGridLayout, QProgressBar, QFrame, QScrollArea,
    QDockWidget, QMainWindow, QComboBox, QSpinBox, QCheckBox,
    QTextEdit, QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer, QThread, QObject
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap, QPainter

import pyqtgraph as pg
from pyqtgraph import PlotWidget, BarGraphItem

from .base_widget import ContainerWidget
from .chart_widgets import RealTimeChartWidget, LineChartWidget
try:
    from ....application.services.portfolio_service import PortfolioService
    from ....application.services.strategy_service import StrategyService
    from ....application.services.data_service import DataService
except ImportError:
    # Fallback for testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))
    from src.application.services.portfolio_service import PortfolioService
    from src.application.services.strategy_service import StrategyService
    from src.application.services.data_service import DataService


class MetricWidget(QFrame):
    """Widget for displaying a single metric with value and trend."""
    
    def __init__(self, title: str, value: str = "0.00", unit: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.current_value = value
        self.unit = unit
        self.trend_data = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the metric widget UI."""
        self.setFrameStyle(QFrame.StyledPanel)
        self.setFixedHeight(120)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setFont(QFont("Arial", 10, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Value
        self.value_label = QLabel(f"{self.current_value} {self.unit}")
        self.value_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)
        
        # Mini trend chart
        self.trend_chart = pg.PlotWidget()
        self.trend_chart.setMaximumHeight(40)
        self.trend_chart.hideAxis('left')
        self.trend_chart.hideAxis('bottom')
        self.trend_chart.setMouseEnabled(x=False, y=False)
        layout.addWidget(self.trend_chart)
    
    def update_value(self, value: float, unit: str = None):
        """Update the metric value and trend."""
        if unit:
            self.unit = unit
        
        self.current_value = f"{value:.2f}"
        self.value_label.setText(f"{self.current_value} {self.unit}")
        
        # Update trend data
        self.trend_data.append(value)
        if len(self.trend_data) > 50:  # Keep last 50 points
            self.trend_data.pop(0)
        
        # Update trend chart
        if len(self.trend_data) > 1:
            pen = pg.mkPen(color='green' if value >= 0 else 'red', width=2)
            self.trend_chart.clear()
            self.trend_chart.plot(self.trend_data, pen=pen)
    
    def set_alert_state(self, is_alert: bool):
        """Set alert state styling."""
        if is_alert:
            self.setStyleSheet("QFrame { border: 2px solid red; background-color: #ffeeee; }")
        else:
            self.setStyleSheet("QFrame { border: 1px solid gray; }")


class PortfolioMonitoringDashboard(ContainerWidget):
    """Real-time portfolio monitoring dashboard."""
    
    # Signals
    portfolio_updated = pyqtSignal(dict)
    alert_triggered = pyqtSignal(str, str)  # alert_type, message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.portfolio_service = PortfolioService()
        self.current_portfolio_data = {}
        self.update_timer = QTimer()
        
        self._setup_dashboard()
        self._setup_update_timer()
    
    def _setup_dashboard(self):
        """Setup the portfolio monitoring dashboard."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Header
        header = QLabel("Portfolio Monitoring Dashboard")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Create main content area
        content_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(content_splitter)
        
        # Left panel - Key metrics
        self._create_metrics_panel(content_splitter)
        
        # Right panel - Portfolio composition and performance
        self._create_portfolio_panel(content_splitter)
        
        content_splitter.setSizes([1, 2])
    
    def _create_metrics_panel(self, parent):
        """Create key metrics panel."""
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout()
        metrics_widget.setLayout(metrics_layout)
        
        # Metrics grid
        metrics_grid = QGridLayout()
        
        # Create metric widgets
        self.metrics = {
            'total_value': MetricWidget("Total Value", "0.00", "$"),
            'daily_pnl': MetricWidget("Daily P&L", "0.00", "$"),
            'total_return': MetricWidget("Total Return", "0.00", "%"),
            'sharpe_ratio': MetricWidget("Sharpe Ratio", "0.00", ""),
            'max_drawdown': MetricWidget("Max Drawdown", "0.00", "%"),
            'volatility': MetricWidget("Volatility", "0.00", "%")
        }
        
        # Arrange metrics in grid
        row, col = 0, 0
        for metric_widget in self.metrics.values():
            metrics_grid.addWidget(metric_widget, row, col)
            col += 1
            if col >= 2:
                col = 0
                row += 1
        
        metrics_layout.addLayout(metrics_grid)
        metrics_layout.addStretch()
        
        parent.addWidget(metrics_widget)
    
    def _create_portfolio_panel(self, parent):
        """Create portfolio composition and performance panel."""
        portfolio_widget = QWidget()
        portfolio_layout = QVBoxLayout()
        portfolio_widget.setLayout(portfolio_layout)
        
        # Tab widget for different views
        tab_widget = QTabWidget()
        portfolio_layout.addWidget(tab_widget)
        
        # Holdings tab
        self._create_holdings_tab(tab_widget)
        
        # Performance tab
        self._create_performance_tab(tab_widget)
        
        # Allocation tab
        self._create_allocation_tab(tab_widget)
        
        parent.addWidget(portfolio_widget)
    
    def _create_holdings_tab(self, tab_widget):
        """Create holdings table tab."""
        holdings_widget = QWidget()
        holdings_layout = QVBoxLayout()
        holdings_widget.setLayout(holdings_layout)
        
        # Holdings table
        self.holdings_table = QTableWidget()
        self.holdings_table.setColumnCount(8)
        self.holdings_table.setHorizontalHeaderLabels([
            "Symbol", "Quantity", "Price", "Market Value", 
            "Weight", "Day Change", "Total Return", "P&L"
        ])
        
        # Configure table
        header = self.holdings_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.holdings_table.setAlternatingRowColors(True)
        self.holdings_table.setSortingEnabled(True)
        
        holdings_layout.addWidget(self.holdings_table)
        
        tab_widget.addTab(holdings_widget, "Holdings")
    
    def _create_performance_tab(self, tab_widget):
        """Create performance chart tab."""
        performance_widget = QWidget()
        performance_layout = QVBoxLayout()
        performance_widget.setLayout(performance_layout)
        
        # Performance chart
        self.performance_chart = RealTimeChartWidget()
        self.performance_chart.set_title("Portfolio Performance")
        performance_layout.addWidget(self.performance_chart)
        
        tab_widget.addTab(performance_widget, "Performance")
    
    def _create_allocation_tab(self, tab_widget):
        """Create allocation pie chart tab."""
        allocation_widget = QWidget()
        allocation_layout = QVBoxLayout()
        allocation_widget.setLayout(allocation_layout)
        
        # Allocation chart (using bar chart as placeholder)
        self.allocation_chart = PlotWidget()
        self.allocation_chart.setLabel('left', 'Weight (%)')
        self.allocation_chart.setLabel('bottom', 'Assets')
        allocation_layout.addWidget(self.allocation_chart)
        
        tab_widget.addTab(allocation_widget, "Allocation")
    
    def _setup_update_timer(self):
        """Setup automatic update timer."""
        self.update_timer.timeout.connect(self._update_dashboard)
        self.update_timer.start(5000)  # Update every 5 seconds
    
    def update_portfolio_data(self, data: Dict[str, Any]):
        """Update dashboard with new portfolio data."""
        self.current_portfolio_data = data
        self._update_dashboard()
    
    def _update_dashboard(self):
        """Update all dashboard components."""
        if not self.current_portfolio_data:
            return
        
        try:
            # Update metrics
            self._update_metrics()
            
            # Update holdings table
            self._update_holdings_table()
            
            # Update performance chart
            self._update_performance_chart()
            
            # Update allocation chart
            self._update_allocation_chart()
            
            # Emit update signal
            self.portfolio_updated.emit(self.current_portfolio_data)
            
        except Exception as e:
            print(f"Error updating portfolio dashboard: {e}")
    
    def _update_metrics(self):
        """Update key metrics widgets."""
        data = self.current_portfolio_data
        
        # Calculate metrics from portfolio data
        if 'portfolio_returns' in data:
            returns = data['portfolio_returns']
            if not returns.empty:
                total_return = (returns.cumsum().iloc[-1]) * 100
                volatility = returns.std() * np.sqrt(252) * 100
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                max_drawdown = self._calculate_max_drawdown(returns) * 100
                
                self.metrics['total_return'].update_value(total_return)
                self.metrics['volatility'].update_value(volatility)
                self.metrics['sharpe_ratio'].update_value(sharpe_ratio)
                self.metrics['max_drawdown'].update_value(max_drawdown)
        
        # Update other metrics if available
        if 'portfolio_weights' in data:
            weights = data['portfolio_weights']
            total_value = sum(weights.values()) * 100000  # Assume $100k portfolio
            self.metrics['total_value'].update_value(total_value)
    
    def _update_holdings_table(self):
        """Update holdings table."""
        if 'portfolio_weights' not in self.current_portfolio_data:
            return
        
        weights = self.current_portfolio_data['portfolio_weights']
        market_data = self.current_portfolio_data.get('market_data', pd.DataFrame())
        
        self.holdings_table.setRowCount(len(weights))
        
        for row, (symbol, weight) in enumerate(weights.items()):
            # Symbol
            self.holdings_table.setItem(row, 0, QTableWidgetItem(symbol))
            
            # Weight
            weight_item = QTableWidgetItem(f"{weight*100:.2f}%")
            self.holdings_table.setItem(row, 4, weight_item)
            
            # Get price data if available
            if not market_data.empty and symbol in market_data.columns:
                price_data = market_data[symbol].dropna()
                if not price_data.empty:
                    current_price = price_data.iloc[-1]
                    self.holdings_table.setItem(row, 2, QTableWidgetItem(f"${current_price:.2f}"))
                    
                    if len(price_data) > 1:
                        day_change = (current_price - price_data.iloc[-2]) / price_data.iloc[-2] * 100
                        change_item = QTableWidgetItem(f"{day_change:.2f}%")
                        change_item.setForeground(QColor('green' if day_change >= 0 else 'red'))
                        self.holdings_table.setItem(row, 5, change_item)
    
    def _update_performance_chart(self):
        """Update performance chart."""
        if 'portfolio_returns' not in self.current_portfolio_data:
            return
        
        returns = self.current_portfolio_data['portfolio_returns']
        if not returns.empty:
            cumulative_returns = (1 + returns).cumprod()
            self.performance_chart.update_data(cumulative_returns.values, x_data=returns.index)
    
    def _update_allocation_chart(self):
        """Update allocation chart."""
        if 'portfolio_weights' not in self.current_portfolio_data:
            return
        
        weights = self.current_portfolio_data['portfolio_weights']
        
        # Create bar chart
        self.allocation_chart.clear()
        
        symbols = list(weights.keys())
        values = [w * 100 for w in weights.values()]
        
        x = np.arange(len(symbols))
        bar_graph = BarGraphItem(x=x, height=values, width=0.6, brush='b')
        self.allocation_chart.addItem(bar_graph)
        
        # Set x-axis labels
        self.allocation_chart.getAxis('bottom').setTicks([[(i, symbol) for i, symbol in enumerate(symbols)]])
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


class RiskMetricsDashboard(ContainerWidget):
    """Risk metrics dashboard with alert system."""
    
    # Signals
    risk_alert = pyqtSignal(str, str, float)  # alert_type, message, severity
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.risk_limits = {
            'var_95': 0.05,
            'max_drawdown': 0.20,
            'volatility': 0.25,
            'concentration': 0.30
        }
        self.current_alerts = []
        
        self._setup_dashboard()
    
    def _setup_dashboard(self):
        """Setup risk metrics dashboard."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Header
        header = QLabel("Risk Metrics Dashboard")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Create main content
        content_splitter = QSplitter(Qt.Vertical)
        layout.addWidget(content_splitter)
        
        # Risk metrics panel
        self._create_risk_metrics_panel(content_splitter)
        
        # Alerts panel
        self._create_alerts_panel(content_splitter)
        
        content_splitter.setSizes([2, 1])
    
    def _create_risk_metrics_panel(self, parent):
        """Create risk metrics panel."""
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout()
        metrics_widget.setLayout(metrics_layout)
        
        # Risk metrics grid
        metrics_grid = QGridLayout()
        
        # Create risk metric widgets
        self.risk_metrics = {
            'var_95': MetricWidget("VaR (95%)", "0.00", "%"),
            'var_99': MetricWidget("VaR (99%)", "0.00", "%"),
            'expected_shortfall': MetricWidget("Expected Shortfall", "0.00", "%"),
            'beta': MetricWidget("Beta", "0.00", ""),
            'tracking_error': MetricWidget("Tracking Error", "0.00", "%"),
            'information_ratio': MetricWidget("Information Ratio", "0.00", "")
        }
        
        # Arrange in grid
        row, col = 0, 0
        for metric_widget in self.risk_metrics.values():
            metrics_grid.addWidget(metric_widget, row, col)
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        metrics_layout.addLayout(metrics_grid)
        
        # Risk limits configuration
        self._create_risk_limits_panel(metrics_layout)
        
        parent.addWidget(metrics_widget)
    
    def _create_risk_limits_panel(self, parent_layout):
        """Create risk limits configuration panel."""
        limits_group = QGroupBox("Risk Limits")
        limits_layout = QGridLayout()
        limits_group.setLayout(limits_layout)
        
        self.limit_controls = {}
        
        row = 0
        for limit_name, limit_value in self.risk_limits.items():
            # Label
            label = QLabel(limit_name.replace('_', ' ').title() + ":")
            limits_layout.addWidget(label, row, 0)
            
            # Spin box for limit value
            spin_box = QSpinBox()
            spin_box.setRange(1, 100)
            spin_box.setValue(int(limit_value * 100))
            spin_box.setSuffix("%")
            spin_box.valueChanged.connect(
                lambda value, name=limit_name: self._update_risk_limit(name, value/100)
            )
            limits_layout.addWidget(spin_box, row, 1)
            
            # Enable checkbox
            checkbox = QCheckBox("Enabled")
            checkbox.setChecked(True)
            limits_layout.addWidget(checkbox, row, 2)
            
            self.limit_controls[limit_name] = {
                'spin_box': spin_box,
                'checkbox': checkbox
            }
            
            row += 1
        
        parent_layout.addWidget(limits_group)
    
    def _create_alerts_panel(self, parent):
        """Create alerts panel."""
        alerts_widget = QWidget()
        alerts_layout = QVBoxLayout()
        alerts_widget.setLayout(alerts_layout)
        
        # Alerts header
        alerts_header = QLabel("Risk Alerts")
        alerts_header.setFont(QFont("Arial", 12, QFont.Bold))
        alerts_layout.addWidget(alerts_header)
        
        # Alerts list
        self.alerts_list = QListWidget()
        alerts_layout.addWidget(self.alerts_list)
        
        # Alert controls
        controls_layout = QHBoxLayout()
        
        acknowledge_btn = QPushButton("Acknowledge Selected")
        acknowledge_btn.clicked.connect(self._acknowledge_alert)
        controls_layout.addWidget(acknowledge_btn)
        
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self._clear_all_alerts)
        controls_layout.addWidget(clear_all_btn)
        
        controls_layout.addStretch()
        alerts_layout.addLayout(controls_layout)
        
        parent.addWidget(alerts_widget)
    
    def update_risk_data(self, data: Dict[str, Any]):
        """Update risk dashboard with new data."""
        try:
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(data)
            
            # Update metric widgets
            for metric_name, value in risk_metrics.items():
                if metric_name in self.risk_metrics:
                    self.risk_metrics[metric_name].update_value(value)
                    
                    # Check for limit breaches
                    self._check_risk_limits(metric_name, value)
            
        except Exception as e:
            print(f"Error updating risk dashboard: {e}")
    
    def _calculate_risk_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics from portfolio data."""
        metrics = {}
        
        if 'portfolio_returns' in data:
            returns = data['portfolio_returns']
            if not returns.empty:
                # VaR calculations
                metrics['var_95'] = np.percentile(returns, 5) * 100
                metrics['var_99'] = np.percentile(returns, 1) * 100
                
                # Expected Shortfall (CVaR)
                var_95 = np.percentile(returns, 5)
                tail_returns = returns[returns <= var_95]
                metrics['expected_shortfall'] = tail_returns.mean() * 100 if not tail_returns.empty else 0
                
                # Other metrics
                metrics['tracking_error'] = returns.std() * np.sqrt(252) * 100
                
        # Beta calculation (placeholder)
        metrics['beta'] = 1.0
        metrics['information_ratio'] = 0.5
        
        return metrics
    
    def _check_risk_limits(self, metric_name: str, value: float):
        """Check if risk limits are breached."""
        if metric_name in self.risk_limits:
            limit = self.risk_limits[metric_name]
            
            # Check if limit is breached (assuming negative values are bad for most metrics)
            if abs(value/100) > limit:
                self._trigger_alert(metric_name, value, limit)
                
                # Set alert state on metric widget
                if metric_name in self.risk_metrics:
                    self.risk_metrics[metric_name].set_alert_state(True)
            else:
                if metric_name in self.risk_metrics:
                    self.risk_metrics[metric_name].set_alert_state(False)
    
    def _trigger_alert(self, metric_name: str, current_value: float, limit: float):
        """Trigger a risk alert."""
        alert_message = f"{metric_name.replace('_', ' ').title()} breach: {current_value:.2f}% (limit: {limit*100:.1f}%)"
        
        # Add to alerts list
        alert_item = QListWidgetItem(f"[{datetime.now().strftime('%H:%M:%S')}] {alert_message}")
        alert_item.setForeground(QColor('red'))
        self.alerts_list.insertItem(0, alert_item)
        
        # Emit signal
        self.risk_alert.emit(metric_name, alert_message, 1.0)
    
    def _update_risk_limit(self, limit_name: str, value: float):
        """Update risk limit value."""
        self.risk_limits[limit_name] = value
    
    def _acknowledge_alert(self):
        """Acknowledge selected alert."""
        current_item = self.alerts_list.currentItem()
        if current_item:
            current_item.setForeground(QColor('gray'))
    
    def _clear_all_alerts(self):
        """Clear all alerts."""
        self.alerts_list.clear()


class StrategyComparisonDashboard(ContainerWidget):
    """Strategy performance comparison dashboard."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.strategy_service = StrategyService()
        self.strategies_data = {}
        
        self._setup_dashboard()
    
    def _setup_dashboard(self):
        """Setup strategy comparison dashboard."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Header
        header = QLabel("Strategy Performance Comparison")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Create main content
        content_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(content_splitter)
        
        # Strategy selection panel
        self._create_strategy_selection_panel(content_splitter)
        
        # Comparison charts panel
        self._create_comparison_panel(content_splitter)
        
        content_splitter.setSizes([1, 3])
    
    def _create_strategy_selection_panel(self, parent):
        """Create strategy selection panel."""
        selection_widget = QWidget()
        selection_layout = QVBoxLayout()
        selection_widget.setLayout(selection_layout)
        
        # Strategy list
        strategies_label = QLabel("Available Strategies:")
        strategies_label.setFont(QFont("Arial", 10, QFont.Bold))
        selection_layout.addWidget(strategies_label)
        
        self.strategy_tree = QTreeWidget()
        self.strategy_tree.setHeaderLabels(["Strategy", "Status"])
        selection_layout.addWidget(self.strategy_tree)
        
        # Add sample strategies
        self._populate_strategy_tree()
        
        # Controls
        controls_layout = QVBoxLayout()
        
        compare_btn = QPushButton("Compare Selected")
        compare_btn.clicked.connect(self._compare_strategies)
        controls_layout.addWidget(compare_btn)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_strategies)
        controls_layout.addWidget(refresh_btn)
        
        controls_layout.addStretch()
        selection_layout.addLayout(controls_layout)
        
        parent.addWidget(selection_widget)
    
    def _create_comparison_panel(self, parent):
        """Create strategy comparison panel."""
        comparison_widget = QWidget()
        comparison_layout = QVBoxLayout()
        comparison_widget.setLayout(comparison_layout)
        
        # Comparison tabs
        comparison_tabs = QTabWidget()
        comparison_layout.addWidget(comparison_tabs)
        
        # Performance comparison chart
        self.performance_chart = RealTimeChartWidget()
        self.performance_chart.set_title("Cumulative Returns Comparison")
        comparison_tabs.addTab(self.performance_chart, "Performance")
        
        # Metrics comparison table
        self.metrics_table = QTableWidget()
        comparison_tabs.addTab(self.metrics_table, "Metrics")
        
        # Risk comparison chart
        self.risk_chart = PlotWidget()
        self.risk_chart.setLabel('left', 'Risk (%)')
        self.risk_chart.setLabel('bottom', 'Return (%)')
        self.risk_chart.setTitle('Risk-Return Scatter')
        comparison_tabs.addTab(self.risk_chart, "Risk-Return")
        
        parent.addWidget(comparison_widget)
    
    def _populate_strategy_tree(self):
        """Populate strategy tree with sample strategies."""
        strategies = [
            ("Momentum Strategy", "Active"),
            ("Mean Reversion", "Active"),
            ("Buy and Hold", "Active"),
            ("Risk Parity", "Inactive"),
            ("Factor Model", "Active")
        ]
        
        for strategy_name, status in strategies:
            item = QTreeWidgetItem([strategy_name, status])
            item.setCheckState(0, Qt.Unchecked)
            self.strategy_tree.addTopLevelItem(item)
    
    def _compare_strategies(self):
        """Compare selected strategies."""
        selected_strategies = []
        
        # Get selected strategies
        for i in range(self.strategy_tree.topLevelItemCount()):
            item = self.strategy_tree.topLevelItem(i)
            if item.checkState(0) == Qt.Checked:
                selected_strategies.append(item.text(0))
        
        if not selected_strategies:
            return
        
        # Generate comparison data (placeholder)
        self._generate_comparison_data(selected_strategies)
        
        # Update comparison charts
        self._update_comparison_charts()
    
    def _generate_comparison_data(self, strategies: List[str]):
        """Generate comparison data for selected strategies."""
        # Generate sample data for demonstration
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        for strategy in strategies:
            # Generate random returns for each strategy
            np.random.seed(hash(strategy) % 2**32)  # Consistent random data
            returns = np.random.normal(0.0005, 0.02, len(dates))
            
            # Add strategy-specific characteristics
            if "Momentum" in strategy:
                returns += np.random.normal(0.0002, 0.01, len(dates))
            elif "Mean Reversion" in strategy:
                returns = -0.5 * returns + np.random.normal(0.0001, 0.015, len(dates))
            
            cumulative_returns = (1 + pd.Series(returns)).cumprod()
            
            self.strategies_data[strategy] = {
                'returns': pd.Series(returns, index=dates),
                'cumulative_returns': cumulative_returns,
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
                'max_drawdown': self._calculate_max_drawdown(pd.Series(returns)),
                'volatility': returns.std() * np.sqrt(252)
            }
    
    def _update_comparison_charts(self):
        """Update comparison charts with strategy data."""
        if not self.strategies_data:
            return
        
        # Update performance chart
        self.performance_chart.clear_data()
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (strategy, data) in enumerate(self.strategies_data.items()):
            color = colors[i % len(colors)]
            cumulative_returns = data['cumulative_returns']
            self.performance_chart.add_series(
                strategy, 
                cumulative_returns.values, 
                cumulative_returns.index,
                color=color
            )
        
        # Update metrics table
        self._update_metrics_table()
        
        # Update risk-return chart
        self._update_risk_return_chart()
    
    def _update_metrics_table(self):
        """Update metrics comparison table."""
        if not self.strategies_data:
            return
        
        metrics = ['sharpe_ratio', 'max_drawdown', 'volatility']
        
        self.metrics_table.setRowCount(len(metrics))
        self.metrics_table.setColumnCount(len(self.strategies_data) + 1)
        
        # Set headers
        headers = ['Metric'] + list(self.strategies_data.keys())
        self.metrics_table.setHorizontalHeaderLabels(headers)
        
        # Populate table
        for row, metric in enumerate(metrics):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(metric.replace('_', ' ').title()))
            
            for col, (strategy, data) in enumerate(self.strategies_data.items()):
                value = data[metric]
                if metric == 'max_drawdown':
                    value_str = f"{value*100:.2f}%"
                elif metric == 'volatility':
                    value_str = f"{value*100:.2f}%"
                else:
                    value_str = f"{value:.3f}"
                
                self.metrics_table.setItem(row, col + 1, QTableWidgetItem(value_str))
        
        # Resize columns
        self.metrics_table.resizeColumnsToContents()
    
    def _update_risk_return_chart(self):
        """Update risk-return scatter plot."""
        self.risk_chart.clear()
        
        colors = ['b', 'r', 'g', 'orange', 'purple']
        
        for i, (strategy, data) in enumerate(self.strategies_data.items()):
            returns = data['returns']
            annual_return = returns.mean() * 252 * 100
            annual_vol = data['volatility'] * 100
            
            # Add point to scatter plot
            scatter = pg.ScatterPlotItem(
                x=[annual_vol], 
                y=[annual_return], 
                brush=colors[i % len(colors)],
                size=10,
                name=strategy
            )
            self.risk_chart.addItem(scatter)
    
    def _refresh_strategies(self):
        """Refresh strategy list."""
        # Clear current data
        self.strategies_data.clear()
        
        # Clear charts
        self.performance_chart.clear_data()
        self.metrics_table.clear()
        self.risk_chart.clear()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


class DashboardSystem(QMainWindow):
    """Main dashboard system with dockable widgets."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Comprehensive Dashboard System")
        self.resize(1400, 900)
        
        self._setup_dashboard_system()
    
    def _setup_dashboard_system(self):
        """Setup the main dashboard system."""
        # Create central widget (portfolio monitoring)
        self.portfolio_dashboard = PortfolioMonitoringDashboard()
        self.setCentralWidget(self.portfolio_dashboard)
        
        # Create dockable widgets
        self._create_dockable_widgets()
        
        # Setup connections
        self._setup_connections()
    
    def _create_dockable_widgets(self):
        """Create dockable dashboard widgets."""
        # Risk metrics dock
        self.risk_dock = QDockWidget("Risk Metrics", self)
        self.risk_dashboard = RiskMetricsDashboard()
        self.risk_dock.setWidget(self.risk_dashboard)
        self.addDockWidget(Qt.RightDockWidgetArea, self.risk_dock)
        
        # Strategy comparison dock
        self.strategy_dock = QDockWidget("Strategy Comparison", self)
        self.strategy_dashboard = StrategyComparisonDashboard()
        self.strategy_dock.setWidget(self.strategy_dashboard)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.strategy_dock)
        
        # Allow docking on all sides
        self.setDockOptions(
            QMainWindow.AllowNestedDocks | 
            QMainWindow.AllowTabbedDocks |
            QMainWindow.AnimatedDocks
        )
    
    def _setup_connections(self):
        """Setup signal connections between dashboards."""
        # Connect portfolio updates to risk dashboard
        self.portfolio_dashboard.portfolio_updated.connect(
            self.risk_dashboard.update_risk_data
        )
        
        # Connect risk alerts to portfolio dashboard
        self.risk_dashboard.risk_alert.connect(
            lambda alert_type, message, severity: 
            self.portfolio_dashboard.alert_triggered.emit(alert_type, message)
        )
    
    def update_all_dashboards(self, data: Dict[str, Any]):
        """Update all dashboards with new data."""
        self.portfolio_dashboard.update_portfolio_data(data)
        self.risk_dashboard.update_risk_data(data)