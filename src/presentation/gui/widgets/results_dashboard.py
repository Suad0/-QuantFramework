"""
Interactive results dashboard with charts and visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QScrollArea, QGridLayout, QSplitter, QTabWidget
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPainter, QPen, QBrush, QColor
import pyqtgraph as pg
from pyqtgraph import PlotWidget, BarGraphItem


class MetricCard(QFrame):
    """A card widget to display a single metric."""
    
    def __init__(self, title: str, value: str = "N/A", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setProperty("class", "metric_card")
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        self.title_label = QLabel(title)
        self.title_label.setProperty("class", "metric_title")
        layout.addWidget(self.title_label)
        
        # Value
        self.value_label = QLabel(value)
        self.value_label.setProperty("class", "metric_value")
        layout.addWidget(self.value_label)
        
        # Subtitle
        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setProperty("class", "metric_subtitle")
            layout.addWidget(self.subtitle_label)
    
    def update_value(self, value: str, subtitle: str = ""):
        """Update the metric value."""
        self.value_label.setText(value)
        if hasattr(self, 'subtitle_label') and subtitle:
            self.subtitle_label.setText(subtitle)


class InteractiveChart(QWidget):
    """Interactive chart widget using PyQtGraph."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the chart UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title_label = QLabel(self.title)
        title_label.setProperty("class", "chart_title")
        layout.addWidget(title_label)
        
        # Chart
        self.plot_widget = PlotWidget()
        self.plot_widget.setBackground('#2d2d2d')
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='#ffffff'))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color='#ffffff'))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color='#ffffff'))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='#ffffff'))
        
        # Ensure the plot widget is visible and has proper size
        self.plot_widget.setMinimumHeight(200)
        self.plot_widget.setMinimumWidth(300)
        self.plot_widget.show()
        
        layout.addWidget(self.plot_widget)
        print(f"Created chart widget: {self.title}")  # Debug
    
    def plot_line(self, x_data, y_data, name: str = "", color: str = '#4a9eff'):
        """Plot a line chart."""
        print(f"Plotting line chart: {name} with {len(x_data)} data points")  # Debug
        self.plot_widget.clear()
        pen = pg.mkPen(color=color, width=2)
        self.plot_widget.plot(x_data, y_data, pen=pen, name=name)
        self.plot_widget.autoRange()  # Ensure proper scaling
        print(f"Line chart plotted for {self.title}")  # Debug
    
    def plot_multiple_lines(self, data_dict: Dict[str, tuple]):
        """Plot multiple lines on the same chart."""
        self.plot_widget.clear()
        colors = ['#4a9eff', '#28a745', '#dc3545', '#ffc107', '#17a2b8']
        
        for i, (name, (x_data, y_data)) in enumerate(data_dict.items()):
            color = colors[i % len(colors)]
            pen = pg.mkPen(color=color, width=2)
            self.plot_widget.plot(x_data, y_data, pen=pen, name=name)
        
        # Add legend
        self.plot_widget.addLegend()
    
    def plot_bar(self, x_data, y_data, color: str = '#4a9eff'):
        """Plot a bar chart."""
        print(f"Plotting bar chart for {self.title} with {len(x_data)} bars")  # Debug
        self.plot_widget.clear()
        bar_item = BarGraphItem(x=x_data, height=y_data, width=0.8, brush=color)
        self.plot_widget.addItem(bar_item)
        self.plot_widget.autoRange()  # Ensure proper scaling
        print(f"Bar chart plotted for {self.title}")  # Debug


class PortfolioAnalysisTab(QWidget):
    """Portfolio analysis tab with interactive charts."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.results_data = None
    
    def _setup_ui(self):
        """Setup the portfolio analysis UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Metrics cards row
        metrics_frame = QFrame()
        metrics_frame.setMinimumHeight(120)
        metrics_frame.setStyleSheet("QFrame { background-color: transparent; }")
        metrics_layout = QHBoxLayout()
        metrics_frame.setLayout(metrics_layout)
        
        self.return_card = MetricCard("Total Return", "N/A", "Since inception")
        self.sharpe_card = MetricCard("Sharpe Ratio", "N/A", "Risk-adjusted return")
        self.volatility_card = MetricCard("Volatility", "N/A", "Annualized")
        self.max_dd_card = MetricCard("Max Drawdown", "N/A", "Peak to trough")
        
        # Ensure cards are visible
        for card in [self.return_card, self.sharpe_card, self.volatility_card, self.max_dd_card]:
            card.setMinimumHeight(100)
            card.setMinimumWidth(150)
        
        metrics_layout.addWidget(self.return_card)
        metrics_layout.addWidget(self.sharpe_card)
        metrics_layout.addWidget(self.volatility_card)
        metrics_layout.addWidget(self.max_dd_card)
        
        layout.addWidget(metrics_frame)
        
        # Charts section
        charts_splitter = QSplitter(Qt.Horizontal)
        charts_splitter.setMinimumHeight(300)
        
        # Portfolio performance chart
        self.performance_chart = InteractiveChart("Portfolio Performance")
        self.performance_chart.setMinimumWidth(400)
        charts_splitter.addWidget(self.performance_chart)
        
        # Portfolio weights chart
        self.weights_chart = InteractiveChart("Portfolio Weights")
        self.weights_chart.setMinimumWidth(400)
        charts_splitter.addWidget(self.weights_chart)
        
        layout.addWidget(charts_splitter)
        
        # Risk metrics chart
        self.risk_chart = InteractiveChart("Risk Metrics Over Time")
        self.risk_chart.setMinimumHeight(200)
        layout.addWidget(self.risk_chart)
    
    def update_data(self, results: Dict[str, Any]):
        """Update the tab with analysis results."""
        print("PortfolioAnalysisTab.update_data called")  # Debug
        self.results_data = results
        
        # Update metric cards
        risk_metrics = results.get('risk_metrics', {})
        print(f"Risk metrics: {risk_metrics}")  # Debug
        if risk_metrics:
            self.return_card.update_value(f"{risk_metrics.get('Mean_Return', 0):.2%}")
            self.sharpe_card.update_value(f"{risk_metrics.get('Sharpe_Ratio', 0):.2f}")
            self.volatility_card.update_value(f"{risk_metrics.get('Volatility', 0):.2%}")
            self.max_dd_card.update_value(f"{risk_metrics.get('Max_Drawdown', 0):.2%}")
            print("Metric cards updated")  # Debug
        
        # Update performance chart
        portfolio_returns = results.get('portfolio_returns')
        print(f"Portfolio returns type: {type(portfolio_returns)}, empty: {portfolio_returns.empty if portfolio_returns is not None else 'None'}")  # Debug
        if portfolio_returns is not None and not portfolio_returns.empty:
            cumulative_returns = (1 + portfolio_returns).cumprod()
            dates = range(len(cumulative_returns))
            self.performance_chart.plot_line(dates, cumulative_returns.values, 
                                           "Portfolio Performance", '#28a745')
            print("Performance chart updated")  # Debug
        
        # Update weights chart
        portfolio_weights = results.get('portfolio_weights', {})
        print(f"Portfolio weights: {portfolio_weights}")  # Debug
        if portfolio_weights:
            symbols = list(portfolio_weights.keys())
            weights = list(portfolio_weights.values())
            x_pos = range(len(symbols))
            self.weights_chart.plot_bar(x_pos, weights, '#4a9eff')
            
            # Set x-axis labels
            ax = self.weights_chart.plot_widget.getAxis('bottom')
            ax.setTicks([[(i, symbol) for i, symbol in enumerate(symbols)]])
            print("Weights chart updated")  # Debug


class MarketAnalysisTab(QWidget):
    """Market analysis tab with price charts and indicators."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.results_data = None
    
    def _setup_ui(self):
        """Setup the market analysis UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Price charts
        self.price_chart = InteractiveChart("Stock Prices")
        layout.addWidget(self.price_chart)
        
        # Signals chart
        self.signals_chart = InteractiveChart("Trading Signals")
        layout.addWidget(self.signals_chart)
    
    def update_data(self, results: Dict[str, Any]):
        """Update the tab with analysis results."""
        self.results_data = results
        
        # Update price chart
        market_data = results.get('market_data')
        if market_data is not None and not market_data.empty:
            price_data = {}
            for col in market_data.columns:
                if '_Close' in col:
                    symbol = col.replace('_Close', '')
                    dates = range(len(market_data))
                    price_data[symbol] = (dates, market_data[col].values)
            
            if price_data:
                self.price_chart.plot_multiple_lines(price_data)
        
        # Update signals chart
        signals = results.get('signals')
        if signals is not None and not signals.empty:
            signal_data = {}
            for col in signals.columns:
                if '_signal' in col:
                    symbol = col.replace('_signal', '')
                    dates = range(len(signals))
                    signal_data[f"{symbol} Signal"] = (dates, signals[col].values)
            
            if signal_data:
                self.signals_chart.plot_multiple_lines(signal_data)


class ResultsDashboard(QWidget):
    """Main results dashboard with tabs and interactive content."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the dashboard UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Portfolio Analysis Tab
        self.portfolio_tab = PortfolioAnalysisTab()
        self.tab_widget.addTab(self.portfolio_tab, "📊 Portfolio Analysis")
        
        # Market Analysis Tab
        self.market_tab = MarketAnalysisTab()
        self.tab_widget.addTab(self.market_tab, "📈 Market Analysis")
        
        # Risk Analysis Tab (placeholder for now)
        risk_tab = QWidget()
        risk_layout = QVBoxLayout()
        risk_tab.setLayout(risk_layout)
        
        risk_label = QLabel("🛡️ Risk Analysis")
        risk_label.setProperty("class", "placeholder_title")
        risk_layout.addWidget(risk_label)
        
        risk_desc = QLabel("Advanced risk metrics and VaR calculations will be displayed here.")
        risk_desc.setProperty("class", "placeholder_description")
        risk_layout.addWidget(risk_desc)
        
        self.tab_widget.addTab(risk_tab, "🛡️ Risk Analysis")
        
        layout.addWidget(self.tab_widget)
    
    def update_data(self, results: Dict[str, Any]):
        """Update all tabs with analysis results."""
        print("ResultsDashboard.update_data called")  # Debug
        self.portfolio_tab.update_data(results)
        self.market_tab.update_data(results)
        
        # Switch to portfolio tab to show results
        self.tab_widget.setCurrentIndex(0)
        print("ResultsDashboard updated and switched to portfolio tab")  # Debug