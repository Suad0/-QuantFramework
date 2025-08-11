"""
Comprehensive visualization dashboard for financial data analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter,
    QComboBox, QLabel, QPushButton, QCheckBox, QGroupBox,
    QScrollArea, QFrame, QGridLayout, QSpinBox, QSlider,
    QButtonGroup, QRadioButton
)
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont

from .base_widget import ContainerWidget
from .chart_widgets import (
    CandlestickChartWidget, LineChartWidget, MultiTimeframeWidget,
    RealTimeChartWidget, FinancialChartWidget
)


class VisualizationDashboard(ContainerWidget):
    """Main visualization dashboard with multiple chart types and layouts."""
    
    # Signals
    chart_type_changed = pyqtSignal(str)
    layout_changed = pyqtSignal(str)
    data_exported = pyqtSignal(str, str)  # chart_type, filename
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_data = {}
        self._chart_layouts = {
            'single': 'Single Chart',
            'dual_horizontal': 'Dual Horizontal',
            'dual_vertical': 'Dual Vertical',
            'quad': 'Quad View',
            'multi_timeframe': 'Multi-Timeframe'
        }
        self._current_layout = 'single'
        self._active_charts = {}
        
        self._setup_dashboard()
    
    def _setup_dashboard(self):
        """Setup the main dashboard structure."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create control panel
        self._create_control_panel()
        layout.addWidget(self._control_panel)
        
        # Create main chart area
        self._create_chart_area()
        layout.addWidget(self._chart_area)
        
        # Initialize with single chart layout
        self._setup_single_chart_layout()
    
    def _create_control_panel(self):
        """Create dashboard control panel."""
        self._control_panel = QFrame()
        self._control_panel.setMaximumHeight(80)
        control_layout = QVBoxLayout()
        self._control_panel.setLayout(control_layout)
        
        # First row of controls
        first_row = QHBoxLayout()
        control_layout.addLayout(first_row)
        
        # Layout selector
        first_row.addWidget(QLabel("Layout:"))
        self._layout_combo = QComboBox()
        for key, name in self._chart_layouts.items():
            self._layout_combo.addItem(name, key)
        self._layout_combo.currentTextChanged.connect(lambda text: self._on_layout_changed(self._layout_combo.currentData()))
        first_row.addWidget(self._layout_combo)
        
        # Chart type selector
        first_row.addWidget(QLabel("Chart Type:"))
        self._chart_type_combo = QComboBox()
        self._chart_type_combo.addItems(['Candlestick', 'Line', 'Real-time'])
        self._chart_type_combo.currentTextChanged.connect(self._on_chart_type_changed)
        first_row.addWidget(self._chart_type_combo)
        
        # Sync charts checkbox
        self._sync_charts_cb = QCheckBox("Sync Charts")
        self._sync_charts_cb.setChecked(True)
        self._sync_charts_cb.toggled.connect(self._on_sync_charts_toggled)
        first_row.addWidget(self._sync_charts_cb)
        
        first_row.addStretch()
        
        # Export button
        export_btn = QPushButton("Export Charts")
        export_btn.clicked.connect(self._export_charts)
        first_row.addWidget(export_btn)
        
        # Second row of controls
        second_row = QHBoxLayout()
        control_layout.addLayout(second_row)
        
        # Indicator controls
        self._create_indicator_controls(second_row)
        
        second_row.addStretch()
    
    def _create_indicator_controls(self, layout: QHBoxLayout):
        """Create technical indicator controls."""
        # Indicators group
        indicators_group = QGroupBox("Technical Indicators")
        indicators_layout = QHBoxLayout()
        indicators_group.setLayout(indicators_layout)
        layout.addWidget(indicators_group)
        
        # Common indicators
        self._indicator_checkboxes = {}
        indicators = ['SMA', 'EMA', 'Bollinger Bands', 'RSI', 'MACD', 'Volume']
        
        for indicator in indicators:
            cb = QCheckBox(indicator)
            cb.toggled.connect(lambda checked, ind=indicator: self._toggle_indicator(ind, checked))
            indicators_layout.addWidget(cb)
            self._indicator_checkboxes[indicator] = cb
    
    def _create_chart_area(self):
        """Create the main chart display area."""
        self._chart_area = QWidget()
        self._chart_layout = QVBoxLayout()
        self._chart_area.setLayout(self._chart_layout)
    
    def _setup_single_chart_layout(self):
        """Setup single chart layout."""
        self._clear_chart_area()
        
        # Create single candlestick chart
        chart = CandlestickChartWidget()
        self._setup_chart_connections(chart, 'main')
        
        self._chart_layout.addWidget(chart)
        self._active_charts['main'] = chart
        
        self._current_layout = 'single'
    
    def _setup_dual_horizontal_layout(self):
        """Setup dual horizontal chart layout."""
        self._clear_chart_area()
        
        # Create horizontal splitter
        splitter = QSplitter(Qt.Horizontal)
        self._chart_layout.addWidget(splitter)
        
        # Create two charts
        chart1 = CandlestickChartWidget()
        chart2 = LineChartWidget()
        
        self._setup_chart_connections(chart1, 'left')
        self._setup_chart_connections(chart2, 'right')
        
        splitter.addWidget(chart1)
        splitter.addWidget(chart2)
        splitter.setSizes([1, 1])  # Equal sizes
        
        self._active_charts['left'] = chart1
        self._active_charts['right'] = chart2
        
        self._current_layout = 'dual_horizontal'
    
    def _setup_dual_vertical_layout(self):
        """Setup dual vertical chart layout."""
        self._clear_chart_area()
        
        # Create vertical splitter
        splitter = QSplitter(Qt.Vertical)
        self._chart_layout.addWidget(splitter)
        
        # Create two charts
        chart1 = CandlestickChartWidget()
        chart2 = LineChartWidget()
        
        self._setup_chart_connections(chart1, 'top')
        self._setup_chart_connections(chart2, 'bottom')
        
        splitter.addWidget(chart1)
        splitter.addWidget(chart2)
        splitter.setSizes([2, 1])  # Top chart larger
        
        self._active_charts['top'] = chart1
        self._active_charts['bottom'] = chart2
        
        self._current_layout = 'dual_vertical'
    
    def _setup_quad_layout(self):
        """Setup quad chart layout."""
        self._clear_chart_area()
        
        # Create main splitter (horizontal)
        main_splitter = QSplitter(Qt.Horizontal)
        self._chart_layout.addWidget(main_splitter)
        
        # Create left and right splitters (vertical)
        left_splitter = QSplitter(Qt.Vertical)
        right_splitter = QSplitter(Qt.Vertical)
        
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        
        # Create four charts
        chart_tl = CandlestickChartWidget()  # Top-left
        chart_bl = LineChartWidget()         # Bottom-left
        chart_tr = LineChartWidget()         # Top-right
        chart_br = RealTimeChartWidget()     # Bottom-right
        
        self._setup_chart_connections(chart_tl, 'top_left')
        self._setup_chart_connections(chart_bl, 'bottom_left')
        self._setup_chart_connections(chart_tr, 'top_right')
        self._setup_chart_connections(chart_br, 'bottom_right')
        
        left_splitter.addWidget(chart_tl)
        left_splitter.addWidget(chart_bl)
        right_splitter.addWidget(chart_tr)
        right_splitter.addWidget(chart_br)
        
        # Set equal sizes
        main_splitter.setSizes([1, 1])
        left_splitter.setSizes([1, 1])
        right_splitter.setSizes([1, 1])
        
        self._active_charts['top_left'] = chart_tl
        self._active_charts['bottom_left'] = chart_bl
        self._active_charts['top_right'] = chart_tr
        self._active_charts['bottom_right'] = chart_br
        
        self._current_layout = 'quad'
    
    def _setup_multi_timeframe_layout(self):
        """Setup multi-timeframe layout."""
        self._clear_chart_area()
        
        # Create multi-timeframe widget
        multi_tf_widget = MultiTimeframeWidget()
        self._chart_layout.addWidget(multi_tf_widget)
        
        self._active_charts['multi_timeframe'] = multi_tf_widget
        
        self._current_layout = 'multi_timeframe'
    
    def _setup_chart_connections(self, chart: FinancialChartWidget, chart_id: str):
        """Setup connections for a chart widget."""
        chart.set_widget_id(chart_id)
        
        # Connect chart signals
        chart.timeframe_changed.connect(self._on_chart_timeframe_changed)
        chart.indicator_toggled.connect(self._on_chart_indicator_toggled)
        chart.zoom_changed.connect(self._on_chart_zoom_changed)
        
        # Connect to dashboard signals
        chart.action_requested.connect(self.action_requested.emit)
        chart.error_occurred.connect(self.error_occurred.emit)
    
    def _clear_chart_area(self):
        """Clear all widgets from chart area."""
        # Clear layout
        while self._chart_layout.count():
            child = self._chart_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Clear active charts
        self._active_charts.clear()
    
    def _on_layout_changed(self, layout_key: str):
        """Handle layout change."""
        if layout_key == self._current_layout:
            return
        
        # Setup new layout
        if layout_key == 'single':
            self._setup_single_chart_layout()
        elif layout_key == 'dual_horizontal':
            self._setup_dual_horizontal_layout()
        elif layout_key == 'dual_vertical':
            self._setup_dual_vertical_layout()
        elif layout_key == 'quad':
            self._setup_quad_layout()
        elif layout_key == 'multi_timeframe':
            self._setup_multi_timeframe_layout()
        
        # Update charts with current data
        self._update_all_charts()
        
        self.layout_changed.emit(layout_key)
    
    def _on_chart_type_changed(self, chart_type: str):
        """Handle chart type change for main chart."""
        # This would update the primary chart type
        # Implementation depends on specific requirements
        self.chart_type_changed.emit(chart_type)
    
    def _on_sync_charts_toggled(self, sync: bool):
        """Handle chart synchronization toggle."""
        # Enable/disable chart synchronization
        # This would sync zoom, pan, and timeframe changes across charts
        pass
    
    def _on_chart_timeframe_changed(self, timeframe: str):
        """Handle timeframe change from individual charts."""
        if self._sync_charts_cb.isChecked():
            # Sync timeframe across all charts
            sender = self.sender()
            for chart in self._active_charts.values():
                if chart != sender and hasattr(chart, '_timeframe_combo'):
                    chart._timeframe_combo.setCurrentText(timeframe)
    
    def _on_chart_indicator_toggled(self, indicator: str, visible: bool):
        """Handle indicator toggle from individual charts."""
        # Update indicator checkbox state
        if indicator in self._indicator_checkboxes:
            self._indicator_checkboxes[indicator].setChecked(visible)
    
    def _on_chart_zoom_changed(self, x_range: float, y_range: float):
        """Handle zoom change from individual charts."""
        if self._sync_charts_cb.isChecked():
            # Sync zoom across all charts
            # Implementation would depend on specific chart types
            pass
    
    def _toggle_indicator(self, indicator: str, enabled: bool):
        """Toggle technical indicator on all charts."""
        for chart in self._active_charts.values():
            if hasattr(chart, 'toggle_indicator'):
                chart.toggle_indicator(indicator, enabled)
            elif hasattr(chart, '_chart_widgets'):  # Multi-timeframe widget
                for sub_chart in chart._chart_widgets.values():
                    sub_chart.toggle_indicator(indicator, enabled)
    
    def _export_charts(self):
        """Export all active charts."""
        for chart_id, chart in self._active_charts.items():
            filename = f"chart_{chart_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            if hasattr(chart, 'export_chart'):
                chart.export_chart(filename)
                self.data_exported.emit(chart_id, filename)
    
    def update_data(self, data: Dict[str, Any]):
        """Update dashboard with new data."""
        self._current_data = data.copy()
        self._update_all_charts()
    
    def _update_all_charts(self):
        """Update all active charts with current data."""
        if not self._current_data:
            return
        
        # Get main market data
        market_data = self._current_data.get('market_data')
        if market_data is None or market_data.empty:
            return
        
        # Update charts based on layout
        if self._current_layout == 'multi_timeframe':
            # Multi-timeframe needs different data for each timeframe
            timeframe_data = self._prepare_multi_timeframe_data(market_data)
            if 'multi_timeframe' in self._active_charts:
                self._active_charts['multi_timeframe'].update_data(timeframe_data)
        else:
            # Update all charts with the same data
            for chart in self._active_charts.values():
                if hasattr(chart, 'update_data'):
                    chart.update_data(market_data)
        
        # Add indicators if available
        self._update_indicators()
    
    def _prepare_multi_timeframe_data(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare data for different timeframes."""
        timeframe_data = {}
        
        # Resample data for different timeframes
        if isinstance(data.index, pd.DatetimeIndex):
            try:
                # Daily data (original)
                timeframe_data['1D'] = data
                
                # Weekly data
                weekly = data.resample('W').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                timeframe_data['1W'] = weekly
                
                # Monthly data
                monthly = data.resample('ME').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                timeframe_data['1M'] = monthly
                
            except Exception:
                # If resampling fails, use original data for all timeframes
                timeframe_data = {'1D': data, '1W': data, '1M': data}
        else:
            # Non-datetime index, use original data
            timeframe_data = {'1D': data, '1W': data, '1M': data}
        
        return timeframe_data
    
    def _update_indicators(self):
        """Update technical indicators on charts."""
        if not self._current_data:
            return
        
        # Get indicator data from current data
        for indicator, checkbox in self._indicator_checkboxes.items():
            if checkbox.isChecked():
                indicator_data = self._current_data.get(f'{indicator.lower()}_data')
                if indicator_data is not None:
                    self._add_indicator_to_charts(indicator, indicator_data)
    
    def _add_indicator_to_charts(self, indicator: str, data: pd.Series):
        """Add indicator to all compatible charts."""
        for chart in self._active_charts.values():
            if hasattr(chart, 'add_indicator'):
                chart.add_indicator(indicator, data)
            elif hasattr(chart, '_chart_widgets'):  # Multi-timeframe widget
                for sub_chart in chart._chart_widgets.values():
                    if hasattr(sub_chart, 'add_indicator'):
                        sub_chart.add_indicator(indicator, data)
    
    def clear_data(self):
        """Clear data from all charts."""
        for chart in self._active_charts.values():
            if hasattr(chart, 'clear_data'):
                chart.clear_data()
        
        self._current_data.clear()
    
    def get_current_layout(self) -> str:
        """Get current dashboard layout."""
        return self._current_layout
    
    def get_active_charts(self) -> Dict[str, FinancialChartWidget]:
        """Get all active chart widgets."""
        return self._active_charts.copy()
    
    def set_chart_theme(self, theme: str):
        """Set theme for all charts."""
        # This would update chart colors based on theme
        colors = self._get_theme_colors(theme)
        
        for chart in self._active_charts.values():
            if hasattr(chart, '_colors'):
                chart._colors.update(colors)
                chart._apply_chart_styling()
    
    def _get_theme_colors(self, theme: str) -> Dict[str, str]:
        """Get color scheme for theme."""
        if theme == 'dark':
            return {
                'background': '#2b2b2b',
                'grid': '#404040',
                'text': '#ffffff',
                'up': '#00ff00',
                'down': '#ff4444',
                'volume': '#4080ff',
                'ma': '#ff8040',
                'signal': '#8040ff'
            }
        else:  # light theme
            return {
                'background': '#ffffff',
                'grid': '#e0e0e0',
                'text': '#000000',
                'up': '#00aa00',
                'down': '#cc0000',
                'volume': '#0066cc',
                'ma': '#cc6600',
                'signal': '#6600cc'
            }


class IndicatorPanel(ContainerWidget):
    """Panel for managing technical indicators."""
    
    # Signals
    indicator_added = pyqtSignal(str, dict)  # indicator_name, parameters
    indicator_removed = pyqtSignal(str)
    indicator_updated = pyqtSignal(str, dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._available_indicators = {
            'SMA': {'name': 'Simple Moving Average', 'params': {'period': 20}},
            'EMA': {'name': 'Exponential Moving Average', 'params': {'period': 20}},
            'Bollinger Bands': {'name': 'Bollinger Bands', 'params': {'period': 20, 'std': 2}},
            'RSI': {'name': 'Relative Strength Index', 'params': {'period': 14}},
            'MACD': {'name': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
            'Stochastic': {'name': 'Stochastic Oscillator', 'params': {'k_period': 14, 'd_period': 3}},
            'Volume': {'name': 'Volume', 'params': {}}
        }
        self._active_indicators = {}
        
        self._setup_panel()
    
    def _setup_panel(self):
        """Setup indicator management panel."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title_label = QLabel("Technical Indicators")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # Available indicators
        available_group = QGroupBox("Available Indicators")
        available_layout = QVBoxLayout()
        available_group.setLayout(available_layout)
        layout.addWidget(available_group)
        
        # Create checkboxes for each indicator
        self._indicator_checkboxes = {}
        for indicator_key, indicator_info in self._available_indicators.items():
            cb = QCheckBox(indicator_info['name'])
            cb.toggled.connect(lambda checked, key=indicator_key: self._toggle_indicator(key, checked))
            available_layout.addWidget(cb)
            self._indicator_checkboxes[indicator_key] = cb
        
        # Active indicators
        active_group = QGroupBox("Active Indicators")
        active_layout = QVBoxLayout()
        active_group.setLayout(active_layout)
        layout.addWidget(active_group)
        
        # Scroll area for active indicators
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self._active_layout = QVBoxLayout()
        scroll_widget.setLayout(self._active_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        active_layout.addWidget(scroll_area)
        
        layout.addStretch()
    
    def _toggle_indicator(self, indicator_key: str, enabled: bool):
        """Toggle indicator on/off."""
        if enabled:
            self._add_indicator(indicator_key)
        else:
            self._remove_indicator(indicator_key)
    
    def _add_indicator(self, indicator_key: str):
        """Add indicator to active list."""
        if indicator_key in self._active_indicators:
            return
        
        indicator_info = self._available_indicators[indicator_key]
        params = indicator_info['params'].copy()
        
        # Create indicator control widget
        control_widget = self._create_indicator_control(indicator_key, indicator_info['name'], params)
        self._active_layout.addWidget(control_widget)
        
        self._active_indicators[indicator_key] = {
            'widget': control_widget,
            'params': params
        }
        
        self.indicator_added.emit(indicator_key, params)
    
    def _remove_indicator(self, indicator_key: str):
        """Remove indicator from active list."""
        if indicator_key not in self._active_indicators:
            return
        
        # Remove widget
        widget = self._active_indicators[indicator_key]['widget']
        self._active_layout.removeWidget(widget)
        widget.deleteLater()
        
        del self._active_indicators[indicator_key]
        
        self.indicator_removed.emit(indicator_key)
    
    def _create_indicator_control(self, indicator_key: str, name: str, params: Dict[str, Any]) -> QWidget:
        """Create control widget for an indicator."""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Box)
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Header with name and remove button
        header_layout = QHBoxLayout()
        layout.addLayout(header_layout)
        
        name_label = QLabel(name)
        name_label.setFont(QFont("Arial", 10, QFont.Bold))
        header_layout.addWidget(name_label)
        
        header_layout.addStretch()
        
        remove_btn = QPushButton("×")
        remove_btn.setMaximumSize(20, 20)
        remove_btn.clicked.connect(lambda: self._remove_indicator_by_button(indicator_key))
        header_layout.addWidget(remove_btn)
        
        # Parameter controls
        param_layout = QGridLayout()
        layout.addLayout(param_layout)
        
        row = 0
        for param_name, param_value in params.items():
            param_layout.addWidget(QLabel(f"{param_name.title()}:"), row, 0)
            
            if isinstance(param_value, int):
                spin_box = QSpinBox()
                spin_box.setRange(1, 200)
                spin_box.setValue(param_value)
                spin_box.valueChanged.connect(
                    lambda value, key=indicator_key, param=param_name: 
                    self._update_parameter(key, param, value)
                )
                param_layout.addWidget(spin_box, row, 1)
            elif isinstance(param_value, float):
                spin_box = QDoubleSpinBox()
                spin_box.setRange(0.1, 10.0)
                spin_box.setSingleStep(0.1)
                spin_box.setValue(param_value)
                spin_box.valueChanged.connect(
                    lambda value, key=indicator_key, param=param_name: 
                    self._update_parameter(key, param, value)
                )
                param_layout.addWidget(spin_box, row, 1)
            
            row += 1
        
        return widget
    
    def _remove_indicator_by_button(self, indicator_key: str):
        """Remove indicator via button click."""
        # Uncheck the checkbox
        if indicator_key in self._indicator_checkboxes:
            self._indicator_checkboxes[indicator_key].setChecked(False)
    
    def _update_parameter(self, indicator_key: str, param_name: str, value):
        """Update indicator parameter."""
        if indicator_key in self._active_indicators:
            self._active_indicators[indicator_key]['params'][param_name] = value
            params = self._active_indicators[indicator_key]['params']
            self.indicator_updated.emit(indicator_key, params)
    
    def get_active_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Get all active indicators and their parameters."""
        return {key: info['params'] for key, info in self._active_indicators.items()}
    
    def clear_indicators(self):
        """Clear all active indicators."""
        # Uncheck all checkboxes
        for cb in self._indicator_checkboxes.values():
            cb.setChecked(False)