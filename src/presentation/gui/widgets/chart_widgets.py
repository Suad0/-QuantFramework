"""
Advanced chart widgets using PyQtGraph for interactive financial visualizations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, 
    QCheckBox, QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox,
    QSplitter, QTabWidget, QScrollArea, QFrame
)
from PyQt5.QtCore import pyqtSignal, QTimer, Qt, QThread, QObject
from PyQt5.QtGui import QFont, QPen, QBrush, QColor

import pyqtgraph as pg
from pyqtgraph import PlotWidget, GraphicsLayoutWidget, ViewBox

from .base_widget import BaseWidget, ContainerWidget


class FinancialChartWidget(BaseWidget):
    """Base class for financial chart widgets with common functionality."""
    
    # Signals
    timeframe_changed = pyqtSignal(str)
    indicator_toggled = pyqtSignal(str, bool)
    zoom_changed = pyqtSignal(float, float)
    selection_changed = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._timeframe = '1D'
        self._indicators = {}
        self._plot_items = {}
        self._crosshair_enabled = True
        self._auto_range = True
        
        # Chart styling
        self._colors = {
            'background': '#ffffff',
            'grid': '#e0e0e0',
            'text': '#000000',
            'up': '#00ff00',
            'down': '#ff0000',
            'volume': '#0080ff',
            'ma': '#ff8000',
            'signal': '#8000ff'
        }
        
        self._setup_chart()
    
    def _setup_chart(self):
        """Setup the basic chart structure."""
        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create control panel
        self._create_control_panel()
        layout.addWidget(self._control_panel)
        
        # Create chart area
        self._create_chart_area()
        layout.addWidget(self._chart_widget)
        
        # Setup crosshair
        self._setup_crosshair()
        
        # Apply initial styling
        self._apply_chart_styling()
    
    def _create_control_panel(self):
        """Create chart control panel."""
        self._control_panel = QFrame()
        self._control_panel.setMaximumHeight(50)
        control_layout = QHBoxLayout()
        self._control_panel.setLayout(control_layout)
        
        # Timeframe selector
        control_layout.addWidget(QLabel("Timeframe:"))
        self._timeframe_combo = QComboBox()
        self._timeframe_combo.addItems(['1m', '5m', '15m', '30m', '1H', '4H', '1D', '1W', '1M'])
        self._timeframe_combo.setCurrentText(self._timeframe)
        self._timeframe_combo.currentTextChanged.connect(self._on_timeframe_changed)
        control_layout.addWidget(self._timeframe_combo)
        
        # Auto-range checkbox
        self._auto_range_cb = QCheckBox("Auto Range")
        self._auto_range_cb.setChecked(self._auto_range)
        self._auto_range_cb.toggled.connect(self._on_auto_range_toggled)
        control_layout.addWidget(self._auto_range_cb)
        
        # Crosshair checkbox
        self._crosshair_cb = QCheckBox("Crosshair")
        self._crosshair_cb.setChecked(self._crosshair_enabled)
        self._crosshair_cb.toggled.connect(self._on_crosshair_toggled)
        control_layout.addWidget(self._crosshair_cb)
        
        # Reset zoom button
        reset_btn = QPushButton("Reset Zoom")
        reset_btn.clicked.connect(self._reset_zoom)
        control_layout.addWidget(reset_btn)
        
        control_layout.addStretch()
    
    def _create_chart_area(self):
        """Create the main chart area. Override in subclasses."""
        self._chart_widget = PlotWidget()
        self._plot = self._chart_widget.getPlotItem()
        
        # Enable mouse interaction
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.enableAutoRange(x=True, y=True)
        
        # Setup grid
        self._plot.showGrid(x=True, y=True, alpha=0.3)
    
    def _setup_crosshair(self):
        """Setup crosshair functionality."""
        self._crosshair_v = pg.InfiniteLine(angle=90, movable=False)
        self._crosshair_h = pg.InfiniteLine(angle=0, movable=False)
        
        self._crosshair_v.setPen(pg.mkPen(color='gray', style=Qt.DashLine))
        self._crosshair_h.setPen(pg.mkPen(color='gray', style=Qt.DashLine))
        
        if self._crosshair_enabled:
            self._plot.addItem(self._crosshair_v, ignoreBounds=True)
            self._plot.addItem(self._crosshair_h, ignoreBounds=True)
        
        # Connect mouse move event
        self._chart_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
    
    def _apply_chart_styling(self):
        """Apply chart styling based on theme."""
        # Set background
        self._chart_widget.setBackground(self._colors['background'])
        
        # Style axes
        self._plot.getAxis('left').setPen(self._colors['text'])
        self._plot.getAxis('bottom').setPen(self._colors['text'])
        
        # Style labels
        font = QFont()
        font.setPixelSize(10)
        self._plot.getAxis('left').setStyle(tickFont=font)
        self._plot.getAxis('bottom').setStyle(tickFont=font)
    
    def _on_timeframe_changed(self, timeframe: str):
        """Handle timeframe change."""
        self._timeframe = timeframe
        self.timeframe_changed.emit(timeframe)
        if self._data is not None:
            self._update_chart_data()
    
    def _on_auto_range_toggled(self, enabled: bool):
        """Handle auto-range toggle."""
        self._auto_range = enabled
        self._plot.enableAutoRange(x=enabled, y=enabled)
    
    def _on_crosshair_toggled(self, enabled: bool):
        """Handle crosshair toggle."""
        self._crosshair_enabled = enabled
        if enabled:
            self._plot.addItem(self._crosshair_v, ignoreBounds=True)
            self._plot.addItem(self._crosshair_h, ignoreBounds=True)
        else:
            self._plot.removeItem(self._crosshair_v)
            self._plot.removeItem(self._crosshair_h)
    
    def _on_mouse_moved(self, pos):
        """Handle mouse movement for crosshair."""
        if self._crosshair_enabled and self._chart_widget.sceneBoundingRect().contains(pos):
            mouse_point = self._plot.vb.mapSceneToView(pos)
            self._crosshair_v.setPos(mouse_point.x())
            self._crosshair_h.setPos(mouse_point.y())
    
    def _reset_zoom(self):
        """Reset chart zoom to show all data."""
        self._plot.autoRange()
    
    def update_data(self, data: pd.DataFrame):
        """Update chart with new data."""
        if not self.validate_data(data):
            return
        
        self._data = data.copy()
        self._update_chart_data()
    
    def _update_chart_data(self):
        """Update chart visualization. Override in subclasses."""
        pass
    
    def validate_data(self, data: Any) -> bool:
        """Validate input data."""
        if not isinstance(data, pd.DataFrame):
            return False
        
        required_columns = self._get_required_columns()
        return all(col in data.columns for col in required_columns)
    
    def _get_required_columns(self) -> List[str]:
        """Get required columns for this chart type. Override in subclasses."""
        return []
    
    def add_indicator(self, name: str, indicator_data: pd.Series, color: str = None, style: str = 'line'):
        """Add technical indicator to chart."""
        if color is None:
            color = self._colors.get(name, '#808080')
        
        pen = pg.mkPen(color=color, width=2)
        
        if style == 'line':
            curve = self._plot.plot(pen=pen, name=name)
            curve.setData(indicator_data.index.values, indicator_data.values)
        elif style == 'scatter':
            scatter = self._plot.plot(pen=None, symbol='o', symbolBrush=color, name=name)
            scatter.setData(indicator_data.index.values, indicator_data.values)
        
        self._indicators[name] = {
            'data': indicator_data,
            'color': color,
            'style': style,
            'visible': True
        }
    
    def remove_indicator(self, name: str):
        """Remove indicator from chart."""
        if name in self._indicators:
            # Remove from plot
            for item in self._plot.listDataItems():
                if hasattr(item, 'name') and item.name == name:
                    self._plot.removeItem(item)
            
            del self._indicators[name]
    
    def toggle_indicator(self, name: str, visible: bool = None):
        """Toggle indicator visibility."""
        if name not in self._indicators:
            return
        
        if visible is None:
            visible = not self._indicators[name]['visible']
        
        self._indicators[name]['visible'] = visible
        
        # Update plot item visibility
        for item in self._plot.listDataItems():
            if hasattr(item, 'name') and item.name == name:
                item.setVisible(visible)
        
        self.indicator_toggled.emit(name, visible)
    
    def clear_data(self):
        """Clear all chart data."""
        self._plot.clear()
        self._data = None
        self._indicators.clear()
    
    def export_chart(self, filename: str, width: int = 1920, height: int = 1080):
        """Export chart as image."""
        exporter = pg.exporters.ImageExporter(self._plot)
        exporter.parameters()['width'] = width
        exporter.parameters()['height'] = height
        exporter.export(filename)


class CandlestickChartWidget(FinancialChartWidget):
    """Candlestick chart widget for OHLC data."""
    
    def __init__(self, parent=None):
        self._volume_plot = None
        self._show_volume = True
        super().__init__(parent)
    
    def _create_chart_area(self):
        """Create chart area with price and volume plots."""
        # Create graphics layout widget
        self._chart_widget = GraphicsLayoutWidget()
        
        # Create price plot
        self._price_plot = self._chart_widget.addPlot(row=0, col=0)
        self._price_plot.setLabel('left', 'Price')
        self._price_plot.showGrid(x=True, y=True, alpha=0.3)
        self._price_plot.setMouseEnabled(x=True, y=True)
        
        # Create volume plot
        self._volume_plot = self._chart_widget.addPlot(row=1, col=0)
        self._volume_plot.setLabel('left', 'Volume')
        self._volume_plot.setLabel('bottom', 'Time')
        self._volume_plot.showGrid(x=True, y=True, alpha=0.3)
        self._volume_plot.setMouseEnabled(x=True, y=True)
        
        # Link x-axes
        self._volume_plot.setXLink(self._price_plot)
        
        # Set height ratios (price plot should be larger)
        self._chart_widget.ci.layout.setRowStretchFactor(0, 3)
        self._chart_widget.ci.layout.setRowStretchFactor(1, 1)
        
        # Set main plot reference
        self._plot = self._price_plot
    
    def _create_control_panel(self):
        """Create enhanced control panel for candlestick chart."""
        super()._create_control_panel()
        
        # Add volume toggle
        self._volume_cb = QCheckBox("Show Volume")
        self._volume_cb.setChecked(self._show_volume)
        self._volume_cb.toggled.connect(self._on_volume_toggled)
        self._control_panel.layout().insertWidget(-1, self._volume_cb)
    
    def _setup_crosshair(self):
        """Setup crosshair for both price and volume plots."""
        # Price plot crosshair
        self._crosshair_v_price = pg.InfiniteLine(angle=90, movable=False)
        self._crosshair_h_price = pg.InfiniteLine(angle=0, movable=False)
        
        # Volume plot crosshair
        self._crosshair_v_volume = pg.InfiniteLine(angle=90, movable=False)
        self._crosshair_h_volume = pg.InfiniteLine(angle=0, movable=False)
        
        # Style crosshairs
        pen = pg.mkPen(color='gray', style=Qt.DashLine)
        self._crosshair_v_price.setPen(pen)
        self._crosshair_h_price.setPen(pen)
        self._crosshair_v_volume.setPen(pen)
        self._crosshair_h_volume.setPen(pen)
        
        if self._crosshair_enabled:
            self._price_plot.addItem(self._crosshair_v_price, ignoreBounds=True)
            self._price_plot.addItem(self._crosshair_h_price, ignoreBounds=True)
            self._volume_plot.addItem(self._crosshair_v_volume, ignoreBounds=True)
            self._volume_plot.addItem(self._crosshair_h_volume, ignoreBounds=True)
        
        # Connect mouse move event
        self._chart_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
    
    def _on_mouse_moved(self, pos):
        """Handle mouse movement for crosshair in both plots."""
        if not self._crosshair_enabled:
            return
        
        # Check if mouse is over price plot
        if self._price_plot.sceneBoundingRect().contains(pos):
            mouse_point = self._price_plot.vb.mapSceneToView(pos)
            self._crosshair_v_price.setPos(mouse_point.x())
            self._crosshair_h_price.setPos(mouse_point.y())
            self._crosshair_v_volume.setPos(mouse_point.x())
        
        # Check if mouse is over volume plot
        elif self._volume_plot.sceneBoundingRect().contains(pos):
            mouse_point = self._volume_plot.vb.mapSceneToView(pos)
            self._crosshair_v_price.setPos(mouse_point.x())
            self._crosshair_v_volume.setPos(mouse_point.x())
            self._crosshair_h_volume.setPos(mouse_point.y())
    
    def _on_volume_toggled(self, show: bool):
        """Handle volume plot toggle."""
        self._show_volume = show
        if show:
            self._volume_plot.show()
        else:
            self._volume_plot.hide()
    
    def _get_required_columns(self) -> List[str]:
        """Required columns for candlestick chart."""
        return ['open', 'high', 'low', 'close']
    
    def _update_chart_data(self):
        """Update candlestick chart with OHLC data."""
        if self._data is None or self._data.empty:
            return
        
        # Clear existing data
        self._price_plot.clear()
        if self._volume_plot:
            self._volume_plot.clear()
        
        # Prepare data
        data = self._data.copy()
        
        # Convert datetime index to numeric for plotting
        if isinstance(data.index, pd.DatetimeIndex):
            x_data = np.arange(len(data))
            x_labels = data.index
        else:
            x_data = np.arange(len(data))
            x_labels = data.index
        
        # Create candlestick items
        self._create_candlesticks(x_data, data)
        
        # Add volume bars if available and enabled
        if 'volume' in data.columns and self._show_volume and self._volume_plot:
            self._create_volume_bars(x_data, data['volume'])
        
        # Update indicators
        self._update_indicators(x_data)
        
        # Set x-axis labels
        self._setup_time_axis(x_data, x_labels)
    
    def _create_candlesticks(self, x_data: np.ndarray, data: pd.DataFrame):
        """Create candlestick visualization."""
        opens = data['open'].values
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # Create candlestick items
        for i, (x, o, h, l, c) in enumerate(zip(x_data, opens, highs, lows, closes)):
            # Determine color
            color = self._colors['up'] if c >= o else self._colors['down']
            
            # Create high-low line
            hl_line = pg.PlotDataItem([x, x], [l, h], pen=pg.mkPen(color, width=1))
            self._price_plot.addItem(hl_line)
            
            # Create body rectangle
            body_height = abs(c - o)
            body_bottom = min(o, c)
            
            if body_height > 0:
                # Create filled rectangle for body
                from PyQt5.QtWidgets import QGraphicsRectItem
                body_rect = QGraphicsRectItem(x - 0.3, body_bottom, 0.6, body_height)
                body_rect.setBrush(pg.mkBrush(color))
                body_rect.setPen(pg.mkPen(color))
                self._price_plot.addItem(body_rect)
            else:
                # Doji - create horizontal line
                doji_line = pg.PlotDataItem([x - 0.3, x + 0.3], [c, c], pen=pg.mkPen(color, width=2))
                self._price_plot.addItem(doji_line)
    
    def _create_volume_bars(self, x_data: np.ndarray, volume_data: pd.Series):
        """Create volume bar chart."""
        # Create volume bars
        volume_bars = pg.BarGraphItem(
            x=x_data, 
            height=volume_data.values, 
            width=0.8, 
            brush=self._colors['volume']
        )
        self._volume_plot.addItem(volume_bars)
    
    def _update_indicators(self, x_data: np.ndarray):
        """Update technical indicators on the chart."""
        for name, indicator_info in self._indicators.items():
            if not indicator_info['visible']:
                continue
            
            indicator_data = indicator_info['data']
            color = indicator_info['color']
            style = indicator_info['style']
            
            # Align indicator data with chart data
            aligned_data = indicator_data.reindex(self._data.index).fillna(np.nan)
            
            pen = pg.mkPen(color=color, width=2)
            
            if style == 'line':
                curve = self._price_plot.plot(x_data, aligned_data.values, pen=pen, name=name)
            elif style == 'scatter':
                scatter = self._price_plot.plot(
                    x_data, aligned_data.values, 
                    pen=None, symbol='o', symbolBrush=color, name=name
                )
    
    def _setup_time_axis(self, x_data: np.ndarray, x_labels):
        """Setup time-based x-axis labels."""
        # Create custom tick labels for time axis
        if len(x_labels) > 0:
            # Sample labels to avoid overcrowding
            step = max(1, len(x_labels) // 10)
            tick_positions = x_data[::step]
            tick_labels = [str(label) for label in x_labels[::step]]
            
            # Set custom ticks
            ticks = list(zip(tick_positions, tick_labels))
            self._price_plot.getAxis('bottom').setTicks([ticks])
            if self._volume_plot:
                self._volume_plot.getAxis('bottom').setTicks([ticks])


class LineChartWidget(FinancialChartWidget):
    """Multi-line chart widget for time series data."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._lines = {}
        self._legend = None
    
    def _create_chart_area(self):
        """Create simple line chart area."""
        super()._create_chart_area()
        
        # Add legend
        self._legend = self._plot.addLegend()
    
    def _get_required_columns(self) -> List[str]:
        """No specific required columns for line chart."""
        return []
    
    def _update_chart_data(self):
        """Update line chart with time series data."""
        if self._data is None or self._data.empty:
            return
        
        # Clear existing lines
        self._plot.clear()
        self._lines.clear()
        
        # Re-add legend
        self._legend = self._plot.addLegend()
        
        # Convert datetime index to numeric for plotting
        if isinstance(self._data.index, pd.DatetimeIndex):
            x_data = np.arange(len(self._data))
        else:
            x_data = self._data.index.values
        
        # Plot each numeric column as a line
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, column in enumerate(self._data.select_dtypes(include=[np.number]).columns):
            color = colors[i % len(colors)]
            pen = pg.mkPen(color=color, width=2)
            
            line = self._plot.plot(
                x_data, 
                self._data[column].values, 
                pen=pen, 
                name=column
            )
            self._lines[column] = line
        
        # Setup time axis if needed
        if isinstance(self._data.index, pd.DatetimeIndex):
            self._setup_time_axis(x_data, self._data.index)
    
    def _setup_time_axis(self, x_data: np.ndarray, x_labels):
        """Setup time-based x-axis labels."""
        if len(x_labels) > 0:
            step = max(1, len(x_labels) // 10)
            tick_positions = x_data[::step]
            tick_labels = [label.strftime('%Y-%m-%d') for label in x_labels[::step]]
            
            ticks = list(zip(tick_positions, tick_labels))
            self._plot.getAxis('bottom').setTicks([ticks])
    
    def add_line(self, name: str, data: pd.Series, color: str = None):
        """Add a new line to the chart."""
        if color is None:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            color = colors[len(self._lines) % len(colors)]
        
        pen = pg.mkPen(color=color, width=2)
        
        # Align data with chart index
        if self._data is not None:
            aligned_data = data.reindex(self._data.index).fillna(np.nan)
            x_data = np.arange(len(self._data))
        else:
            aligned_data = data
            x_data = np.arange(len(data))
        
        line = self._plot.plot(x_data, aligned_data.values, pen=pen, name=name)
        self._lines[name] = line
    
    def remove_line(self, name: str):
        """Remove a line from the chart."""
        if name in self._lines:
            self._plot.removeItem(self._lines[name])
            del self._lines[name]
    
    def toggle_line(self, name: str, visible: bool = None):
        """Toggle line visibility."""
        if name not in self._lines:
            return
        
        line = self._lines[name]
        if visible is None:
            visible = not line.isVisible()
        
        line.setVisible(visible)


class MultiTimeframeWidget(ContainerWidget):
    """Widget for displaying multiple timeframes simultaneously."""
    
    def __init__(self, parent=None):
        self._timeframes = ['1D', '1W', '1M']
        self._chart_widgets = {}
        self._data_cache = {}
        super().__init__(parent)
    
    def _setup_widget(self):
        """Setup multi-timeframe layout."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create tab widget for different timeframes
        self._tab_widget = QTabWidget()
        layout.addWidget(self._tab_widget)
        
        # Create charts for each timeframe
        for timeframe in self._timeframes:
            chart = CandlestickChartWidget()
            chart.timeframe_changed.connect(self._on_timeframe_changed)
            
            self._tab_widget.addTab(chart, timeframe)
            self._chart_widgets[timeframe] = chart
    
    def _on_timeframe_changed(self, timeframe: str):
        """Handle timeframe change from individual charts."""
        sender = self.sender()
        if sender in self._chart_widgets.values():
            # Update data for the specific chart
            if timeframe in self._data_cache:
                sender.update_data(self._data_cache[timeframe])
    
    def update_data(self, data_dict: Dict[str, pd.DataFrame]):
        """Update data for all timeframes."""
        self._data_cache = data_dict.copy()
        
        for timeframe, data in data_dict.items():
            if timeframe in self._chart_widgets:
                self._chart_widgets[timeframe].update_data(data)
    
    def add_timeframe(self, timeframe: str):
        """Add a new timeframe tab."""
        if timeframe not in self._chart_widgets:
            chart = CandlestickChartWidget()
            chart.timeframe_changed.connect(self._on_timeframe_changed)
            
            self._tab_widget.addTab(chart, timeframe)
            self._chart_widgets[timeframe] = chart
            self._timeframes.append(timeframe)
    
    def remove_timeframe(self, timeframe: str):
        """Remove a timeframe tab."""
        if timeframe in self._chart_widgets:
            chart = self._chart_widgets[timeframe]
            index = self._tab_widget.indexOf(chart)
            self._tab_widget.removeTab(index)
            
            del self._chart_widgets[timeframe]
            self._timeframes.remove(timeframe)
    
    def get_current_chart(self) -> Optional[FinancialChartWidget]:
        """Get currently active chart widget."""
        current_widget = self._tab_widget.currentWidget()
        return current_widget if isinstance(current_widget, FinancialChartWidget) else None
    
    def clear_data(self):
        """Clear data from all charts."""
        for chart in self._chart_widgets.values():
            chart.clear_data()
        self._data_cache.clear()


class RealTimeChartWidget(FinancialChartWidget):
    """Real-time streaming chart widget."""
    
    # Signals
    data_stream_started = pyqtSignal()
    data_stream_stopped = pyqtSignal()
    new_data_received = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._streaming = False
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_streaming_data)
        self._update_interval = 1000  # 1 second
        self._max_points = 1000  # Maximum points to display
        self._streaming_data = pd.DataFrame()
        
        self._setup_streaming_controls()
    
    def _setup_streaming_controls(self):
        """Add streaming-specific controls."""
        # Add streaming controls to control panel
        control_layout = self._control_panel.layout()
        
        # Start/Stop streaming button
        self._stream_btn = QPushButton("Start Stream")
        self._stream_btn.clicked.connect(self._toggle_streaming)
        control_layout.insertWidget(-1, self._stream_btn)
        
        # Update interval control
        control_layout.insertWidget(-1, QLabel("Update (ms):"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(100, 10000)
        self._interval_spin.setValue(self._update_interval)
        self._interval_spin.valueChanged.connect(self._on_interval_changed)
        control_layout.insertWidget(-1, self._interval_spin)
        
        # Max points control
        control_layout.insertWidget(-1, QLabel("Max Points:"))
        self._max_points_spin = QSpinBox()
        self._max_points_spin.setRange(100, 10000)
        self._max_points_spin.setValue(self._max_points)
        self._max_points_spin.valueChanged.connect(self._on_max_points_changed)
        control_layout.insertWidget(-1, self._max_points_spin)
    
    def _toggle_streaming(self):
        """Toggle streaming on/off."""
        if self._streaming:
            self.stop_streaming()
        else:
            self.start_streaming()
    
    def start_streaming(self):
        """Start real-time data streaming."""
        if not self._streaming:
            self._streaming = True
            self._update_timer.start(self._update_interval)
            self._stream_btn.setText("Stop Stream")
            self.data_stream_started.emit()
    
    def stop_streaming(self):
        """Stop real-time data streaming."""
        if self._streaming:
            self._streaming = False
            self._update_timer.stop()
            self._stream_btn.setText("Start Stream")
            self.data_stream_stopped.emit()
    
    def _on_interval_changed(self, interval: int):
        """Handle update interval change."""
        self._update_interval = interval
        if self._streaming:
            self._update_timer.setInterval(interval)
    
    def _on_max_points_changed(self, max_points: int):
        """Handle max points change."""
        self._max_points = max_points
        self._trim_data()
    
    def set_title(self, title: str):
        """Set chart title."""
        if hasattr(self, '_plot_widget'):
            self._plot_widget.setTitle(title)
    
    def clear_data(self):
        """Clear all chart data."""
        if hasattr(self, '_plot_widget'):
            self._plot_widget.clear()
        self._streaming_data = pd.DataFrame()
    
    def add_series(self, name: str, y_data: np.ndarray, x_data: Optional[np.ndarray] = None, color: str = 'blue'):
        """Add a data series to the chart."""
        if not hasattr(self, '_plot_widget'):
            return
        
        if x_data is None:
            x_data = np.arange(len(y_data))
        
        # Convert datetime index to numeric if needed
        if hasattr(x_data, 'to_pydatetime'):
            x_data = np.array([x.timestamp() for x in x_data.to_pydatetime()])
        elif isinstance(x_data, pd.DatetimeIndex):
            x_data = np.array([x.timestamp() for x in x_data])
        
        pen = pg.mkPen(color=color, width=2)
        self._plot_widget.plot(x_data, y_data, pen=pen, name=name)
    
    def update_data(self, y_data: np.ndarray, x_data: Optional[np.ndarray] = None):
        """Update chart with new data."""
        if not hasattr(self, '_plot_widget'):
            return
        
        self.clear_data()
        if x_data is not None:
            self.add_series("Data", y_data, x_data)
        else:
            self.add_series("Data", y_data)
    
    def _update_streaming_data(self):
        """Update chart with new streaming data."""
        # This would typically fetch new data from a data source
        # For now, we'll simulate new data
        new_data = self._simulate_new_data()
        
        if new_data is not None:
            self.add_streaming_data(new_data)
    
    def _simulate_new_data(self) -> Optional[pd.Series]:
        """Simulate new streaming data point."""
        if self._data is None or self._data.empty:
            return None
        
        # Get last data point and simulate next point
        last_row = self._data.iloc[-1]
        last_close = last_row['close']
        
        # Simple random walk simulation
        change = np.random.normal(0, last_close * 0.001)  # 0.1% volatility
        new_close = max(0.01, last_close + change)
        
        # Create new data point
        new_timestamp = pd.Timestamp.now()
        new_data = pd.Series({
            'open': last_close,
            'high': max(last_close, new_close),
            'low': min(last_close, new_close),
            'close': new_close,
            'volume': np.random.randint(1000, 10000)
        }, name=new_timestamp)
        
        return new_data
    
    def add_streaming_data(self, new_data: pd.Series):
        """Add new data point to streaming chart."""
        if self._streaming_data.empty:
            self._streaming_data = pd.DataFrame([new_data])
        else:
            self._streaming_data = pd.concat([self._streaming_data, pd.DataFrame([new_data])])
        
        # Trim data if necessary
        self._trim_data()
        
        # Update chart
        self._update_chart_data()
        
        # Emit signal
        self.new_data_received.emit(new_data)
    
    def _trim_data(self):
        """Trim data to maximum points."""
        if len(self._streaming_data) > self._max_points:
            self._streaming_data = self._streaming_data.tail(self._max_points)
    
    def update_data(self, data: pd.DataFrame):
        """Update initial data for streaming."""
        super().update_data(data)
        self._streaming_data = data.copy() if data is not None else pd.DataFrame()
    
    def _update_chart_data(self):
        """Update chart with streaming data."""
        if not self._streaming_data.empty:
            # Use streaming data instead of static data
            original_data = self._data
            self._data = self._streaming_data
            super()._update_chart_data()
            self._data = original_data
    
    def clear_data(self):
        """Clear all data including streaming buffer."""
        super().clear_data()
        self._streaming_data = pd.DataFrame()
        self.stop_streaming()