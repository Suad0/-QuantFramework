"""
Responsive layout management system for different screen sizes.
"""

from typing import Dict, Tuple, List
from PyQt5.QtCore import QObject, QRect, QSize, pyqtSignal
from PyQt5.QtWidgets import QWidget, QSplitter, QLayout, QApplication
from PyQt5.QtGui import QScreen
import math


class LayoutManager(QObject):
    """Manages responsive layouts for different screen sizes."""
    
    layout_changed = pyqtSignal(str)  # layout_type
    
    # Screen size breakpoints (width in pixels)
    BREAKPOINTS = {
        'small': 1024,    # Tablets and small laptops
        'medium': 1366,   # Standard laptops
        'large': 1920,    # Full HD monitors
        'xlarge': 2560    # 4K and ultrawide monitors
    }
    
    def __init__(self):
        super().__init__()
        self._current_layout = "medium"
        self._widgets = {}
        self._splitters = {}
        self._layout_configs = self._get_layout_configurations()
    
    def _get_layout_configurations(self) -> Dict[str, Dict]:
        """Get layout configurations for different screen sizes."""
        return {
            'small': {
                'main_splitter_orientation': 'vertical',
                'main_splitter_sizes': [300, 700],
                'control_panel_width': 280,
                'chart_columns': 2,
                'chart_rows': 3,
                'font_size': 9,
                'button_height': 32,
                'input_height': 28,
                'spacing': 4,
                'margins': (8, 8, 8, 8)
            },
            'medium': {
                'main_splitter_orientation': 'horizontal',
                'main_splitter_sizes': [400, 1000],
                'control_panel_width': 350,
                'chart_columns': 3,
                'chart_rows': 2,
                'font_size': 10,
                'button_height': 36,
                'input_height': 32,
                'spacing': 6,
                'margins': (10, 10, 10, 10)
            },
            'large': {
                'main_splitter_orientation': 'horizontal',
                'main_splitter_sizes': [450, 1470],
                'control_panel_width': 400,
                'chart_columns': 3,
                'chart_rows': 2,
                'font_size': 11,
                'button_height': 40,
                'input_height': 36,
                'spacing': 8,
                'margins': (12, 12, 12, 12)
            },
            'xlarge': {
                'main_splitter_orientation': 'horizontal',
                'main_splitter_sizes': [500, 2060],
                'control_panel_width': 450,
                'chart_columns': 4,
                'chart_rows': 2,
                'font_size': 12,
                'button_height': 44,
                'input_height': 40,
                'spacing': 10,
                'margins': (15, 15, 15, 15)
            }
        }
    
    def detect_screen_size(self) -> str:
        """Detect current screen size category."""
        app = QApplication.instance()
        if not app:
            return "medium"
        
        screen = app.primaryScreen()
        if not screen:
            return "medium"
        
        screen_size = screen.size()
        width = screen_size.width()
        
        if width <= self.BREAKPOINTS['small']:
            return 'small'
        elif width <= self.BREAKPOINTS['medium']:
            return 'medium'
        elif width <= self.BREAKPOINTS['large']:
            return 'large'
        else:
            return 'xlarge'
    
    def get_current_layout(self) -> str:
        """Get current layout type."""
        return self._current_layout
    
    def set_layout(self, layout_type: str):
        """Set the layout type."""
        if layout_type in self._layout_configs:
            self._current_layout = layout_type
            self.layout_changed.emit(layout_type)
    
    def auto_detect_layout(self):
        """Automatically detect and set appropriate layout."""
        detected_layout = self.detect_screen_size()
        self.set_layout(detected_layout)
    
    def get_layout_config(self, layout_type: str = None) -> Dict:
        """Get configuration for specified layout or current layout."""
        layout_type = layout_type or self._current_layout
        return self._layout_configs.get(layout_type, self._layout_configs['medium'])
    
    def register_widget(self, name: str, widget: QWidget):
        """Register a widget for layout management."""
        self._widgets[name] = widget
    
    def register_splitter(self, name: str, splitter: QSplitter):
        """Register a splitter for layout management."""
        self._splitters[name] = splitter
    
    def apply_layout(self, layout_type: str = None):
        """Apply layout configuration to registered widgets."""
        config = self.get_layout_config(layout_type)
        
        # Apply splitter configurations
        for name, splitter in self._splitters.items():
            if name == 'main_splitter':
                # Set orientation
                if config['main_splitter_orientation'] == 'vertical':
                    splitter.setOrientation(1)  # Qt.Vertical
                else:
                    splitter.setOrientation(2)  # Qt.Horizontal
                
                # Set sizes
                splitter.setSizes(config['main_splitter_sizes'])
        
        # Apply widget-specific configurations
        self._apply_widget_styles(config)
    
    def _apply_widget_styles(self, config: Dict):
        """Apply styling based on layout configuration."""
        font_size = config['font_size']
        button_height = config['button_height']
        input_height = config['input_height']
        
        # Create stylesheet for responsive elements
        stylesheet = f"""
        QPushButton {{
            min-height: {button_height}px;
            font-size: {font_size}pt;
        }}
        
        QLineEdit, QComboBox, QDateEdit {{
            min-height: {input_height}px;
            font-size: {font_size}pt;
        }}
        
        QLabel {{
            font-size: {font_size}pt;
        }}
        
        QGroupBox {{
            font-size: {font_size + 1}pt;
        }}
        
        QTextEdit {{
            font-size: {font_size - 1}pt;
        }}
        """
        
        # Apply to registered widgets
        for widget in self._widgets.values():
            widget.setStyleSheet(widget.styleSheet() + stylesheet)
    
    def get_optimal_chart_layout(self, num_charts: int, layout_type: str = None) -> Tuple[int, int]:
        """Get optimal chart grid layout for given number of charts."""
        config = self.get_layout_config(layout_type)
        max_cols = config['chart_columns']
        max_rows = config['chart_rows']
        
        if num_charts <= max_cols:
            return (1, num_charts)
        
        # Calculate optimal rows and columns
        cols = min(max_cols, num_charts)
        rows = math.ceil(num_charts / cols)
        rows = min(rows, max_rows)
        
        return (rows, cols)
    
    def get_control_panel_width(self, layout_type: str = None) -> int:
        """Get optimal control panel width."""
        config = self.get_layout_config(layout_type)
        return config['control_panel_width']
    
    def get_spacing(self, layout_type: str = None) -> int:
        """Get layout spacing."""
        config = self.get_layout_config(layout_type)
        return config['spacing']
    
    def get_margins(self, layout_type: str = None) -> Tuple[int, int, int, int]:
        """Get layout margins (left, top, right, bottom)."""
        config = self.get_layout_config(layout_type)
        return config['margins']
    
    def calculate_widget_size(self, base_width: int, base_height: int, 
                            layout_type: str = None) -> QSize:
        """Calculate responsive widget size."""
        config = self.get_layout_config(layout_type)
        
        # Scale factors based on layout type
        scale_factors = {
            'small': 0.8,
            'medium': 1.0,
            'large': 1.2,
            'xlarge': 1.4
        }
        
        layout_type = layout_type or self._current_layout
        scale = scale_factors.get(layout_type, 1.0)
        
        scaled_width = int(base_width * scale)
        scaled_height = int(base_height * scale)
        
        return QSize(scaled_width, scaled_height)
    
    def is_compact_layout(self, layout_type: str = None) -> bool:
        """Check if current layout is compact (small screen)."""
        layout_type = layout_type or self._current_layout
        return layout_type == 'small'
    
    def get_recommended_window_size(self, layout_type: str = None) -> QSize:
        """Get recommended window size for layout type."""
        layout_type = layout_type or self._current_layout
        
        sizes = {
            'small': QSize(1024, 768),
            'medium': QSize(1366, 900),
            'large': QSize(1920, 1080),
            'xlarge': QSize(2560, 1440)
        }
        
        return sizes.get(layout_type, QSize(1366, 900))


# Global layout manager instance
layout_manager = LayoutManager()