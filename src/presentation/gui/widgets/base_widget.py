"""
Base widget classes for modular widget system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import pyqtSignal, QObject
from ..themes import theme_manager
from ..layout_manager import layout_manager


class BaseWidget(QWidget):
    """Base class for all custom widgets."""
    
    # Signals
    data_changed = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    action_requested = pyqtSignal(str, dict)  # action_name, parameters
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._widget_id = None
        self._config = {}
        self._is_initialized = False
        
        # Connect to theme changes
        theme_manager.theme_changed.connect(self._on_theme_changed)
        layout_manager.layout_changed.connect(self._on_layout_changed)
        
        self._setup_widget()
    
    def _setup_widget(self):
        """Setup the widget. Override in subclasses."""
        pass
    
    def set_widget_id(self, widget_id: str):
        """Set unique identifier for this widget."""
        self._widget_id = widget_id
    
    def get_widget_id(self) -> Optional[str]:
        """Get widget identifier."""
        return self._widget_id
    
    def set_config(self, config: Dict[str, Any]):
        """Set widget configuration."""
        self._config = config.copy()
        self._apply_config()
    
    def get_config(self) -> Dict[str, Any]:
        """Get widget configuration."""
        return self._config.copy()
    
    def _apply_config(self):
        """Apply configuration to widget. Override in subclasses."""
        pass
    
    def _on_theme_changed(self, theme_name: str):
        """Handle theme changes."""
        self._apply_theme(theme_name)
    
    def _on_layout_changed(self, layout_type: str):
        """Handle layout changes."""
        self._apply_layout(layout_type)
    
    def _apply_theme(self, theme_name: str):
        """Apply theme to widget. Override in subclasses."""
        pass
    
    def _apply_layout(self, layout_type: str):
        """Apply layout changes to widget. Override in subclasses."""
        pass
    
    def emit_action(self, action_name: str, parameters: Dict[str, Any] = None):
        """Emit action request signal."""
        if parameters is None:
            parameters = {}
        self.action_requested.emit(action_name, parameters)
    
    def emit_error(self, error_message: str):
        """Emit error signal."""
        self.error_occurred.emit(error_message)
    
    def update_data(self, data: Any):
        """Update widget with new data. Override in subclasses."""
        self.data_changed.emit(data)
    
    def clear_data(self):
        """Clear widget data. Override in subclasses."""
        pass
    
    def validate_data(self, data: Any) -> bool:
        """Validate data before updating. Override in subclasses."""
        return True
    
    def get_current_data(self) -> Any:
        """Get current widget data. Override in subclasses."""
        return None


class ContainerWidget(BaseWidget):
    """Base container widget for organizing other widgets."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._child_widgets = {}
        self._layout = None
    
    def add_widget(self, widget: BaseWidget, widget_id: str, **layout_params):
        """Add a child widget to the container."""
        widget.set_widget_id(widget_id)
        self._child_widgets[widget_id] = widget
        
        # Connect child signals
        widget.action_requested.connect(self._on_child_action)
        widget.error_occurred.connect(self._on_child_error)
        
        # Add to layout
        if self._layout:
            self._add_widget_to_layout(widget, **layout_params)
    
    def remove_widget(self, widget_id: str):
        """Remove a child widget from the container."""
        if widget_id in self._child_widgets:
            widget = self._child_widgets[widget_id]
            if self._layout:
                self._layout.removeWidget(widget)
            widget.setParent(None)
            del self._child_widgets[widget_id]
    
    def get_widget(self, widget_id: str) -> Optional[BaseWidget]:
        """Get child widget by ID."""
        return self._child_widgets.get(widget_id)
    
    def get_all_widgets(self) -> Dict[str, BaseWidget]:
        """Get all child widgets."""
        return self._child_widgets.copy()
    
    def _add_widget_to_layout(self, widget: BaseWidget, **layout_params):
        """Add widget to layout. Override in subclasses."""
        if self._layout:
            self._layout.addWidget(widget)
    
    def _on_child_action(self, action_name: str, parameters: Dict[str, Any]):
        """Handle child widget actions."""
        # Forward to parent or handle locally
        self.action_requested.emit(action_name, parameters)
    
    def _on_child_error(self, error_message: str):
        """Handle child widget errors."""
        self.error_occurred.emit(error_message)
    
    def broadcast_data(self, data: Any):
        """Broadcast data to all child widgets."""
        for widget in self._child_widgets.values():
            if widget.validate_data(data):
                widget.update_data(data)
    
    def clear_all_data(self):
        """Clear data from all child widgets."""
        for widget in self._child_widgets.values():
            widget.clear_data()


class PanelWidget(ContainerWidget):
    """Panel widget with vertical layout."""
    
    def _setup_widget(self):
        """Setup panel with vertical layout."""
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        
        # Apply layout manager settings
        spacing = layout_manager.get_spacing()
        margins = layout_manager.get_margins()
        
        self._layout.setSpacing(spacing)
        self._layout.setContentsMargins(*margins)


class GroupWidget(ContainerWidget):
    """Group widget with title and border."""
    
    def __init__(self, title: str = "", parent=None):
        self._title = title
        super().__init__(parent)
    
    def _setup_widget(self):
        """Setup group widget."""
        from PyQt5.QtWidgets import QGroupBox, QVBoxLayout
        
        # Create group box
        self._group_box = QGroupBox(self._title)
        
        # Setup layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self._group_box)
        self.setLayout(main_layout)
        
        # Internal layout for child widgets
        self._layout = QVBoxLayout()
        self._group_box.setLayout(self._layout)
        
        # Apply layout manager settings
        spacing = layout_manager.get_spacing()
        margins = layout_manager.get_margins()
        
        self._layout.setSpacing(spacing)
        self._layout.setContentsMargins(*margins)
    
    def set_title(self, title: str):
        """Set group title."""
        self._title = title
        if hasattr(self, '_group_box'):
            self._group_box.setTitle(title)


class SplitterWidget(ContainerWidget):
    """Splitter widget for resizable panels."""
    
    def __init__(self, orientation='horizontal', parent=None):
        self._orientation = orientation
        super().__init__(parent)
    
    def _setup_widget(self):
        """Setup splitter widget."""
        from PyQt5.QtWidgets import QSplitter, QVBoxLayout
        from PyQt5.QtCore import Qt
        
        # Create splitter
        orientation = Qt.Horizontal if self._orientation == 'horizontal' else Qt.Vertical
        self._splitter = QSplitter(orientation)
        
        # Setup layout
        layout = QVBoxLayout()
        layout.addWidget(self._splitter)
        self.setLayout(layout)
        
        # Register with layout manager
        layout_manager.register_splitter(f"splitter_{id(self)}", self._splitter)
    
    def _add_widget_to_layout(self, widget: BaseWidget, **layout_params):
        """Add widget to splitter."""
        self._splitter.addWidget(widget)
    
    def set_sizes(self, sizes: list):
        """Set splitter sizes."""
        self._splitter.setSizes(sizes)
    
    def get_sizes(self) -> list:
        """Get current splitter sizes."""
        return self._splitter.sizes()