"""
Control panel widget for input parameters and controls.
"""

from typing import Dict, Any, List
from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, 
    QDateEdit, QComboBox, QTextEdit, QGroupBox, QCheckBox, QSpinBox
)
from PyQt5.QtCore import QDate, pyqtSignal, Qt
from .base_widget import BaseWidget, GroupWidget
from .stock_symbol_widget import StockSymbolWidget
from ..themes import theme_manager


class InputWidget(BaseWidget):
    """Base input widget."""
    
    value_changed = pyqtSignal(object)
    
    def __init__(self, label: str = "", parent=None):
        self._label = label
        self._input_widget = None
        super().__init__(parent)
    
    def _setup_widget(self):
        """Setup input widget layout."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        if self._label:
            label_widget = QLabel(self._label)
            layout.addWidget(label_widget)
        
        self._create_input_widget()
        if self._input_widget:
            layout.addWidget(self._input_widget)
    
    def _create_input_widget(self):
        """Create the input widget. Override in subclasses."""
        pass
    
    def get_value(self):
        """Get current value. Override in subclasses."""
        return None
    
    def set_value(self, value):
        """Set value. Override in subclasses."""
        pass
    
    def _on_value_changed(self):
        """Handle value changes."""
        self.value_changed.emit(self.get_value())


class TextInputWidget(InputWidget):
    """Text input widget."""
    
    def __init__(self, label: str = "", placeholder: str = "", parent=None):
        self._placeholder = placeholder
        super().__init__(label, parent)
    
    def _create_input_widget(self):
        """Create text input."""
        self._input_widget = QLineEdit()
        if self._placeholder:
            self._input_widget.setPlaceholderText(self._placeholder)
        
        # Ensure the widget is enabled and focusable
        self._input_widget.setEnabled(True)
        self._input_widget.setFocusPolicy(Qt.StrongFocus)
        
        self._input_widget.textChanged.connect(self._on_value_changed)
    
    def get_value(self) -> str:
        """Get text value."""
        return self._input_widget.text() if self._input_widget else ""
    
    def set_value(self, value: str):
        """Set text value."""
        if self._input_widget:
            self._input_widget.setText(str(value))


class ComboBoxWidget(InputWidget):
    """Combo box input widget."""
    
    def __init__(self, label: str = "", items: List[str] = None, parent=None):
        self._items = items or []
        super().__init__(label, parent)
    
    def _create_input_widget(self):
        """Create combo box."""
        self._input_widget = QComboBox()
        self._input_widget.addItems(self._items)
        self._input_widget.currentTextChanged.connect(self._on_value_changed)
    
    def get_value(self) -> str:
        """Get selected value."""
        return self._input_widget.currentText() if self._input_widget else ""
    
    def set_value(self, value: str):
        """Set selected value."""
        if self._input_widget:
            index = self._input_widget.findText(value)
            if index >= 0:
                self._input_widget.setCurrentIndex(index)
    
    def add_item(self, item: str):
        """Add item to combo box."""
        if self._input_widget:
            self._input_widget.addItem(item)
    
    def set_items(self, items: List[str]):
        """Set all items."""
        if self._input_widget:
            self._input_widget.clear()
            self._input_widget.addItems(items)


class DateInputWidget(InputWidget):
    """Date input widget."""
    
    def __init__(self, label: str = "", parent=None):
        super().__init__(label, parent)
    
    def _create_input_widget(self):
        """Create date input."""
        self._input_widget = QDateEdit()
        self._input_widget.setCalendarPopup(True)
        self._input_widget.setDate(QDate.currentDate())
        self._input_widget.dateChanged.connect(self._on_value_changed)
    
    def get_value(self) -> str:
        """Get date value as string."""
        if self._input_widget:
            return self._input_widget.date().toString("yyyy-MM-dd")
        return ""
    
    def set_value(self, value):
        """Set date value."""
        if self._input_widget:
            if isinstance(value, str):
                date = QDate.fromString(value, "yyyy-MM-dd")
                self._input_widget.setDate(date)
            elif isinstance(value, QDate):
                self._input_widget.setDate(value)


class CheckBoxWidget(InputWidget):
    """Checkbox input widget."""
    
    def __init__(self, label: str = "", parent=None):
        super().__init__(label, parent)
    
    def _create_input_widget(self):
        """Create checkbox."""
        self._input_widget = QCheckBox()
        self._input_widget.stateChanged.connect(self._on_value_changed)
    
    def get_value(self) -> bool:
        """Get checkbox value."""
        return self._input_widget.isChecked() if self._input_widget else False
    
    def set_value(self, value: bool):
        """Set checkbox value."""
        if self._input_widget:
            self._input_widget.setChecked(bool(value))


class SpinBoxWidget(InputWidget):
    """Spin box input widget."""
    
    def __init__(self, label: str = "", min_val: int = 0, max_val: int = 100, parent=None):
        self._min_val = min_val
        self._max_val = max_val
        super().__init__(label, parent)
    
    def _create_input_widget(self):
        """Create spin box."""
        self._input_widget = QSpinBox()
        self._input_widget.setMinimum(self._min_val)
        self._input_widget.setMaximum(self._max_val)
        self._input_widget.valueChanged.connect(self._on_value_changed)
    
    def get_value(self) -> int:
        """Get spin box value."""
        return self._input_widget.value() if self._input_widget else 0
    
    def set_value(self, value: int):
        """Set spin box value."""
        if self._input_widget:
            self._input_widget.setValue(int(value))


class ControlPanelWidget(BaseWidget):
    """Main control panel widget."""
    
    # Signals
    analysis_requested = pyqtSignal(dict)
    export_requested = pyqtSignal(str)
    clear_requested = pyqtSignal()
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        self._input_widgets = {}
        self._button_widgets = {}
        self._console_widget = None
        super().__init__(parent)
    
    def _setup_widget(self):
        """Setup control panel layout."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create input parameters group
        self._create_input_group()
        
        # Create control buttons group
        self._create_button_group()
        
        # Create results summary group
        self._create_results_group()
        
        # Create console output group
        self._create_console_group()
        
        # Add stretch to push everything to top
        layout.addStretch()
    
    def _create_input_group(self):
        """Create input parameters group."""
        input_group = GroupWidget("Input Parameters")
        
        # Enhanced stock symbol widget
        self.stock_symbol_widget = StockSymbolWidget()
        input_group.layout().addWidget(self.stock_symbol_widget)
        
        # Create a wrapper to maintain compatibility with existing code
        class SymbolWidgetWrapper:
            def __init__(self, stock_widget):
                self.stock_widget = stock_widget
                self.value_changed = stock_widget.symbols_changed
            
            def get_value(self):
                return self.stock_widget.get_symbols()
            
            def set_value(self, value):
                self.stock_widget.set_symbols(value)
        
        tickers_widget = SymbolWidgetWrapper(self.stock_symbol_widget)
        self._input_widgets["tickers"] = tickers_widget
        
        # Date inputs
        start_date_widget = DateInputWidget("Start Date:")
        start_date_widget.set_value(QDate(2023, 1, 1))
        input_group.add_widget(start_date_widget, "start_date")
        self._input_widgets["start_date"] = start_date_widget
        
        end_date_widget = DateInputWidget("End Date:")
        end_date_widget.set_value(QDate(2024, 12, 31))
        input_group.add_widget(end_date_widget, "end_date")
        self._input_widgets["end_date"] = end_date_widget
        
        # Strategy selector
        strategy_widget = ComboBoxWidget("Strategy:", 
                                       ['momentum', 'mean_reversion', 'volatility_breakout'])
        input_group.add_widget(strategy_widget, "strategy")
        self._input_widgets["strategy"] = strategy_widget
        
        # Optimization method
        optimization_widget = ComboBoxWidget("Optimization Method:",
                                           ['mean_variance', 'equal_weight', 'risk_parity'])
        input_group.add_widget(optimization_widget, "optimization")
        self._input_widgets["optimization"] = optimization_widget
        
        # Connect value changes
        for widget in self._input_widgets.values():
            widget.value_changed.connect(self._on_input_changed)
        
        self.layout().addWidget(input_group)
    
    def _create_button_group(self):
        """Create control buttons group."""
        button_group = GroupWidget("Controls")
        
        # Run analysis button
        run_button = QPushButton("Run Full Analysis")
        run_button.clicked.connect(self._on_run_analysis)
        run_button.setObjectName("primary_button")
        button_group.layout().addWidget(run_button)
        self._button_widgets["run"] = run_button
        
        # Clear results button
        clear_button = QPushButton("Clear Results")
        clear_button.clicked.connect(self._on_clear_results)
        button_group.layout().addWidget(clear_button)
        self._button_widgets["clear"] = clear_button
        
        # Export button
        export_button = QPushButton("Export to CSV")
        export_button.clicked.connect(self._on_export_results)
        button_group.layout().addWidget(export_button)
        self._button_widgets["export"] = export_button
        
        # Theme toggle button
        theme_button = QPushButton("Toggle Theme")
        theme_button.clicked.connect(self._on_toggle_theme)
        button_group.layout().addWidget(theme_button)
        self._button_widgets["theme"] = theme_button
        
        self.layout().addWidget(button_group)
    
    def _create_results_group(self):
        """Create results summary group."""
        results_group = GroupWidget("Results Summary")
        
        self._weights_label = QLabel("Portfolio Weights: None")
        self._weights_label.setWordWrap(True)
        results_group.layout().addWidget(self._weights_label)
        
        self._performance_label = QLabel("Performance: None")
        self._performance_label.setWordWrap(True)
        results_group.layout().addWidget(self._performance_label)
        
        self.layout().addWidget(results_group)
    
    def _create_console_group(self):
        """Create console output group."""
        console_group = GroupWidget("Console Output")
        
        self._console_widget = QTextEdit()
        self._console_widget.setMaximumHeight(150)
        self._console_widget.setReadOnly(True)
        console_group.layout().addWidget(self._console_widget)
        
        self.layout().addWidget(console_group)
    
    def _on_input_changed(self, value):
        """Handle input parameter changes."""
        parameters = self.get_parameters()
        self.settings_changed.emit(parameters)
    
    def _on_run_analysis(self):
        """Handle run analysis button click."""
        parameters = self.get_parameters()
        self.analysis_requested.emit(parameters)
    
    def _on_clear_results(self):
        """Handle clear results button click."""
        self.clear_requested.emit()
        self._weights_label.setText("Portfolio Weights: None")
        self._performance_label.setText("Performance: None")
        if self._console_widget:
            self._console_widget.clear()
    
    def _on_export_results(self):
        """Handle export results button click."""
        self.export_requested.emit("csv")
    
    def _on_toggle_theme(self):
        """Handle theme toggle button click."""
        theme_manager.toggle_theme()
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all input parameters."""
        parameters = {}
        for name, widget in self._input_widgets.items():
            parameters[name] = widget.get_value()
        return parameters
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set input parameters."""
        for name, value in parameters.items():
            if name in self._input_widgets:
                self._input_widgets[name].set_value(value)
    
    def update_results(self, results: Dict[str, Any]):
        """Update results display."""
        # Update portfolio weights
        weights = results.get('portfolio_weights', {})
        if weights:
            weights_text = ", ".join([f"{k}: {v:.3f}" for k, v in weights.items()])
            self._weights_label.setText(f"Portfolio Weights: {weights_text}")
        
        # Update performance metrics
        risk_metrics = results.get('risk_metrics', {})
        if risk_metrics:
            performance_text = (
                f"Total Return: {risk_metrics.get('Mean_Return', 0):.2%}, "
                f"Volatility: {risk_metrics.get('Volatility', 0):.2%}, "
                f"Sharpe: {risk_metrics.get('Sharpe_Ratio', 0):.2f}"
            )
            self._performance_label.setText(f"Performance: {performance_text}")
    
    def log_message(self, message: str):
        """Add message to console output."""
        if self._console_widget:
            self._console_widget.append(message)
            self._console_widget.ensureCursorVisible()
    
    def set_loading_state(self, loading: bool):
        """Set loading state for buttons."""
        for button in self._button_widgets.values():
            button.setEnabled(not loading)
        
        if loading:
            self._button_widgets["run"].setText("Running Analysis...")
        else:
            self._button_widgets["run"].setText("Run Full Analysis")
    
    def _apply_theme(self, theme_name: str):
        """Apply theme to control panel."""
        colors = theme_manager.get_theme_colors(theme_name)
        
        # Update primary button style
        primary_button_style = f"""
        QPushButton#primary_button {{
            background-color: {colors['success']};
            color: white;
            font-weight: bold;
        }}
        QPushButton#primary_button:hover {{
            background-color: {colors['primary']};
        }}
        """
        
        if "run" in self._button_widgets:
            self._button_widgets["run"].setStyleSheet(primary_button_style)