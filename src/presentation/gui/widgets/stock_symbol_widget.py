"""
Enhanced stock symbol input widget with dropdown and autocomplete functionality.
"""

from typing import List, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, 
    QPushButton, QCompleter, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal, QStringListModel
from PyQt5.QtGui import QFont


class StockSymbolWidget(QWidget):
    """Enhanced stock symbol input widget with dropdown and autocomplete."""
    
    symbols_changed = pyqtSignal(str)  # Emits comma-separated symbols
    
    # Most common stocks organized by category
    POPULAR_STOCKS = {
        "FAANG+": ["AAPL", "AMZN", "GOOGL", "META", "NFLX", "TSLA", "MSFT", "NVDA"],
        "Dow 30": [
            "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
            "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
            "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
        ],
        "Top Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF"],
        "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "ORCL", "CRM", "ADBE"],
        "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "MRK", "DHR", "BMY", "AMGN"],
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PXD", "KMI", "OKE", "WMB", "MPC"],
        "Consumer": ["WMT", "HD", "PG", "KO", "PEP", "COST", "NKE", "SBUX", "MCD", "TGT"]
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_symbols = []
        self.recent_symbols = []
        self.favorites = []
        self._setup_ui()
        self._setup_connections()
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title_label = QLabel("Stock Symbols")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # Input section
        input_layout = QHBoxLayout()
        
        # Manual input field with autocomplete
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Enter symbols (e.g., AAPL,MSFT,GOOGL)")
        self.symbol_input.setMinimumHeight(30)
        
        # Setup autocomplete
        all_symbols = []
        for category_symbols in self.POPULAR_STOCKS.values():
            all_symbols.extend(category_symbols)
        all_symbols = sorted(list(set(all_symbols)))  # Remove duplicates and sort
        
        completer = QCompleter(all_symbols)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.symbol_input.setCompleter(completer)
        
        input_layout.addWidget(self.symbol_input)
        
        # Add symbol button
        add_button = QPushButton("Add")
        add_button.setMaximumWidth(60)
        add_button.clicked.connect(self._add_current_symbol)
        input_layout.addWidget(add_button)
        
        layout.addLayout(input_layout)
        
        # Category dropdown section
        category_layout = QHBoxLayout()
        
        category_label = QLabel("Quick Select:")
        category_layout.addWidget(category_label)
        
        self.category_combo = QComboBox()
        self.category_combo.addItem("Select Category...")
        for category in self.POPULAR_STOCKS.keys():
            self.category_combo.addItem(category)
        category_layout.addWidget(self.category_combo)
        
        # Stock dropdown (populated based on category)
        self.stock_combo = QComboBox()
        self.stock_combo.addItem("Select Stock...")
        self.stock_combo.setEnabled(False)
        category_layout.addWidget(self.stock_combo)
        
        # Add selected button
        add_selected_button = QPushButton("Add Selected")
        add_selected_button.clicked.connect(self._add_selected_stock)
        category_layout.addWidget(add_selected_button)
        
        layout.addLayout(category_layout)
        
        # Quick preset buttons
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Presets:")
        preset_layout.addWidget(preset_label)
        
        faang_button = QPushButton("FAANG+")
        faang_button.clicked.connect(lambda: self._load_preset("FAANG+"))
        preset_layout.addWidget(faang_button)
        
        dow_button = QPushButton("Dow 30")
        dow_button.clicked.connect(lambda: self._load_preset("Dow 30"))
        preset_layout.addWidget(dow_button)
        
        tech_button = QPushButton("Tech Giants")
        tech_button.clicked.connect(lambda: self._load_preset("Tech Giants"))
        preset_layout.addWidget(tech_button)
        
        preset_layout.addStretch()
        layout.addLayout(preset_layout)
        
        # Selected symbols display
        selected_label = QLabel("Selected Symbols:")
        layout.addWidget(selected_label)
        
        self.selected_list = QListWidget()
        self.selected_list.setMaximumHeight(100)
        layout.addWidget(self.selected_list)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self._clear_all)
        control_layout.addWidget(clear_button)
        
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._remove_selected)
        control_layout.addWidget(remove_button)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Apply styling
        self._apply_styling()
        
        # Set default symbols
        self._load_preset("FAANG+")
    
    def _setup_connections(self):
        """Setup signal connections."""
        self.symbol_input.returnPressed.connect(self._add_current_symbol)
        self.symbol_input.textChanged.connect(self._validate_input)
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        self.stock_combo.currentTextChanged.connect(self._on_stock_selected)
        self.selected_list.itemDoubleClicked.connect(self._remove_selected)
    
    def _validate_input(self, text: str):
        """Validate input and provide visual feedback."""
        # Basic validation - check for valid symbol format
        if text:
            symbols = [s.strip().upper() for s in text.split(',') if s.strip()]
            valid_symbols = []
            for symbol in symbols:
                if symbol.isalpha() and 1 <= len(symbol) <= 5:
                    valid_symbols.append(symbol)
            
            if valid_symbols:
                self.symbol_input.setStyleSheet("QLineEdit { border: 2px solid green; }")
            else:
                self.symbol_input.setStyleSheet("QLineEdit { border: 2px solid orange; }")
        else:
            self.symbol_input.setStyleSheet("")
    
    def _on_category_changed(self, category: str):
        """Handle category selection change."""
        self.stock_combo.clear()
        self.stock_combo.addItem("Select Stock...")
        
        if category in self.POPULAR_STOCKS:
            self.stock_combo.setEnabled(True)
            for stock in self.POPULAR_STOCKS[category]:
                self.stock_combo.addItem(stock)
        else:
            self.stock_combo.setEnabled(False)
    
    def _on_stock_selected(self, stock: str):
        """Handle individual stock selection."""
        if stock != "Select Stock..." and stock not in self.selected_symbols:
            self.selected_symbols.append(stock)
            self._update_selected_list()
            self._emit_symbols_changed()
    
    def _add_current_symbol(self):
        """Add symbols from the input field."""
        text = self.symbol_input.text().strip()
        if text:
            symbols = [s.strip().upper() for s in text.split(',') if s.strip()]
            for symbol in symbols:
                if symbol.isalpha() and 1 <= len(symbol) <= 5 and symbol not in self.selected_symbols:
                    self.selected_symbols.append(symbol)
            
            self.symbol_input.clear()
            self._update_selected_list()
            self._emit_symbols_changed()
    
    def _add_selected_stock(self):
        """Add the currently selected stock from dropdown."""
        stock = self.stock_combo.currentText()
        if stock != "Select Stock..." and stock not in self.selected_symbols:
            self.selected_symbols.append(stock)
            self._update_selected_list()
            self._emit_symbols_changed()
    
    def _load_preset(self, preset_name: str):
        """Load a preset group of stocks."""
        if preset_name in self.POPULAR_STOCKS:
            self.selected_symbols = self.POPULAR_STOCKS[preset_name].copy()
            self._update_selected_list()
            self._emit_symbols_changed()
    
    def _update_selected_list(self):
        """Update the selected symbols list widget."""
        self.selected_list.clear()
        for symbol in self.selected_symbols:
            item = QListWidgetItem(symbol)
            item.setToolTip(f"Double-click to remove {symbol}")
            self.selected_list.addItem(item)
    
    def _remove_selected(self):
        """Remove the selected symbol from the list."""
        current_item = self.selected_list.currentItem()
        if current_item:
            symbol = current_item.text()
            if symbol in self.selected_symbols:
                self.selected_symbols.remove(symbol)
                self._update_selected_list()
                self._emit_symbols_changed()
    
    def _clear_all(self):
        """Clear all selected symbols."""
        self.selected_symbols.clear()
        self._update_selected_list()
        self._emit_symbols_changed()
    
    def _emit_symbols_changed(self):
        """Emit the symbols changed signal."""
        symbols_str = ','.join(self.selected_symbols)
        self.symbols_changed.emit(symbols_str)
    
    def get_symbols(self) -> str:
        """Get the current symbols as a comma-separated string."""
        return ','.join(self.selected_symbols)
    
    def set_symbols(self, symbols_str: str):
        """Set symbols from a comma-separated string."""
        if symbols_str:
            symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
            self.selected_symbols = symbols
            self._update_selected_list()
            self._emit_symbols_changed()
    
    def get_symbols_list(self) -> List[str]:
        """Get the current symbols as a list."""
        return self.selected_symbols.copy()
    
    def _apply_styling(self):
        """Apply custom styling to the widget."""
        self.setStyleSheet("""
            StockSymbolWidget {
                background-color: transparent;
            }
            
            QLabel {
                color: #ffffff;
                font-weight: 600;
            }
            
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #4a9eff, stop: 1 #0078d4);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: 600;
                font-size: 12px;
                min-height: 16px;
            }
            
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #5ba7ff, stop: 1 #106ebe);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #3d8bfd, stop: 1 #0056b3);
            }
            
            QLineEdit {
                background-color: #3d3d3d;
                border: 2px solid #555555;
                border-radius: 6px;
                padding: 8px 12px;
                color: #ffffff;
                font-size: 13px;
            }
            
            QLineEdit:focus {
                border: 2px solid #0078d4;
                background-color: #404040;
            }
            
            QComboBox {
                background-color: #3d3d3d;
                border: 2px solid #555555;
                border-radius: 6px;
                padding: 6px 10px;
                color: #ffffff;
                font-size: 12px;
                min-width: 120px;
            }
            
            QComboBox:hover {
                border: 2px solid #666666;
            }
            
            QComboBox:focus {
                border: 2px solid #0078d4;
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                border: 1px solid #555555;
                border-radius: 6px;
                color: #ffffff;
                selection-background-color: #0078d4;
            }
            
            QListWidget {
                background-color: #3d3d3d;
                border: 2px solid #555555;
                border-radius: 6px;
                color: #ffffff;
                font-size: 12px;
            }
            
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #555555;
            }
            
            QListWidget::item:selected {
                background-color: #0078d4;
                color: white;
            }
            
            QListWidget::item:hover {
                background-color: #404040;
            }
        """)