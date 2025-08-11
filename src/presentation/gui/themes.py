"""
Theme management system for the PyQt application.
Provides dark/light mode switching and responsive styling.
"""

from typing import Dict, Any
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
# Beautiful dark theme stylesheet
DARK_THEME_STYLE = """
/* Main Window */
QMainWindow {
    background-color: #1e1e1e;
    color: #ffffff;
}

/* Central Widget */
QWidget {
    background-color: #1e1e1e;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Frames and Panels */
QFrame {
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 8px;
}

/* Labels */
QLabel {
    color: #ffffff;
    font-size: 13px;
    background-color: transparent;
    border: none;
}

/* Buttons */
QPushButton {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #4a9eff, stop: 1 #0078d4);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    font-size: 13px;
    min-height: 20px;
}

QPushButton:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #5ba7ff, stop: 1 #106ebe);
}

QPushButton:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #3d8bfd, stop: 1 #0056b3);
}

QPushButton:disabled {
    background-color: #404040;
    color: #808080;
}

/* Primary Button (Run Analysis) */
QPushButton#primary_button {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #28a745, stop: 1 #1e7e34);
    font-size: 14px;
    font-weight: 700;
    min-height: 25px;
    padding: 12px 24px;
}

QPushButton#primary_button:hover {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #34ce57, stop: 1 #228b3d);
}

QPushButton#primary_button:pressed {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #1e7e34, stop: 1 #155724);
}

/* Input Fields */
QLineEdit, QTextEdit, QDateEdit {
    background-color: #3d3d3d;
    border: 2px solid #555555;
    border-radius: 6px;
    padding: 8px 12px;
    color: #ffffff;
    font-size: 13px;
    selection-background-color: #0078d4;
}

QLineEdit:focus, QTextEdit:focus, QDateEdit:focus {
    border: 2px solid #0078d4;
    background-color: #404040;
}

QLineEdit:hover, QTextEdit:hover, QDateEdit:hover {
    border: 2px solid #666666;
}

/* Combo Boxes */
QComboBox {
    background-color: #3d3d3d;
    border: 2px solid #555555;
    border-radius: 6px;
    padding: 8px 12px;
    color: #ffffff;
    font-size: 13px;
    min-width: 100px;
}

QComboBox:hover {
    border: 2px solid #666666;
}

QComboBox:focus {
    border: 2px solid #0078d4;
}

QComboBox QAbstractItemView {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 6px;
    color: #ffffff;
    selection-background-color: #0078d4;
    outline: none;
}

/* List Widgets */
QListWidget {
    background-color: #3d3d3d;
    border: 2px solid #555555;
    border-radius: 6px;
    color: #ffffff;
    font-size: 13px;
    outline: none;
}

QListWidget::item {
    padding: 6px 12px;
    border-bottom: 1px solid #555555;
}

QListWidget::item:selected {
    background-color: #0078d4;
    color: white;
}

QListWidget::item:hover {
    background-color: #404040;
}

/* Group Boxes */
QGroupBox {
    font-weight: 600;
    font-size: 14px;
    color: #ffffff;
    border: 2px solid #555555;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
    background-color: #2d2d2d;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px 0 8px;
    background-color: #2d2d2d;
    color: #4a9eff;
}

/* Tab Widget */
QTabWidget::pane {
    border: 2px solid #555555;
    background-color: #2d2d2d;
    border-radius: 8px;
}

QTabWidget::tab-bar {
    alignment: left;
}

QTabBar::tab {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #404040, stop: 1 #2d2d2d);
    border: 2px solid #555555;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 10px 16px;
    margin-right: 2px;
    color: #cccccc;
    font-weight: 500;
    min-width: 80px;
}

QTabBar::tab:selected {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #4a9eff, stop: 1 #0078d4);
    color: white;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #505050, stop: 1 #3d3d3d);
    color: #ffffff;
}

/* Progress Bar */
QProgressBar {
    background-color: #3d3d3d;
    border: 2px solid #555555;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #28a745, stop: 1 #1e7e34);
    border-radius: 4px;
}

/* Menu Bar */
QMenuBar {
    background-color: #2d2d2d;
    color: #ffffff;
    border-bottom: 1px solid #555555;
}

QMenuBar::item {
    background-color: transparent;
    padding: 6px 12px;
}

QMenuBar::item:selected {
    background-color: #0078d4;
    border-radius: 4px;
}

QMenu {
    background-color: #2d2d2d;
    border: 1px solid #555555;
    border-radius: 6px;
    color: #ffffff;
}

QMenu::item {
    padding: 8px 16px;
}

QMenu::item:selected {
    background-color: #0078d4;
    border-radius: 4px;
}

/* Status Bar */
QStatusBar {
    background-color: #2d2d2d;
    color: #ffffff;
    border-top: 1px solid #555555;
}

/* Panel Title */
QLabel#panel_title {
    font-size: 18px;
    font-weight: 700;
    color: #4a9eff;
    margin-bottom: 16px;
    padding: 8px 0px;
}

/* Splitter Handle */
QSplitter::handle {
    background-color: #555555;
    width: 2px;
    height: 2px;
}

QSplitter::handle:horizontal {
    width: 2px;
}

QSplitter::handle:vertical {
    height: 2px;
}

QSplitter::handle:hover {
    background-color: #0078d4;
}

/* Section Titles */
QLabel#section_title {
    font-size: 20px;
    font-weight: 700;
    color: #4a9eff;
    margin-bottom: 16px;
    padding: 8px 0px;
}

QLabel#subsection_title {
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin-top: 16px;
    margin-bottom: 8px;
}

/* Panel Title */
QLabel#panel_title {
    font-size: 18px;
    font-weight: 700;
    color: #4a9eff;
    margin-bottom: 16px;
    padding: 8px 0px;
    background-color: transparent;
}

/* Control Panel Frame */
QFrame#control_panel_frame {
    background-color: #2d2d2d;
    border-right: 2px solid #555555;
    border-radius: 0px;
}

/* Dashboard Frames */
QFrame[class="dashboard_frame"] {
    background-color: #2d2d2d;
    border: 2px solid #555555;
    border-radius: 8px;
    padding: 20px;
}

/* Dashboard Labels */
QLabel[class="dashboard_title"] {
    font-size: 24px;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
    background-color: transparent;
}

QLabel[class="dashboard_subtitle"] {
    font-size: 16px;
    color: #cccccc;
    margin-bottom: 16px;
    background-color: transparent;
}

QLabel[class="feature_label"] {
    font-size: 14px;
    color: #ffffff;
    margin: 4px 0;
    background-color: transparent;
}

QLabel[class="step_label"] {
    font-size: 14px;
    color: #ffffff;
    margin: 6px 0;
    padding-left: 16px;
    background-color: transparent;
}

/* Results Frame */
QFrame[class="results_frame"] {
    background-color: #1e1e1e;
    border: none;
    border-radius: 0px;
}

/* Placeholder Frames */
QFrame[class="placeholder_frame"] {
    background-color: #2d2d2d;
    border: 2px solid #555555;
    border-radius: 8px;
    padding: 40px;
}

QLabel[class="placeholder_title"] {
    font-size: 20px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 16px;
    background-color: transparent;
}

QLabel[class="placeholder_description"] {
    font-size: 14px;
    color: #cccccc;
    text-align: center;
    background-color: transparent;
}

/* Metric Cards */
QFrame[class="metric_card"] {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                stop: 0 #3d3d3d, stop: 1 #2d2d2d);
    border: 2px solid #555555;
    border-radius: 8px;
    padding: 16px;
    margin: 4px;
    min-width: 120px;
    max-width: 200px;
}

QLabel[class="metric_title"] {
    font-size: 12px;
    font-weight: 600;
    color: #cccccc;
    margin-bottom: 8px;
    background-color: transparent;
}

QLabel[class="metric_value"] {
    font-size: 24px;
    font-weight: 700;
    color: #4a9eff;
    margin-bottom: 4px;
    background-color: transparent;
}

QLabel[class="metric_subtitle"] {
    font-size: 10px;
    color: #999999;
    background-color: transparent;
}

/* Chart Titles */
QLabel[class="chart_title"] {
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 8px;
    padding: 8px 0px;
    background-color: transparent;
}
"""


class ThemeManager(QObject):
    """Manages application themes and styling."""
    
    theme_changed = pyqtSignal(str)  # Emitted when theme changes
    
    def __init__(self):
        super().__init__()
        self._current_theme = "light"
        self._themes = {
            "light": self._get_light_theme(),
            "dark": self._get_dark_theme()
        }
    
    def _get_light_theme(self) -> Dict[str, Any]:
        """Get light theme configuration."""
        return {
            "name": "light",
            "colors": {
                "primary": "#2196F3",
                "secondary": "#FFC107",
                "success": "#4CAF50",
                "danger": "#F44336",
                "warning": "#FF9800",
                "info": "#00BCD4",
                "background": "#FFFFFF",
                "surface": "#F5F5F5",
                "text_primary": "#212121",
                "text_secondary": "#757575",
                "border": "#E0E0E0",
                "hover": "#F0F0F0",
                "selected": "#E3F2FD"
            },
            "stylesheet": self._get_light_stylesheet()
        }
    
    def _get_dark_theme(self) -> Dict[str, Any]:
        """Get dark theme configuration."""
        return {
            "name": "dark",
            "colors": {
                "primary": "#2196F3",
                "secondary": "#FFC107",
                "success": "#4CAF50",
                "danger": "#F44336",
                "warning": "#FF9800",
                "info": "#00BCD4",
                "background": "#121212",
                "surface": "#1E1E1E",
                "text_primary": "#FFFFFF",
                "text_secondary": "#B0B0B0",
                "border": "#333333",
                "hover": "#2C2C2C",
                "selected": "#1976D2"
            },
            "stylesheet": self._get_dark_stylesheet()
        }
    
    def _get_light_stylesheet(self) -> str:
        """Get light theme stylesheet."""
        return """
        QMainWindow {
            background-color: #FFFFFF;
            color: #212121;
        }
        
        QWidget {
            background-color: #FFFFFF;
            color: #212121;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #E0E0E0;
            border-radius: 5px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QPushButton {
            background-color: #2196F3;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #1976D2;
        }
        
        QPushButton:pressed {
            background-color: #0D47A1;
        }
        
        QPushButton:disabled {
            background-color: #BDBDBD;
            color: #757575;
        }
        
        QLineEdit, QComboBox, QDateEdit {
            border: 2px solid #E0E0E0;
            border-radius: 4px;
            padding: 5px;
            background-color: white;
        }
        
        QLineEdit:focus, QComboBox:focus, QDateEdit:focus {
            border-color: #2196F3;
        }
        
        QTextEdit {
            border: 2px solid #E0E0E0;
            border-radius: 4px;
            background-color: white;
        }
        
        QSplitter::handle {
            background-color: #E0E0E0;
        }
        
        QSplitter::handle:horizontal {
            width: 3px;
        }
        
        QSplitter::handle:vertical {
            height: 3px;
        }
        
        QTabWidget::pane {
            border: 1px solid #E0E0E0;
            background-color: white;
        }
        
        QTabBar::tab {
            background-color: #F5F5F5;
            border: 1px solid #E0E0E0;
            padding: 8px 16px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background-color: white;
            border-bottom-color: white;
        }
        
        QTabBar::tab:hover {
            background-color: #F0F0F0;
        }
        """
    
    def _get_dark_stylesheet(self) -> str:
        """Get dark theme stylesheet."""
        return DARK_THEME_STYLE
    
    @property
    def current_theme(self) -> str:
        """Get current theme name."""
        return self._current_theme
    
    @property
    def available_themes(self) -> list:
        """Get list of available theme names."""
        return list(self._themes.keys())
    
    def get_theme_colors(self, theme_name: str = None) -> Dict[str, str]:
        """Get colors for specified theme or current theme."""
        theme_name = theme_name or self._current_theme
        return self._themes.get(theme_name, {}).get("colors", {})
    
    def set_theme(self, theme_name: str):
        """Set the current theme."""
        if theme_name not in self._themes:
            raise ValueError(f"Theme '{theme_name}' not found")
        
        self._current_theme = theme_name
        self._apply_theme()
        self.theme_changed.emit(theme_name)
    
    def toggle_theme(self):
        """Toggle between light and dark themes."""
        new_theme = "dark" if self._current_theme == "light" else "light"
        self.set_theme(new_theme)
    
    def _apply_theme(self):
        """Apply the current theme to the application."""
        app = QApplication.instance()
        if app:
            theme = self._themes[self._current_theme]
            app.setStyleSheet(theme["stylesheet"])
            
            # Set palette for better integration
            palette = self._create_palette(theme["colors"])
            app.setPalette(palette)
    
    def _create_palette(self, colors: Dict[str, str]) -> QPalette:
        """Create a QPalette from theme colors."""
        palette = QPalette()
        
        # Window colors
        palette.setColor(QPalette.Window, QColor(colors["background"]))
        palette.setColor(QPalette.WindowText, QColor(colors["text_primary"]))
        
        # Base colors (for input fields)
        palette.setColor(QPalette.Base, QColor(colors["surface"]))
        palette.setColor(QPalette.AlternateBase, QColor(colors["hover"]))
        
        # Text colors
        palette.setColor(QPalette.Text, QColor(colors["text_primary"]))
        palette.setColor(QPalette.BrightText, QColor(colors["text_primary"]))
        
        # Button colors
        palette.setColor(QPalette.Button, QColor(colors["surface"]))
        palette.setColor(QPalette.ButtonText, QColor(colors["text_primary"]))
        
        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor(colors["primary"]))
        palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
        
        return palette
    
    def get_color(self, color_name: str, theme_name: str = None) -> str:
        """Get a specific color from the theme."""
        theme_name = theme_name or self._current_theme
        colors = self.get_theme_colors(theme_name)
        return colors.get(color_name, "#000000")


# Global theme manager instance
theme_manager = ThemeManager()