"""
Main application window implementing MVC/MVP pattern.
"""

import sys
import pandas as pd
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QApplication,
    QMenuBar, QStatusBar, QAction, QMessageBox, QProgressBar, QTabWidget, QLabel,
    QSplitter, QFrame, QToolBar, QStyle
)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, QObject, Qt
from PyQt5.QtGui import QIcon, QKeySequence, QFont, QPalette, QColor

# Import framework components
from ...application.services.data_service import DataService
from ...application.services.strategy_service import StrategyService
from ...application.services.portfolio_service import PortfolioService

# Import GUI components
from .models.base_model import AnalysisModel, PortfolioModel, SettingsModel
from .widgets.base_widget import SplitterWidget
from .widgets.control_panel import ControlPanelWidget
from .widgets.results_dashboard import ResultsDashboard
from .themes import theme_manager
from .layout_manager import layout_manager


class AnalysisWorker(QObject):
    """Worker thread for running analysis."""
    
    progress_updated = pyqtSignal(str, int)
    analysis_completed = pyqtSignal(dict)
    analysis_failed = pyqtSignal(str)
    
    def __init__(self, data_service, strategy_service, portfolio_service):
        super().__init__()
        self.data_service = data_service
        self.strategy_service = strategy_service
        self.portfolio_service = portfolio_service
        self._parameters = {}
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set analysis parameters."""
        self._parameters = parameters
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        try:
            # Extract parameters
            tickers = [t.strip().upper() for t in self._parameters.get('tickers', '').split(',')]
            start_date = self._parameters.get('start_date', '')
            end_date = self._parameters.get('end_date', '')
            strategy = self._parameters.get('strategy', 'momentum')
            optimization = self._parameters.get('optimization', 'mean_variance')
            
            self.progress_updated.emit("Starting analysis...", 0)
            
            # Step 1: Fetch data
            self.progress_updated.emit("Fetching market data...", 10)
            market_data = self.data_service.fetch_market_data(tickers, start_date, end_date)
            
            # Step 2: Clean data
            self.progress_updated.emit("Cleaning data...", 20)
            cleaned_data = self.data_service.clean_data(market_data)
            
            # Step 3: Engineer features
            self.progress_updated.emit("Engineering features...", 30)
            feature_data = self.data_service.engineer_features(cleaned_data)
            
            # Step 4: Generate signals
            self.progress_updated.emit("Generating trading signals...", 50)
            signals = self.strategy_service.generate_signals(feature_data, strategy)
            
            # Step 5: Backtest strategy
            self.progress_updated.emit("Backtesting strategy...", 60)
            portfolio_returns = self.strategy_service.backtest_strategy(feature_data, signals)
            
            # Step 6: Optimize portfolio
            self.progress_updated.emit("Optimizing portfolio...", 70)
            portfolio_weights = self.portfolio_service.optimize_portfolio(
                feature_data, method=optimization
            )
            
            # Step 7: Calculate risk metrics
            self.progress_updated.emit("Calculating risk metrics...", 80)
            risk_metrics = self.portfolio_service.calculate_risk_metrics(portfolio_returns)
            
            # Step 8: Fetch factor data
            self.progress_updated.emit("Fetching factor data...", 90)
            factor_data = self.data_service.fetch_factor_data(start_date, end_date)
            
            # Compile results
            results = {
                'market_data': feature_data,
                'signals': signals,
                'portfolio_returns': portfolio_returns,
                'portfolio_weights': portfolio_weights,
                'risk_metrics': risk_metrics,
                'factor_data': factor_data,
                'parameters': self._parameters
            }
            
            self.progress_updated.emit("Analysis complete!", 100)
            self.analysis_completed.emit(results)
            
        except Exception as e:
            self.analysis_failed.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window with MVC pattern."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize models
        self.analysis_model = AnalysisModel()
        self.portfolio_model = PortfolioModel()
        self.settings_model = SettingsModel()
        
        # Initialize services
        self.data_service = DataService()
        self.strategy_service = StrategyService()
        self.portfolio_service = PortfolioService()
        
        # Initialize worker thread
        self.analysis_thread = QThread()
        self.analysis_worker = AnalysisWorker(
            self.data_service, self.strategy_service, self.portfolio_service
        )
        self.analysis_worker.moveToThread(self.analysis_thread)
        
        # Connect worker signals
        self.analysis_worker.progress_updated.connect(self._on_progress_updated)
        self.analysis_worker.analysis_completed.connect(self._on_analysis_completed)
        self.analysis_worker.analysis_failed.connect(self._on_analysis_failed)
        self.analysis_thread.started.connect(self.analysis_worker.run_analysis)
        
        # UI components
        self.control_panel = None
        self.results_panel = None
        self.progress_bar = None
        
        # Setup window
        self._setup_window()
        self._setup_ui()
        self._setup_connections()
        self._apply_initial_settings()
    
    def _setup_window(self):
        """Setup main window properties with macOS native styling."""
        self.setWindowTitle("Quantitative Research Framework - Professional Edition")
        
        # Set window properties for macOS
        self.setUnifiedTitleAndToolBarOnMac(True)
        
        # Set initial size for macOS (larger, more professional)
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)
        
        # Apply macOS native styling
        self._apply_macos_styling()
        
        # Center window on screen
        self._center_window()
        
        # Setup menu bar
        self._setup_menu_bar()
        
        # Setup status bar
        self._setup_status_bar()
    
    def _apply_macos_styling(self):
        """Apply macOS native styling - now handled by theme manager."""
        # Remove hardcoded styling to let theme manager handle it
        pass
    
    def _setup_menu_bar(self):
        """Setup native macOS menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_action = QAction('New Analysis', self)
        new_action.setShortcut(QKeySequence.New)
        file_menu.addAction(new_action)
        
        open_action = QAction('Open...', self)
        open_action.setShortcut(QKeySequence.Open)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        save_action = QAction('Save', self)
        save_action.setShortcut(QKeySequence.Save)
        file_menu.addAction(save_action)
        
        save_as_action = QAction('Save As...', self)
        save_as_action.setShortcut(QKeySequence.SaveAs)
        file_menu.addAction(save_as_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        
        undo_action = QAction('Undo', self)
        undo_action.setShortcut(QKeySequence.Undo)
        edit_menu.addAction(undo_action)
        
        redo_action = QAction('Redo', self)
        redo_action.setShortcut(QKeySequence.Redo)
        edit_menu.addAction(redo_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        refresh_action = QAction('Refresh', self)
        refresh_action.setShortcut(QKeySequence.Refresh)
        view_menu.addAction(refresh_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')
        
        run_analysis_action = QAction('Run Analysis', self)
        run_analysis_action.setShortcut(QKeySequence('Cmd+R'))
        analysis_menu.addAction(run_analysis_action)
        
        # Window menu (important for macOS)
        window_menu = menubar.addMenu('Window')
        
        minimize_action = QAction('Minimize', self)
        minimize_action.setShortcut(QKeySequence('Cmd+M'))
        window_menu.addAction(minimize_action)
        
        zoom_action = QAction('Zoom', self)
        window_menu.addAction(zoom_action)
    
    def _setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Add progress bar to status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Setup menu bar
        self._setup_menu_bar()
        
        # Setup status bar
        self._setup_status_bar()
    
    def _center_window(self):
        """Center window on screen."""
        screen = QApplication.desktop().screenGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)
    
    def _setup_menu_bar(self):
        """Setup application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # New analysis action
        new_action = QAction('New Analysis', self)
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._on_new_analysis)
        file_menu.addAction(new_action)
        
        # Open action
        open_action = QAction('Open...', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)
        
        # Save action
        save_action = QAction('Save...', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._on_save_file)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Export action
        export_action = QAction('Export Results...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self._on_export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        # Theme actions
        light_theme_action = QAction('Light Theme', self)
        light_theme_action.triggered.connect(lambda: theme_manager.set_theme('light'))
        view_menu.addAction(light_theme_action)
        
        dark_theme_action = QAction('Dark Theme', self)
        dark_theme_action.triggered.connect(lambda: theme_manager.set_theme('dark'))
        view_menu.addAction(dark_theme_action)
        
        view_menu.addSeparator()
        
        # Layout actions
        layout_menu = view_menu.addMenu('Layout')
        
        auto_layout_action = QAction('Auto Detect', self)
        auto_layout_action.triggered.connect(layout_manager.auto_detect_layout)
        layout_menu.addAction(auto_layout_action)
        
        for layout_type in layout_manager._layout_configs.keys():
            action = QAction(layout_type.title(), self)
            action.triggered.connect(lambda checked, lt=layout_type: layout_manager.set_layout(lt))
            layout_menu.addAction(action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self._show_about_dialog)
        help_menu.addAction(about_action)
    
    def _setup_status_bar(self):
        """Setup status bar."""
        self.status_bar = self.statusBar()
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Set initial status
        self.status_bar.showMessage("Ready")
    
    def _setup_ui(self):
        """Setup main UI components with professional macOS styling."""
        try:
            # Create central widget
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            # Create main layout with proper margins
            main_layout = QHBoxLayout()
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            central_widget.setLayout(main_layout)
            
            # Create main splitter with native styling
            main_splitter = QSplitter(Qt.Horizontal)
            main_splitter.setChildrenCollapsible(False)
            main_layout.addWidget(main_splitter)
            
            # Create control panel with dark theme frame
            control_frame = QFrame()
            control_frame.setFrameStyle(QFrame.StyledPanel)
            control_frame.setObjectName("control_panel_frame")
            
            control_layout = QVBoxLayout()
            control_layout.setContentsMargins(16, 16, 16, 16)
            control_frame.setLayout(control_layout)
            
            # Add control panel title
            title_label = QLabel("Analysis Parameters")
            title_label.setObjectName("panel_title")
            control_layout.addWidget(title_label)
            
            # Create control panel widget
            self.control_panel = ControlPanelWidget()
            control_layout.addWidget(self.control_panel)
            
            # Add stretch to push content to top
            control_layout.addStretch()
            
            # Set minimum width for control panel
            control_frame.setMinimumWidth(350)
            control_frame.setMaximumWidth(400)
            main_splitter.addWidget(control_frame)
            
            # Create results panel
            self.results_panel = self._create_professional_results_panel()
            main_splitter.addWidget(self.results_panel)
            
            # Set splitter sizes (control panel smaller, results larger)
            main_splitter.setSizes([350, 1050])
            
            # Store splitter reference
            self.main_splitter = main_splitter
            
        except Exception as e:
            print(f"Error in _setup_ui: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_professional_results_panel(self):
        """Create a professional results panel with native macOS styling."""
        # Create main results frame
        results_frame = QFrame()
        results_frame.setProperty("class", "results_frame")
        
        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(16, 16, 16, 16)
        results_frame.setLayout(results_layout)
        
        # Create tab widget with welcome and results
        tab_widget = QTabWidget()
        tab_widget.setTabPosition(QTabWidget.North)
        results_layout.addWidget(tab_widget)
        
        # Dashboard Tab (welcome/overview)
        self.dashboard_tab = self._create_dashboard_tab()
        tab_widget.addTab(self.dashboard_tab, "📊 Overview")
        
        # Interactive Results Dashboard
        self.results_dashboard = ResultsDashboard()
        tab_widget.addTab(self.results_dashboard, "📈 Analysis Results")
        
        # Store tab widget reference for updating
        self.results_tab_widget = tab_widget
        
        return results_frame
    
    def _create_dashboard_tab(self):
        """Create the main dashboard tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Welcome section
        welcome_frame = QFrame()
        welcome_frame.setProperty("class", "dashboard_frame")
        welcome_layout = QVBoxLayout()
        welcome_frame.setLayout(welcome_layout)
        
        title = QLabel("🚀 Professional Quantitative Framework")
        title.setProperty("class", "dashboard_title")
        welcome_layout.addWidget(title)
        
        subtitle = QLabel("Advanced quantitative analysis with GPU acceleration and parallel processing")
        subtitle.setProperty("class", "dashboard_subtitle")
        welcome_layout.addWidget(subtitle)
        
        # Feature highlights
        features = [
            "✅ Real-time market data analysis",
            "✅ GPU-accelerated computations (Apple MPS supported)",
            "✅ Advanced portfolio optimization",
            "✅ Risk management and VaR calculations",
            "✅ Machine learning integration",
            "✅ Professional reporting system"
        ]
        
        for feature in features:
            feature_label = QLabel(feature)
            feature_label.setProperty("class", "feature_label")
            welcome_layout.addWidget(feature_label)
        
        layout.addWidget(welcome_frame)
        
        # Quick start section
        quickstart_frame = QFrame()
        quickstart_frame.setProperty("class", "dashboard_frame")
        quickstart_layout = QVBoxLayout()
        quickstart_frame.setLayout(quickstart_layout)
        
        quickstart_title = QLabel("🎯 Quick Start Guide")
        quickstart_title.setProperty("class", "dashboard_title")
        quickstart_layout.addWidget(quickstart_title)
        
        steps = [
            "1. Enter stock symbols in the control panel (e.g., AAPL, MSFT, GOOGL)",
            "2. Select date range for analysis",
            "3. Choose analysis type and parameters",
            "4. Click 'Run Analysis' to start processing",
            "5. View results in the Analysis and Portfolio tabs"
        ]
        
        for step in steps:
            step_label = QLabel(step)
            step_label.setProperty("class", "step_label")
            quickstart_layout.addWidget(step_label)
        
        layout.addWidget(quickstart_frame)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def _create_analysis_tab(self):
        """Create the analysis results tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Create scrollable area for results
        from PyQt5.QtWidgets import QScrollArea, QTextEdit
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_widget.setLayout(content_layout)
        
        # Title
        title = QLabel("📈 Analysis Results")
        title.setObjectName("section_title")
        content_layout.addWidget(title)
        
        # Results text area
        self.analysis_results_text = QTextEdit()
        self.analysis_results_text.setReadOnly(True)
        self.analysis_results_text.setMinimumHeight(400)
        self.analysis_results_text.setText("No analysis results yet. Click 'Run Full Analysis' to start.")
        content_layout.addWidget(self.analysis_results_text)
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        tab.setLayout(layout)
        return tab
    
    def _create_portfolio_tab(self):
        """Create the portfolio optimization tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Create scrollable area for results
        from PyQt5.QtWidgets import QScrollArea
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Create content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_widget.setLayout(content_layout)
        
        # Title
        title = QLabel("💼 Portfolio Optimization")
        title.setObjectName("section_title")
        content_layout.addWidget(title)
        
        # Portfolio weights display
        weights_label = QLabel("Portfolio Weights:")
        weights_label.setObjectName("subsection_title")
        content_layout.addWidget(weights_label)
        
        self.portfolio_weights_text = QTextEdit()
        self.portfolio_weights_text.setReadOnly(True)
        self.portfolio_weights_text.setMaximumHeight(200)
        self.portfolio_weights_text.setText("No portfolio weights calculated yet.")
        content_layout.addWidget(self.portfolio_weights_text)
        
        # Risk metrics display
        risk_label = QLabel("Risk Metrics:")
        risk_label.setObjectName("subsection_title")
        content_layout.addWidget(risk_label)
        
        self.risk_metrics_text = QTextEdit()
        self.risk_metrics_text.setReadOnly(True)
        self.risk_metrics_text.setMaximumHeight(200)
        self.risk_metrics_text.setText("No risk metrics calculated yet.")
        content_layout.addWidget(self.risk_metrics_text)
        
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        tab.setLayout(layout)
        return tab
    
    def _create_reports_tab(self):
        """Create the reports tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        placeholder_frame = QFrame()
        placeholder_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 40px;
            }
        """)
        placeholder_layout = QVBoxLayout()
        placeholder_frame.setLayout(placeholder_layout)
        
        title = QLabel("📋 Professional Reports")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 600;
                color: #1a1a1a;
                margin-bottom: 16px;
            }
        """)
        placeholder_layout.addWidget(title)
        
        description = QLabel("Generated PDF reports and export options will be available here.")
        description.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666666;
                text-align: center;
            }
        """)
        description.setAlignment(Qt.AlignCenter)
        placeholder_layout.addWidget(description)
        
        layout.addWidget(placeholder_frame)
        tab.setLayout(layout)
        return tab
    
    def _create_performance_tab(self):
        """Create the performance monitoring tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        
        placeholder_frame = QFrame()
        placeholder_frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 40px;
            }
        """)
        placeholder_layout = QVBoxLayout()
        placeholder_frame.setLayout(placeholder_layout)
        
        title = QLabel("⚡ Performance Monitoring")
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 600;
                color: #1a1a1a;
                margin-bottom: 16px;
            }
        """)
        placeholder_layout.addWidget(title)
        
        description = QLabel("GPU acceleration status, memory usage, and performance metrics will be shown here.")
        description.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666666;
                text-align: center;
            }
        """)
        description.setAlignment(Qt.AlignCenter)
        placeholder_layout.addWidget(description)
        
        layout.addWidget(placeholder_frame)
        tab.setLayout(layout)
        return tab
    
    def _create_results_panel(self):
        """Create results panel with comprehensive dashboard system."""
        from .widgets.visualization_dashboard import VisualizationDashboard, IndicatorPanel
        from .widgets.dashboard_system import DashboardSystem
        from PyQt5.QtWidgets import QSplitter, QTabWidget
        
        # Create main results widget with tabs
        results_widget = QTabWidget()
        
        # Tab 1: Advanced Visualization Dashboard
        visualization_tab = QWidget()
        viz_layout = QVBoxLayout()
        visualization_tab.setLayout(viz_layout)
        
        # Create visualization splitter
        viz_splitter = SplitterWidget('horizontal')
        viz_layout.addWidget(viz_splitter)
        
        # Create visualization dashboard
        self.visualization_dashboard = VisualizationDashboard()
        viz_splitter.add_widget(self.visualization_dashboard, 'visualization')
        
        # Create indicator panel
        self.indicator_panel = IndicatorPanel()
        viz_splitter.add_widget(self.indicator_panel, 'indicators')
        
        # Set sizes (dashboard should be larger)
        viz_splitter.set_sizes([4, 1])
        
        # Connect dashboard signals
        self.visualization_dashboard.chart_type_changed.connect(self._on_chart_type_changed)
        self.visualization_dashboard.layout_changed.connect(self._on_visualization_layout_changed)
        self.visualization_dashboard.data_exported.connect(self._on_chart_exported)
        
        # Connect indicator panel signals
        self.indicator_panel.indicator_added.connect(self._on_indicator_added)
        self.indicator_panel.indicator_removed.connect(self._on_indicator_removed)
        self.indicator_panel.indicator_updated.connect(self._on_indicator_updated)
        
        results_widget.addTab(visualization_tab, "Charts & Visualization")
        
        # Tab 2: Comprehensive Dashboard System
        self.dashboard_system = DashboardSystem()
        results_widget.addTab(self.dashboard_system, "Portfolio & Risk Dashboard")
        
        # Connect dashboard system signals
        self.dashboard_system.portfolio_dashboard.portfolio_updated.connect(self._on_portfolio_updated)
        self.dashboard_system.portfolio_dashboard.alert_triggered.connect(self._on_dashboard_alert)
        self.dashboard_system.risk_dashboard.risk_alert.connect(self._on_risk_alert)
        
        # Add some basic content to the tabs to make them visible
        results_widget.addTab(QWidget(), "Dashboard")
        results_widget.addTab(QWidget(), "Analysis")
        results_widget.addTab(QWidget(), "Reports")
        
        # Add a simple label to the first tab
        first_tab = results_widget.widget(0)
        tab_layout = QVBoxLayout()
        tab_layout.addWidget(QLabel("Welcome to the Quantitative Framework"))
        tab_layout.addWidget(QLabel("Select parameters in the control panel and run analysis"))
        first_tab.setLayout(tab_layout)
        
        # Wrap the QTabWidget in a ContainerWidget
        from .widgets.base_widget import ContainerWidget
        results_container = ContainerWidget()
        container_layout = QVBoxLayout()
        container_layout.addWidget(results_widget)
        results_container.setLayout(container_layout)
        
        return results_container
    
    def _setup_connections(self):
        """Setup signal connections."""
        # Connect control panel signals
        self.control_panel.analysis_requested.connect(self._on_analysis_requested)
        self.control_panel.export_requested.connect(self._on_export_requested)
        self.control_panel.clear_requested.connect(self._on_clear_requested)
        self.control_panel.settings_changed.connect(self._on_settings_changed)
        
        # Connect model signals
        self.analysis_model.analysis_completed.connect(self._on_model_analysis_completed)
        self.analysis_model.error_occurred.connect(self._on_model_error)
        self.analysis_model.loading_started.connect(self._on_loading_started)
        self.analysis_model.loading_finished.connect(self._on_loading_finished)
        
        # Connect theme and layout changes
        theme_manager.theme_changed.connect(self._on_theme_changed)
        layout_manager.layout_changed.connect(self._on_layout_changed)
    
    def _apply_initial_settings(self):
        """Apply initial application settings."""
        # Auto-detect layout
        layout_manager.auto_detect_layout()
        
        # Apply default theme (dark theme for better appearance)
        default_theme = self.settings_model.get_setting('theme', 'dark')
        theme_manager.set_theme(default_theme)
    
    def _on_analysis_requested(self, parameters: Dict[str, Any]):
        """Handle analysis request from control panel."""
        print(f"Analysis requested with parameters: {parameters}")  # Debug
        
        # Update UI state
        self.control_panel.set_loading_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Starting analysis...")
        
        self.analysis_model.set_parameters(parameters)
        self.analysis_worker.set_parameters(parameters)
        
        # Start analysis in worker thread
        if not self.analysis_thread.isRunning():
            self.analysis_thread.start()
        else:
            # Thread is already running, restart it
            self.analysis_thread.quit()
            self.analysis_thread.wait()
            self.analysis_thread.start()
    
    def _on_export_requested(self, format_type: str):
        """Handle export request."""
        results = self.analysis_model.analysis_results
        if results:
            # Implement export logic
            self.control_panel.log_message(f"Exporting results in {format_type} format...")
            # TODO: Implement actual export functionality
        else:
            self.control_panel.log_message("No results to export.")
    
    def _on_clear_requested(self):
        """Handle clear request."""
        self.analysis_model.clear_results()
        self.control_panel.log_message("Results cleared.")
    
    def _on_settings_changed(self, settings: Dict[str, Any]):
        """Handle settings changes."""
        self.settings_model.update_settings(settings)
    
    def _on_progress_updated(self, message: str, percentage: int):
        """Handle progress updates from worker."""
        self.control_panel.log_message(message)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(percentage)
        self.status_bar.showMessage(message)
        self.status_bar.showMessage(message)
    
    def _on_analysis_completed(self, results: Dict[str, Any]):
        """Handle completed analysis from worker."""
        self.analysis_model.set_analysis_results(results)
        self.control_panel.update_results(results)
        self.control_panel.log_message("=== Analysis Complete ===")
        
        # Update UI state
        self.progress_bar.setVisible(False)
        self.control_panel.set_loading_state(False)
        self.status_bar.showMessage("Analysis completed successfully")
        
        # Update results tabs with actual data
        self._update_results_tabs(results)
        
        # Switch to analysis tab to show results
        if hasattr(self, 'results_tab_widget'):
            self.results_tab_widget.setCurrentIndex(1)  # Switch to Analysis tab
        
        # Update results dashboard with results
        if hasattr(self, 'results_dashboard'):
            print(f"Updating results dashboard with data keys: {list(results.keys())}")  # Debug
            self.results_dashboard.update_data(results)
            # Switch to results tab to show the analysis
            self.results_tab_widget.setCurrentIndex(1)
            print("Results dashboard updated and switched to results tab")  # Debug
        else:
            print("ERROR: No results_dashboard attribute found!")  # Debug
        
        # Stop worker thread
        self.analysis_thread.quit()
    
    def _update_results_tabs(self, results: Dict[str, Any]):
        """Update the results tabs with analysis data."""
        try:
            # Update Analysis tab
            if hasattr(self, 'analysis_results_text'):
                analysis_text = "📈 ANALYSIS RESULTS\n" + "="*50 + "\n\n"
                
                # Market data info
                if 'market_data' in results and results['market_data'] is not None:
                    market_data = results['market_data']
                    analysis_text += f"📊 Market Data:\n"
                    analysis_text += f"   • Data points: {len(market_data)}\n"
                    analysis_text += f"   • Columns: {list(market_data.columns)}\n"
                    analysis_text += f"   • Date range: {market_data.index[0]} to {market_data.index[-1]}\n\n"
                
                # Signals info
                if 'signals' in results and results['signals'] is not None:
                    signals = results['signals']
                    analysis_text += f"🎯 Trading Signals:\n"
                    analysis_text += f"   • Signal columns: {list(signals.columns)}\n"
                    analysis_text += f"   • Total signals: {len(signals)}\n"
                    
                    # Show signal summary
                    for col in signals.columns:
                        if 'signal' in col.lower():
                            signal_counts = signals[col].value_counts()
                            analysis_text += f"   • {col}: {dict(signal_counts)}\n"
                    analysis_text += "\n"
                
                # Portfolio returns
                if 'portfolio_returns' in results and results['portfolio_returns'] is not None:
                    returns = results['portfolio_returns']
                    analysis_text += f"💰 Portfolio Returns:\n"
                    analysis_text += f"   • Total return periods: {len(returns)}\n"
                    analysis_text += f"   • Mean return: {returns.mean():.4f}\n"
                    analysis_text += f"   • Std deviation: {returns.std():.4f}\n"
                    analysis_text += f"   • Min return: {returns.min():.4f}\n"
                    analysis_text += f"   • Max return: {returns.max():.4f}\n\n"
                
                # Parameters used
                if 'parameters' in results:
                    params = results['parameters']
                    analysis_text += f"⚙️ Analysis Parameters:\n"
                    for key, value in params.items():
                        analysis_text += f"   • {key}: {value}\n"
                
                self.analysis_results_text.setText(analysis_text)
            
            # Update Portfolio tab
            if hasattr(self, 'portfolio_weights_text') and hasattr(self, 'risk_metrics_text'):
                # Portfolio weights
                if 'portfolio_weights' in results and results['portfolio_weights']:
                    weights = results['portfolio_weights']
                    weights_text = "💼 PORTFOLIO WEIGHTS\n" + "="*30 + "\n\n"
                    for symbol, weight in weights.items():
                        weights_text += f"{symbol}: {weight:.4f} ({weight*100:.2f}%)\n"
                    self.portfolio_weights_text.setText(weights_text)
                
                # Risk metrics
                if 'risk_metrics' in results and results['risk_metrics']:
                    metrics = results['risk_metrics']
                    risk_text = "📊 RISK METRICS\n" + "="*20 + "\n\n"
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            risk_text += f"{metric}: {value:.4f}\n"
                        else:
                            risk_text += f"{metric}: {value}\n"
                    self.risk_metrics_text.setText(risk_text)
                    
        except Exception as e:
            print(f"Error updating results tabs: {e}")
            if hasattr(self, 'analysis_results_text'):
                self.analysis_results_text.setText(f"Error displaying results: {str(e)}")
    
    def _on_analysis_failed(self, error_message: str):
        """Handle failed analysis from worker."""
        self.analysis_model.emit_error(error_message)
        self.control_panel.log_message(f"Analysis failed: {error_message}")
        
        # Update UI state
        self.progress_bar.setVisible(False)
        self.control_panel.set_loading_state(False)
        self.status_bar.showMessage(f"Analysis failed: {error_message}")
        
        # Stop worker thread
        self.analysis_thread.quit()
    
    def _on_model_analysis_completed(self, results: Dict[str, Any]):
        """Handle analysis completion from model."""
        # Update UI components with results
        pass
    
    def _on_model_error(self, error_message: str):
        """Handle model errors."""
        QMessageBox.critical(self, "Error", error_message)
    
    def _on_loading_started(self):
        """Handle loading start."""
        self.progress_bar.setVisible(True)
        self.control_panel.set_loading_state(True)
        self.status_bar.showMessage("Loading...")
    
    def _on_loading_finished(self):
        """Handle loading finish."""
        self.progress_bar.setVisible(False)
        self.control_panel.set_loading_state(False)
        self.status_bar.showMessage("Ready")
    
    def _on_theme_changed(self, theme_name: str):
        """Handle theme changes."""
        self.settings_model.set_setting('theme', theme_name)
        self.control_panel.log_message(f"Theme changed to: {theme_name}")
    
    def _on_layout_changed(self, layout_type: str):
        """Handle layout changes."""
        layout_manager.apply_layout(layout_type)
        self.control_panel.log_message(f"Layout changed to: {layout_type}")
    
    # Menu action handlers
    def _on_new_analysis(self):
        """Handle new analysis action."""
        self._on_clear_requested()
    
    def _on_open_file(self):
        """Handle open file action."""
        # TODO: Implement file opening
        pass
    
    def _on_save_file(self):
        """Handle save file action."""
        # TODO: Implement file saving
        pass
    
    def _on_export_results(self):
        """Handle export results action."""
        self._on_export_requested('csv')
    
    def _show_about_dialog(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Quantitative Research Framework",
            "Professional-grade quantitative finance platform\n"
            "Version 2.0\n\n"
            "Built with PyQt5 and modern architecture patterns."
        )
    
    def _on_chart_type_changed(self, chart_type: str):
        """Handle chart type change from visualization dashboard."""
        self.control_panel.log_message(f"Chart type changed to: {chart_type}")
    
    def _on_visualization_layout_changed(self, layout_type: str):
        """Handle visualization layout change."""
        self.control_panel.log_message(f"Visualization layout changed to: {layout_type}")
    
    def _on_chart_exported(self, chart_id: str, filename: str):
        """Handle chart export completion."""
        self.control_panel.log_message(f"Chart {chart_id} exported to: {filename}")
    
    def _on_indicator_added(self, indicator_name: str, parameters: Dict[str, Any]):
        """Handle technical indicator addition."""
        self.control_panel.log_message(f"Added indicator: {indicator_name} with parameters: {parameters}")
        
        # Calculate and add indicator data to visualization
        if hasattr(self, 'visualization_dashboard') and self.analysis_model.analysis_results:
            indicator_data = self._calculate_indicator(indicator_name, parameters)
            if indicator_data is not None:
                # Add indicator to current data
                current_data = self.analysis_model.analysis_results.copy()
                current_data[f'{indicator_name.lower()}_data'] = indicator_data
                self.visualization_dashboard.update_data(current_data)
    
    def _on_indicator_removed(self, indicator_name: str):
        """Handle technical indicator removal."""
        self.control_panel.log_message(f"Removed indicator: {indicator_name}")
    
    def _on_indicator_updated(self, indicator_name: str, parameters: Dict[str, Any]):
        """Handle technical indicator parameter update."""
        self.control_panel.log_message(f"Updated indicator {indicator_name}: {parameters}")
        
        # Recalculate and update indicator
        self._on_indicator_added(indicator_name, parameters)
    
    def _calculate_indicator(self, indicator_name: str, parameters: Dict[str, Any]) -> Optional[pd.Series]:
        """Calculate technical indicator data."""
        try:
            # Get market data
            results = self.analysis_model.analysis_results
            if not results or 'market_data' not in results:
                return None
            
            market_data = results['market_data']
            if market_data.empty or 'close' not in market_data.columns:
                return None
            
            close_prices = market_data['close']
            
            # Calculate indicator based on type
            if indicator_name == 'SMA':
                period = parameters.get('period', 20)
                return close_prices.rolling(window=period).mean()
            
            elif indicator_name == 'EMA':
                period = parameters.get('period', 20)
                return close_prices.ewm(span=period).mean()
            
            elif indicator_name == 'RSI':
                period = parameters.get('period', 14)
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            elif indicator_name == 'Bollinger Bands':
                period = parameters.get('period', 20)
                std_dev = parameters.get('std', 2)
                sma = close_prices.rolling(window=period).mean()
                std = close_prices.rolling(window=period).std()
                # Return middle band (SMA) for now
                return sma
            
            elif indicator_name == 'MACD':
                fast = parameters.get('fast', 12)
                slow = parameters.get('slow', 26)
                ema_fast = close_prices.ewm(span=fast).mean()
                ema_slow = close_prices.ewm(span=slow).mean()
                return ema_fast - ema_slow
            
            elif indicator_name == 'Volume':
                if 'volume' in market_data.columns:
                    return market_data['volume']
            
            return None
            
        except Exception as e:
            self.control_panel.log_message(f"Error calculating {indicator_name}: {str(e)}")
            return None
    
    def _on_portfolio_updated(self, portfolio_data: Dict[str, Any]):
        """Handle portfolio dashboard updates."""
        self.control_panel.log_message("Portfolio dashboard updated")
    
    def _on_dashboard_alert(self, alert_type: str, message: str):
        """Handle dashboard alerts."""
        self.control_panel.log_message(f"ALERT [{alert_type}]: {message}")
        self.status_bar.showMessage(f"Alert: {message}", 10000)  # Show for 10 seconds
    
    def _on_risk_alert(self, alert_type: str, message: str, severity: float):
        """Handle risk alerts."""
        severity_text = "HIGH" if severity > 0.8 else "MEDIUM" if severity > 0.5 else "LOW"
        self.control_panel.log_message(f"RISK ALERT [{severity_text}] {alert_type}: {message}")
        
        # Show critical alerts in status bar
        if severity > 0.8:
            self.status_bar.showMessage(f"CRITICAL RISK ALERT: {message}", 15000)
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop worker thread if running
        if self.analysis_thread.isRunning():
            self.analysis_thread.quit()
            self.analysis_thread.wait()
        
        event.accept()


def create_application():
    """Create and configure the QApplication with macOS native styling."""
    app = QApplication(sys.argv)
    app.setApplicationName("Quantitative Research Framework")
    app.setApplicationDisplayName("Quantitative Research Framework")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("QuantFramework")
    app.setOrganizationDomain("quantframework.com")
    
    # Set macOS specific properties
    app.setAttribute(Qt.AA_DontShowIconsInMenus, False)
    app.setAttribute(Qt.AA_NativeWindows, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Use native macOS style
    if sys.platform == "darwin":  # macOS
        app.setStyle('macintosh')
    else:
        app.setStyle('Fusion')
    
    # Set application font to system font
    system_font = QFont()
    if sys.platform == "darwin":  # macOS
        system_font.setFamily(".AppleSystemUIFont")  # macOS system font
    else:
        system_font.setFamily("Segoe UI")  # Windows/Linux
    system_font.setPointSize(13)
    app.setFont(system_font)
    
    return app


def main():
    """Main application entry point."""
    app = create_application()
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())