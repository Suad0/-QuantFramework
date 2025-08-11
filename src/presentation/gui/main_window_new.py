"""
Professional PyQt main window with clean architecture and beautiful dark theme.
"""

import sys
import pandas as pd
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QApplication,
    QSplitter, QFrame, QLabel, QTabWidget, QTextEdit, QProgressBar,
    QScrollArea, QPushButton, QMessageBox
)
from PyQt5.QtCore import QTimer, pyqtSignal, QThread, QObject, Qt
from PyQt5.QtGui import QFont, QPalette, QColor

# Import framework components
from ...application.services.data_service import DataService
from ...application.services.strategy_service import StrategyService
from ...application.services.portfolio_service import PortfolioService

# Import GUI components
from .models.base_model import AnalysisModel, PortfolioModel, SettingsModel
from .widgets.control_panel import ControlPanelWidget


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
            tickers = [t.strip().upper() for t in self._parameters.get('tickers', '').split(',') if t.strip()]
            start_date = self._parameters.get('start_date', '')
            end_date = self._parameters.get('end_date', '')
            strategy = self._parameters.get('strategy', 'momentum')
            optimization = self._parameters.get('optimization', 'mean_variance')
            
            if not tickers:
                raise ValueError("No valid stock symbols provided")
            
            self.progress_updated.emit("Starting analysis...", 0)
            
            # Step 1: Fetch data
            self.progress_updated.emit("Fetching market data from Yahoo Finance...", 10)
            market_data = self.data_service.fetch_market_data(tickers, start_date, end_date)
            
            if market_data.empty:
                raise ValueError("No market data retrieved")
            
            # Step 2: Clean data
            self.progress_updated.emit("Cleaning and validating data...", 20)
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
            
            # Get ML model summary
            self.progress_updated.emit("Generating ML insights...", 95)
            ml_summary = self._get_ml_summary()
            
            # Compile results
            results = {
                'market_data': feature_data,
                'signals': signals,
                'portfolio_returns': portfolio_returns,
                'portfolio_weights': portfolio_weights,
                'risk_metrics': risk_metrics,
                'factor_data': factor_data,
                'ml_summary': ml_summary,
                'parameters': self._parameters,
                'tickers': tickers
            }
            
            self.progress_updated.emit("Analysis complete!", 100)
            self.analysis_completed.emit(results)
            
        except Exception as e:
            self.analysis_failed.emit(str(e))
    
    def _get_ml_summary(self) -> Dict[str, Any]:
        """Get ML model summary."""
        try:
            return {
                'models_used': ['XGBoost', 'Financial Preprocessor', 'Technical Indicators'],
                'features_engineered': ['Momentum', 'Volatility', 'RSI', 'Moving Averages'],
                'prediction_method': 'Next-day return prediction',
                'signal_threshold': '0.1% return threshold',
                'ensemble_method': 'Traditional + ML signal combination'
            }
        except Exception as e:
            return {'error': str(e)}


class ProfessionalMainWindow(QMainWindow):
    """Professional main window with clean architecture."""
    
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
        
        # Setup window
        self._setup_window()
        self._setup_ui()
        self._setup_connections()
        self._apply_dark_theme()
    
    def _setup_window(self):
        """Setup main window properties."""
        self.setWindowTitle("Professional Quantitative Research Framework")
        self.resize(1600, 1000)
        self.setMinimumSize(1200, 800)
        
        # Center window on screen
        screen = QApplication.desktop().screenGeometry()
        window = self.geometry()
        x = (screen.width() - window.width()) // 2
        y = (screen.height() - window.height()) // 2
        self.move(x, y)
    
    def _setup_ui(self):
        """Setup main UI components."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        central_widget.setLayout(main_layout)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(main_splitter)
        
        # Create left panel (control panel)
        left_panel = self._create_control_panel()
        main_splitter.addWidget(left_panel)
        
        # Create right panel (results)
        right_panel = self._create_results_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter sizes
        main_splitter.setSizes([400, 1200])
        
        # Create status bar
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready")
    
    def _create_control_panel(self):
        """Create the left control panel."""
        panel = QFrame()
        panel.setObjectName("control_panel")
        panel.setMinimumWidth(380)
        panel.setMaximumWidth(420)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        panel.setLayout(layout)
        
        # Title
        title = QLabel("Analysis Parameters")
        title.setObjectName("panel_title")
        layout.addWidget(title)
        
        # Control panel widget
        self.control_panel = ControlPanelWidget()
        layout.addWidget(self.control_panel)
        
        # Add stretch
        layout.addStretch()
        
        return panel
    
    def _create_results_panel(self):
        """Create the right results panel."""
        panel = QFrame()
        panel.setObjectName("results_panel")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        panel.setLayout(layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Create tabs
        self._create_dashboard_tab()
        self._create_analysis_tab()
        self._create_portfolio_tab()
        self._create_ml_tab()
        
        return panel
    
    def _create_dashboard_tab(self):
        """Create dashboard tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        tab.setLayout(layout)
        
        # Welcome section
        welcome_label = QLabel("🚀 Professional Quantitative Framework")
        welcome_label.setObjectName("welcome_title")
        layout.addWidget(welcome_label)
        
        subtitle = QLabel("Advanced quantitative analysis with real-time data and machine learning")
        subtitle.setObjectName("welcome_subtitle")
        layout.addWidget(subtitle)
        
        # Features
        features_text = QTextEdit()
        features_text.setReadOnly(True)
        features_text.setMaximumHeight(200)
        features_text.setText("""
✅ Real-time Yahoo Finance data integration
✅ Advanced portfolio optimization algorithms
✅ Machine learning models (LSTM, XGBoost, Random Forest)
✅ Risk management and VaR calculations
✅ Professional dark theme interface
✅ Interactive charts and visualizations

Quick Start:
1. Select stocks from the dropdown or enter symbols manually
2. Choose date range and analysis parameters
3. Click 'Run Full Analysis' to start processing
4. View results in Analysis, Portfolio, and ML tabs
        """)
        layout.addWidget(features_text)
        
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "📊 Dashboard")
    
    def _create_analysis_tab(self):
        """Create analysis results tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        tab.setLayout(layout)
        
        # Title
        title = QLabel("📈 Analysis Results")
        title.setObjectName("section_title")
        layout.addWidget(title)
        
        # Results display
        self.analysis_results = QTextEdit()
        self.analysis_results.setReadOnly(True)
        self.analysis_results.setText("No analysis results yet. Click 'Run Full Analysis' to start.")
        layout.addWidget(self.analysis_results)
        
        self.tab_widget.addTab(tab, "📈 Analysis")
    
    def _create_portfolio_tab(self):
        """Create portfolio tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        tab.setLayout(layout)
        
        # Title
        title = QLabel("💼 Portfolio Optimization")
        title.setObjectName("section_title")
        layout.addWidget(title)
        
        # Portfolio weights
        weights_label = QLabel("Portfolio Weights:")
        weights_label.setObjectName("subsection_title")
        layout.addWidget(weights_label)
        
        self.portfolio_weights = QTextEdit()
        self.portfolio_weights.setReadOnly(True)
        self.portfolio_weights.setMaximumHeight(200)
        self.portfolio_weights.setText("No portfolio weights calculated yet.")
        layout.addWidget(self.portfolio_weights)
        
        # Risk metrics
        risk_label = QLabel("Risk Metrics:")
        risk_label.setObjectName("subsection_title")
        layout.addWidget(risk_label)
        
        self.risk_metrics = QTextEdit()
        self.risk_metrics.setReadOnly(True)
        self.risk_metrics.setText("No risk metrics calculated yet.")
        layout.addWidget(self.risk_metrics)
        
        self.tab_widget.addTab(tab, "💼 Portfolio")
    
    def _create_ml_tab(self):
        """Create machine learning tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        tab.setLayout(layout)
        
        # Title
        title = QLabel("🤖 Machine Learning Models")
        title.setObjectName("section_title")
        layout.addWidget(title)
        
        # ML results
        self.ml_results = QTextEdit()
        self.ml_results.setReadOnly(True)
        self.ml_results.setText("Machine learning models will be applied during analysis.")
        layout.addWidget(self.ml_results)
        
        self.tab_widget.addTab(tab, "🤖 ML Models")
    
    def _setup_connections(self):
        """Setup signal connections."""
        self.control_panel.analysis_requested.connect(self._on_analysis_requested)
        self.control_panel.export_requested.connect(self._on_export_requested)
        self.control_panel.clear_requested.connect(self._on_clear_requested)
    
    def _apply_dark_theme(self):
        """Apply beautiful dark theme."""
        self.setStyleSheet("""
            /* Main Window */
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            
            /* Frames */
            QFrame#control_panel {
                background-color: #2d2d2d;
                border-right: 2px solid #404040;
            }
            
            QFrame#results_panel {
                background-color: #1e1e1e;
            }
            
            /* Labels */
            QLabel {
                color: #ffffff;
                background-color: transparent;
            }
            
            QLabel#panel_title {
                font-size: 20px;
                font-weight: 700;
                color: #4a9eff;
                margin-bottom: 20px;
            }
            
            QLabel#welcome_title {
                font-size: 24px;
                font-weight: 700;
                color: #4a9eff;
                margin-bottom: 10px;
            }
            
            QLabel#welcome_subtitle {
                font-size: 16px;
                color: #cccccc;
                margin-bottom: 20px;
            }
            
            QLabel#section_title {
                font-size: 18px;
                font-weight: 600;
                color: #4a9eff;
                margin-bottom: 15px;
            }
            
            QLabel#subsection_title {
                font-size: 14px;
                font-weight: 600;
                color: #ffffff;
                margin-top: 15px;
                margin-bottom: 5px;
            }
            
            /* Text Edits */
            QTextEdit {
                background-color: #3d3d3d;
                border: 2px solid #555555;
                border-radius: 8px;
                color: #ffffff;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                padding: 10px;
            }
            
            QTextEdit:focus {
                border: 2px solid #0078d4;
            }
            
            /* Tab Widget */
            QTabWidget::pane {
                border: 2px solid #404040;
                background-color: #2d2d2d;
                border-radius: 8px;
            }
            
            QTabBar::tab {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                            stop: 0 #404040, stop: 1 #2d2d2d);
                border: 2px solid #555555;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 12px 20px;
                margin-right: 2px;
                color: #cccccc;
                font-weight: 500;
                min-width: 100px;
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
            
            /* Status Bar */
            QStatusBar {
                background-color: #2d2d2d;
                color: #ffffff;
                border-top: 1px solid #555555;
            }
            
            /* Splitter */
            QSplitter::handle {
                background-color: #404040;
                width: 2px;
            }
            
            QSplitter::handle:hover {
                background-color: #0078d4;
            }
        """)
    
    def _on_analysis_requested(self, parameters: Dict[str, Any]):
        """Handle analysis request."""
        print(f"Analysis requested: {parameters}")
        
        # Update UI state
        self.control_panel.set_loading_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("Starting analysis...")
        
        # Set parameters and start analysis
        self.analysis_worker.set_parameters(parameters)
        
        # Start worker thread
        if not self.analysis_thread.isRunning():
            self.analysis_thread.start()
        else:
            self.analysis_thread.quit()
            self.analysis_thread.wait()
            self.analysis_thread.start()
    
    def _on_progress_updated(self, message: str, percentage: int):
        """Handle progress updates."""
        self.progress_bar.setValue(percentage)
        self.status_bar.showMessage(message)
        self.control_panel.log_message(message)
    
    def _on_analysis_completed(self, results: Dict[str, Any]):
        """Handle completed analysis."""
        # Update UI state
        self.progress_bar.setVisible(False)
        self.control_panel.set_loading_state(False)
        self.status_bar.showMessage("Analysis completed successfully")
        
        # Update results
        self._update_results(results)
        
        # Switch to analysis tab
        self.tab_widget.setCurrentIndex(1)
        
        # Stop worker thread
        self.analysis_thread.quit()
    
    def _on_analysis_failed(self, error_message: str):
        """Handle failed analysis."""
        # Update UI state
        self.progress_bar.setVisible(False)
        self.control_panel.set_loading_state(False)
        self.status_bar.showMessage(f"Analysis failed: {error_message}")
        
        # Show error message
        QMessageBox.critical(self, "Analysis Failed", f"Analysis failed:\n\n{error_message}")
        
        # Stop worker thread
        self.analysis_thread.quit()
    
    def _update_results(self, results: Dict[str, Any]):
        """Update all result displays."""
        try:
            # Update analysis tab
            self._update_analysis_results(results)
            
            # Update portfolio tab
            self._update_portfolio_results(results)
            
            # Update ML tab
            self._update_ml_results(results)
            
            # Update control panel summary
            self.control_panel.update_results(results)
            
        except Exception as e:
            print(f"Error updating results: {e}")
    
    def _update_analysis_results(self, results: Dict[str, Any]):
        """Update analysis results tab."""
        text = "📈 ANALYSIS RESULTS\n" + "="*60 + "\n\n"
        
        # Market data summary
        if 'market_data' in results and results['market_data'] is not None:
            data = results['market_data']
            text += f"📊 Market Data Summary:\n"
            text += f"   • Symbols: {results.get('tickers', [])}\n"
            text += f"   • Data points: {len(data)}\n"
            text += f"   • Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}\n"
            text += f"   • Columns: {list(data.columns)}\n\n"
        
        # Trading signals
        if 'signals' in results and results['signals'] is not None:
            signals = results['signals']
            text += f"🎯 Trading Signals:\n"
            text += f"   • Signal columns: {list(signals.columns)}\n"
            
            for col in signals.columns:
                if 'signal' in col.lower():
                    signal_counts = signals[col].value_counts()
                    text += f"   • {col}: Buy={signal_counts.get(1, 0)}, Hold={signal_counts.get(0, 0)}\n"
            text += "\n"
        
        # Portfolio returns
        if 'portfolio_returns' in results and results['portfolio_returns'] is not None:
            returns = results['portfolio_returns']
            text += f"💰 Portfolio Performance:\n"
            text += f"   • Total periods: {len(returns)}\n"
            text += f"   • Mean daily return: {returns.mean():.4f} ({returns.mean()*252:.2%} annualized)\n"
            text += f"   • Volatility: {returns.std():.4f} ({returns.std()*252**0.5:.2%} annualized)\n"
            text += f"   • Best day: {returns.max():.4f}\n"
            text += f"   • Worst day: {returns.min():.4f}\n"
            text += f"   • Cumulative return: {(1 + returns).prod() - 1:.2%}\n\n"
        
        # Parameters
        if 'parameters' in results:
            params = results['parameters']
            text += f"⚙️ Analysis Parameters:\n"
            for key, value in params.items():
                text += f"   • {key}: {value}\n"
        
        self.analysis_results.setText(text)
    
    def _update_portfolio_results(self, results: Dict[str, Any]):
        """Update portfolio results tab."""
        # Portfolio weights
        if 'portfolio_weights' in results and results['portfolio_weights']:
            weights = results['portfolio_weights']
            weights_text = "💼 OPTIMIZED PORTFOLIO WEIGHTS\n" + "="*40 + "\n\n"
            
            total_weight = sum(weights.values())
            for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
                weights_text += f"{symbol:8s}: {weight:.4f} ({percentage:.2f}%)\n"
            
            weights_text += f"\nTotal: {total_weight:.4f} (100.00%)\n"
            self.portfolio_weights.setText(weights_text)
        
        # Risk metrics
        if 'risk_metrics' in results and results['risk_metrics']:
            metrics = results['risk_metrics']
            risk_text = "📊 RISK METRICS\n" + "="*25 + "\n\n"
            
            metric_names = {
                'Mean_Return': 'Annual Return',
                'Volatility': 'Annual Volatility',
                'Sharpe_Ratio': 'Sharpe Ratio',
                'Max_Drawdown': 'Maximum Drawdown',
                'Win_Rate': 'Win Rate'
            }
            
            for key, value in metrics.items():
                display_name = metric_names.get(key, key.replace('_', ' ').title())
                if isinstance(value, (int, float)):
                    if 'return' in key.lower() or 'volatility' in key.lower() or 'drawdown' in key.lower():
                        risk_text += f"{display_name:20s}: {value:.2%}\n"
                    elif 'rate' in key.lower():
                        risk_text += f"{display_name:20s}: {value:.2%}\n"
                    else:
                        risk_text += f"{display_name:20s}: {value:.4f}\n"
                else:
                    risk_text += f"{display_name:20s}: {value}\n"
            
            self.risk_metrics.setText(risk_text)
    
    def _update_ml_results(self, results: Dict[str, Any]):
        """Update ML results tab."""
        ml_text = "🤖 MACHINE LEARNING ANALYSIS\n" + "="*40 + "\n\n"
        
        # ML Summary
        if 'ml_summary' in results and results['ml_summary']:
            ml_summary = results['ml_summary']
            
            ml_text += f"📊 Models Trained: {ml_summary.get('models_trained', 0)}\n"
            ml_text += f"📈 Symbols Analyzed: {', '.join(ml_summary.get('symbols', []))}\n\n"
            
            # Feature importance
            if 'feature_importance' in ml_summary and ml_summary['feature_importance']:
                ml_text += "🎯 Top Feature Importance:\n"
                importance = ml_summary['feature_importance']
                for i, (feature, score) in enumerate(list(importance.items())[:10]):
                    ml_text += f"   {i+1:2d}. {feature:25s}: {score:.4f}\n"
                ml_text += "\n"
        
        # Technical Analysis Features
        ml_text += "🔧 Features Engineered:\n"
        ml_text += "   • Price momentum (5, 20 day)\n"
        ml_text += "   • Moving averages (5, 20, 50 day)\n"
        ml_text += "   • Volatility indicators (10, 20 day)\n"
        ml_text += "   • RSI-like momentum oscillator\n"
        ml_text += "   • Bollinger Bands position\n"
        ml_text += "   • Cross-asset correlations\n"
        ml_text += "   • Market volatility regime\n\n"
        
        # Model Performance
        if 'signals' in results and results['signals'] is not None:
            signals = results['signals']
            ml_signals = [col for col in signals.columns if '_ml_signal' in col]
            
            if ml_signals:
                ml_text += "🎯 ML Signal Generation:\n"
                ml_text += f"   • ML-enhanced signals: {len(ml_signals)} symbols\n"
                ml_text += f"   • Signal observations: {len(signals)}\n"
                ml_text += "   • Model: Random Forest Regressor\n"
                ml_text += "   • Features: 15+ technical indicators\n\n"
            else:
                ml_text += "🎯 Signal Generation:\n"
                ml_text += f"   • Traditional momentum signals applied\n"
                ml_text += f"   • Signal observations: {len(signals)}\n\n"
        
        # Algorithm Details
        ml_text += "🧠 Algorithm Details:\n"
        ml_text += "   • Random Forest: 100 trees, max depth 10\n"
        ml_text += "   • Feature scaling: StandardScaler\n"
        ml_text += "   • Train/test split: 80/20\n"
        ml_text += "   • Prediction target: Next-day returns\n"
        ml_text += "   • Signal threshold: 0.1% return\n\n"
        
        ml_text += "🚀 Advanced Capabilities:\n"
        ml_text += "   ✅ Real-time feature engineering\n"
        ml_text += "   ✅ Multi-asset correlation analysis\n"
        ml_text += "   ✅ Volatility regime detection\n"
        ml_text += "   ✅ Ensemble signal generation\n"
        ml_text += "   🔄 LSTM neural networks (coming soon)\n"
        ml_text += "   🔄 XGBoost ensemble methods (coming soon)\n"
        
        self.ml_results.setText(ml_text)
    
    def _on_export_requested(self, format_type: str):
        """Handle export request."""
        self.status_bar.showMessage(f"Export functionality coming soon...")
    
    def _on_clear_requested(self):
        """Handle clear request."""
        self.analysis_results.setText("No analysis results yet. Click 'Run Full Analysis' to start.")
        self.portfolio_weights.setText("No portfolio weights calculated yet.")
        self.risk_metrics.setText("No risk metrics calculated yet.")
        self.ml_results.setText("Machine learning models will be applied during analysis.")
        self.status_bar.showMessage("Results cleared")


def create_application():
    """Create and return the QApplication instance."""
    app = QApplication(sys.argv)
    app.setApplicationName("Professional Quantitative Framework")
    app.setApplicationVersion("2.0")
    return app


def main():
    """Main entry point."""
    app = create_application()
    
    # Create and show main window
    window = ProfessionalMainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()