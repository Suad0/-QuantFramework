import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QDateEdit, QComboBox, QTextEdit,
    QSplitter, QGroupBox
)
from PyQt5.QtCore import QDate, Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import warnings

# Import fixed modules
from data_pipeline import DataPipeline
from strategy import Strategy
from portfolio import PortfolioOptimizer
from risk import RiskAnalyzer

warnings.filterwarnings('ignore')


class QuantWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quantitative Research Framework")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize framework components
        self.data_pipeline = DataPipeline()
        self.strategy = Strategy()
        self.optimizer = PortfolioOptimizer()
        self.risk_analyzer = RiskAnalyzer()

        # Data storage
        self.current_data = None
        self.current_signals = None
        self.portfolio_returns = None

        # Setup UI
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(main_splitter)

        # Left panel: Controls
        left_panel = self.create_control_panel()
        main_splitter.addWidget(left_panel)

        # Right panel: Plots and Results
        right_panel = self.create_results_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([400, 1000])

    def create_control_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Input Group
        input_group = QGroupBox("Input Parameters")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        # Tickers input
        input_layout.addWidget(QLabel("Tickers (comma-separated):"))
        self.tickers_input = QLineEdit("AAPL,MSFT,GOOGL")
        input_layout.addWidget(self.tickers_input)

        # Date inputs
        input_layout.addWidget(QLabel("Start Date:"))
        self.start_date = QDateEdit(QDate(2023, 1, 1))
        self.start_date.setCalendarPopup(True)
        input_layout.addWidget(self.start_date)

        input_layout.addWidget(QLabel("End Date:"))
        self.end_date = QDateEdit(QDate(2024, 12, 31))
        self.end_date.setCalendarPopup(True)
        input_layout.addWidget(self.end_date)

        # Optimization method
        input_layout.addWidget(QLabel("Optimization Method:"))
        self.optimization_method = QComboBox()
        self.optimization_method.addItems(['mean_variance', 'equal_weight', 'risk_parity'])
        input_layout.addWidget(self.optimization_method)

        layout.addWidget(input_group)

        # Control buttons
        button_group = QGroupBox("Controls")
        button_layout = QVBoxLayout()
        button_group.setLayout(button_layout)

        run_button = QPushButton("Run Full Analysis")
        run_button.clicked.connect(self.run_analysis)
        run_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(run_button)

        clear_button = QPushButton("Clear Results")
        clear_button.clicked.connect(self.clear_results)
        button_layout.addWidget(clear_button)

        layout.addWidget(button_group)

        # Results display
        results_group = QGroupBox("Results Summary")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)

        self.weights_label = QLabel("Portfolio Weights: None")
        self.weights_label.setWordWrap(True)
        results_layout.addWidget(self.weights_label)

        self.performance_label = QLabel("Performance: None")
        self.performance_label.setWordWrap(True)
        results_layout.addWidget(self.performance_label)

        layout.addWidget(results_group)

        # Console output
        console_group = QGroupBox("Console Output")
        console_layout = QVBoxLayout()
        console_group.setLayout(console_layout)

        self.console_output = QTextEdit()
        self.console_output.setMaximumHeight(150)
        self.console_output.setReadOnly(True)
        console_layout.addWidget(self.console_output)

        layout.addWidget(console_group)

        layout.addStretch()
        return panel

    def create_results_panel(self):
        """Create the right results panel with plots."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Create matplotlib figure with subplots
        self.figure = Figure(figsize=(12, 10))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create subplots
        self.ax1 = self.figure.add_subplot(2, 2, 1)  # Returns
        self.ax2 = self.figure.add_subplot(2, 2, 2)  # Risk metrics
        self.ax3 = self.figure.add_subplot(2, 2, 3)  # Technical indicator
        self.ax4 = self.figure.add_subplot(2, 2, 4)  # Portfolio weights

        self.figure.tight_layout(pad=3.0)
        return panel

    def log_message(self, message):
        """Add message to console output."""
        self.console_output.append(message)
        self.console_output.ensureCursorVisible()

    def clear_results(self):
        """Clear all results and plots."""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.canvas.draw()

        self.weights_label.setText("Portfolio Weights: None")
        self.performance_label.setText("Performance: None")
        self.console_output.clear()

        self.current_data = None
        self.current_signals = None
        self.portfolio_returns = None

    def run_analysis(self):
        """Run the complete quantitative analysis pipeline."""
        try:
            self.log_message("=== Starting Analysis ===")

            # Get input parameters
            tickers = [t.strip().upper() for t in self.tickers_input.text().split(',')]
            start_date = self.start_date.date().toString("yyyy-MM-dd")
            end_date = self.end_date.date().toString("yyyy-MM-dd")
            opt_method = self.optimization_method.currentText()

            self.log_message(f"Analyzing tickers: {tickers}")
            self.log_message(f"Date range: {start_date} to {end_date}")

            # Step 1: Fetch and process data
            self.log_message("Step 1: Fetching stock data...")
            df = self.data_pipeline.fetch_data(tickers, start_date, end_date)

            self.log_message("Step 2: Cleaning data...")
            df = self.data_pipeline.clean_data(df)

            self.log_message("Step 3: Engineering features...")
            df = self.data_pipeline.engineer_features(df)

            self.current_data = df

            # Step 4: Generate trading signals
            self.log_message("Step 4: Generating trading signals...")
            signals = self.strategy.generate_signals(df)
            self.current_signals = signals

            # Step 5: Backtest strategy
            self.log_message("Step 5: Backtesting strategy...")
            portfolio_returns = self.strategy.backtest(df, signals)
            self.portfolio_returns = portfolio_returns

            # Step 6: Optimize portfolio
            self.log_message(f"Step 6: Optimizing portfolio using {opt_method}...")
            weights = self.optimizer.optimize(df, method=opt_method)

            # Step 7: Calculate risk metrics
            self.log_message("Step 7: Calculating risk metrics...")
            risk_metrics = self.risk_analyzer.calculate_risk(portfolio_returns)

            # Step 8: Update visualizations
            self.log_message("Step 8: Updating visualizations...")
            self.update_plots(df, portfolio_returns, risk_metrics, weights, signals)

            # Update results display
            weights_text = ", ".join([f"{k}: {v:.3f}" for k, v in weights.items()])
            self.weights_label.setText(f"Portfolio Weights: {weights_text}")

            performance_text = f"Total Return: {risk_metrics.get('Mean_Return', 0):.2%}, " \
                               f"Volatility: {risk_metrics.get('Volatility', 0):.2%}, " \
                               f"Sharpe: {risk_metrics.get('Sharpe_Ratio', 0):.2f}"
            self.performance_label.setText(f"Performance: {performance_text}")

            self.log_message("=== Analysis Complete ===")

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.log_message(error_msg)
            print(error_msg)

            # Clear plots and show error
            self.ax1.clear()
            self.ax1.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=self.ax1.transAxes)
            self.ax1.set_title("Error in Analysis")
            self.canvas.draw()

    def update_plots(self, df, returns, risk_metrics, weights, signals):
        """Update all plots with analysis results."""
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        try:
            # Plot 1: Cumulative Returns
            if returns is not None and not returns.empty:
                cumulative_returns = (1 + returns).cumprod()
                cumulative_returns.plot(ax=self.ax1, linewidth=2)
                self.ax1.set_title("Portfolio Cumulative Returns", fontsize=12, fontweight='bold')
                self.ax1.set_xlabel("Date")
                self.ax1.set_ylabel("Cumulative Return")
                self.ax1.grid(True, alpha=0.3)

            # Plot 2: Risk Metrics Bar Chart
            if risk_metrics:
                # Select key metrics for display
                key_metrics = {k: v for k, v in risk_metrics.items()
                               if k in ['Volatility', 'Sharpe_Ratio', 'Max_Drawdown', 'VaR_95']}

                if key_metrics:
                    metrics_names = list(key_metrics.keys())
                    metrics_values = list(key_metrics.values())

                    bars = self.ax2.bar(metrics_names, metrics_values,
                                        color=['red' if v < 0 else 'green' for v in metrics_values])
                    self.ax2.set_title("Key Risk Metrics", fontsize=12, fontweight='bold')
                    self.ax2.set_ylabel("Value")
                    self.ax2.tick_params(axis='x', rotation=45)

                    # Add value labels on bars
                    for bar, value in zip(bars, metrics_values):
                        height = bar.get_height()
                        self.ax2.text(bar.get_x() + bar.get_width() / 2., height,
                                      f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')

            # Plot 3: Technical Indicator (RSI for first ticker)
            rsi_cols = [col for col in df.columns if 'RSI' in col]
            if rsi_cols:
                df[rsi_cols[0]].plot(ax=self.ax3, color='purple', linewidth=1)
                self.ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
                self.ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
                self.ax3.axhline(y=50, color='black', linestyle='-', alpha=0.5, label='Neutral (50)')
                self.ax3.set_title(f"RSI: {rsi_cols[0].replace('_RSI', '')}", fontsize=12, fontweight='bold')
                self.ax3.set_ylabel("RSI Value")
                self.ax3.set_ylim(0, 100)
                self.ax3.legend(fontsize=8)
                self.ax3.grid(True, alpha=0.3)

            # Plot 4: Portfolio Weights
            if weights is not None and not weights.empty:
                weights_plot = weights[weights > 0.001]  # Only show significant weights
                if not weights_plot.empty:
                    wedges, texts, autotexts = self.ax4.pie(weights_plot.values,
                                                            labels=weights_plot.index,
                                                            autopct='%1.1f%%',
                                                            startangle=90)
                    self.ax4.set_title("Portfolio Weights", fontsize=12, fontweight='bold')

                    # Improve text readability
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')

        except Exception as e:
            self.log_message(f"Error updating plots: {e}")

        # Adjust layout and refresh
        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = QuantWindow()
    window.show()

    sys.exit(app.exec_())
