import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QDateEdit, QComboBox, QTextEdit,
    QSplitter, QGroupBox, QCheckBox, QSpinBox, QProgressBar,
    QTabWidget, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import QDate, Qt, QThread, pyqtSignal
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
from lstm_model import NeuralNetworkStrategy, LSTMPredictor, xLSTMPredictor

warnings.filterwarnings('ignore')


class LSTMTrainingThread(QThread):
    """Background thread for LSTM training to prevent UI freezing."""
    progress_update = pyqtSignal(str)
    training_complete = pyqtSignal(object, dict)
    training_error = pyqtSignal(str)

    def __init__(self, df, model_type, tickers, sequence_length, epochs):
        super().__init__()
        self.df = df
        self.model_type = model_type
        self.tickers = tickers
        self.sequence_length = sequence_length
        self.epochs = epochs

    def run(self):
        try:
            self.progress_update.emit("Initializing neural network strategy...")

            # Initialize neural network strategy
            nn_strategy = NeuralNetworkStrategy(
                model_type=self.model_type,
                sequence_length=self.sequence_length,
                lstm_units=128,
                dropout_rate=0.3,
                learning_rate=0.001
            )

            self.progress_update.emit("Training LSTM models...")

            # Train models for each ticker
            nn_strategy.train_models(self.df, self.tickers)

            self.progress_update.emit("Generating neural network signals...")

            # Generate signals
            neural_signals = nn_strategy.generate_neural_signals(self.df)

            # Calculate performance metrics for each model
            performance_metrics = {}
            for ticker, model in nn_strategy.models.items():
                if model.is_trained:
                    target_col = f'{ticker}_Adj Close' if f'{ticker}_Adj Close' in self.df.columns else 'Adj Close'
                    try:
                        # Get recent predictions for evaluation
                        feature_cols = model.feature_columns
                        df_clean = self.df[feature_cols + [target_col]].dropna()

                        if len(df_clean) >= model.sequence_length:
                            X_scaled = model.scaler_features.transform(df_clean[feature_cols])

                            # Create sequences for the last portion of data
                            test_size = min(100, len(X_scaled) // 4)  # Use last 25% or 100 points
                            X_test = X_scaled[-test_size - model.sequence_length:]
                            y_test = df_clean[target_col].iloc[-test_size - model.sequence_length:].values

                            X_sequences = []
                            y_sequences = []
                            for i in range(model.sequence_length, len(X_test)):
                                X_sequences.append(X_test[i - model.sequence_length:i])
                                y_sequences.append(y_test[i])

                            if X_sequences:
                                X_sequences = np.array(X_sequences)
                                y_sequences = np.array(y_sequences)

                                predictions = model.predict(X_sequences)

                                # Calculate metrics
                                mse = np.mean((predictions - y_sequences) ** 2)
                                mae = np.mean(np.abs(predictions - y_sequences))

                                # Direction accuracy
                                actual_direction = np.diff(y_sequences) > 0
                                predicted_direction = np.diff(predictions) > 0
                                direction_accuracy = np.mean(actual_direction == predicted_direction) * 100

                                performance_metrics[ticker] = {
                                    'MSE': mse,
                                    'MAE': mae,
                                    'Direction_Accuracy': direction_accuracy,
                                    'Predictions_Count': len(predictions)
                                }
                    except Exception as e:
                        self.progress_update.emit(f"Error evaluating {ticker}: {str(e)}")
                        performance_metrics[ticker] = {'Error': str(e)}

            self.progress_update.emit("Training completed successfully!")
            self.training_complete.emit(nn_strategy, performance_metrics)

        except Exception as e:
            self.training_error.emit(f"Training failed: {str(e)}")


class EnhancedQuantWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced Quantitative Research Framework with LSTM/xLSTM")
        self.setGeometry(100, 100, 1600, 1000)

        # Initialize framework components
        self.data_pipeline = DataPipeline()
        self.strategy = Strategy()
        self.optimizer = PortfolioOptimizer()
        self.risk_analyzer = RiskAnalyzer()
        self.neural_strategy = None

        # Data storage
        self.current_data = None
        self.current_signals = None
        self.neural_signals = None
        self.portfolio_returns = None
        self.neural_performance = None

        # Training thread
        self.training_thread = None

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

        # Right panel: Tabbed results
        right_panel = self.create_tabbed_results_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions
        main_splitter.setSizes([450, 1150])

    def create_control_panel(self):
        """Create the enhanced left control panel with LSTM options."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Input Group
        input_group = QGroupBox("Input Parameters")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        # Tickers input
        input_layout.addWidget(QLabel("Tickers (comma-separated):"))
        self.tickers_input = QLineEdit("AAPL,MSFT,GOOGL,TSLA")
        input_layout.addWidget(self.tickers_input)

        # Date inputs
        input_layout.addWidget(QLabel("Start Date:"))
        self.start_date = QDateEdit(QDate(2022, 1, 1))
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

        # LSTM Configuration Group
        lstm_group = QGroupBox("LSTM/xLSTM Configuration")
        lstm_layout = QVBoxLayout()
        lstm_group.setLayout(lstm_layout)

        # Enable LSTM checkbox
        self.enable_lstm = QCheckBox("Enable Neural Network Strategy")
        self.enable_lstm.setChecked(True)
        lstm_layout.addWidget(self.enable_lstm)

        # Model type selection
        lstm_layout.addWidget(QLabel("Model Type:"))
        self.model_type = QComboBox()
        self.model_type.addItems(['lstm', 'xlstm'])
        self.model_type.setCurrentText('xlstm')
        lstm_layout.addWidget(self.model_type)

        # Sequence length
        lstm_layout.addWidget(QLabel("Sequence Length:"))
        self.sequence_length = QSpinBox()
        self.sequence_length.setRange(20, 120)
        self.sequence_length.setValue(60)
        lstm_layout.addWidget(self.sequence_length)

        # Training epochs
        lstm_layout.addWidget(QLabel("Training Epochs:"))
        self.epochs = QSpinBox()
        self.epochs.setRange(10, 200)
        self.epochs.setValue(50)
        lstm_layout.addWidget(self.epochs)

        layout.addWidget(lstm_group)

        # Control buttons
        button_group = QGroupBox("Controls")
        button_layout = QVBoxLayout()
        button_group.setLayout(button_layout)

        # Enhanced run button
        run_button = QPushButton("🚀 Run Enhanced Analysis")
        run_button.clicked.connect(self.run_enhanced_analysis)
        run_button.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        button_layout.addWidget(run_button)

        # Traditional analysis button
        traditional_button = QPushButton("📊 Traditional Analysis Only")
        traditional_button.clicked.connect(self.run_traditional_analysis)
        traditional_button.setStyleSheet("""
            QPushButton { 
                background-color: #2196F3; 
                color: white; 
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        button_layout.addWidget(traditional_button)

        clear_button = QPushButton("🗑️ Clear Results")
        clear_button.clicked.connect(self.clear_results)
        clear_button.setStyleSheet("""
            QPushButton { 
                background-color: #f44336; 
                color: white; 
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #d32f2f; }
        """)
        button_layout.addWidget(clear_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        button_layout.addWidget(self.progress_bar)

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

        self.neural_status_label = QLabel("Neural Network: Not trained")
        self.neural_status_label.setWordWrap(True)
        results_layout.addWidget(self.neural_status_label)

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

    def create_tabbed_results_panel(self):
        """Create tabbed results panel with separate tabs for different analyses."""
        self.tab_widget = QTabWidget()

        # Traditional Analysis Tab
        traditional_tab = QWidget()
        traditional_layout = QVBoxLayout()
        traditional_tab.setLayout(traditional_layout)

        self.traditional_figure = Figure(figsize=(12, 10))
        self.traditional_canvas = FigureCanvas(self.traditional_figure)
        traditional_layout.addWidget(self.traditional_canvas)

        self.traditional_axes = []
        for i in range(4):
            ax = self.traditional_figure.add_subplot(2, 2, i + 1)
            self.traditional_axes.append(ax)

        self.traditional_figure.tight_layout(pad=3.0)
        self.tab_widget.addTab(traditional_tab, "📊 Traditional Analysis")

        # Neural Network Analysis Tab
        neural_tab = QWidget()
        neural_layout = QVBoxLayout()
        neural_tab.setLayout(neural_layout)

        self.neural_figure = Figure(figsize=(12, 10))
        self.neural_canvas = FigureCanvas(self.neural_figure)
        neural_layout.addWidget(self.neural_canvas)

        self.neural_axes = []
        for i in range(4):
            ax = self.neural_figure.add_subplot(2, 2, i + 1)
            self.neural_axes.append(ax)

        self.neural_figure.tight_layout(pad=3.0)
        self.tab_widget.addTab(neural_tab, "🧠 Neural Network Analysis")

        # Performance Comparison Tab
        comparison_tab = QWidget()
        comparison_layout = QVBoxLayout()
        comparison_tab.setLayout(comparison_layout)

        self.comparison_figure = Figure(figsize=(12, 8))
        self.comparison_canvas = FigureCanvas(self.comparison_figure)
        comparison_layout.addWidget(self.comparison_canvas)

        self.comparison_ax = self.comparison_figure.add_subplot(1, 1, 1)
        self.comparison_figure.tight_layout(pad=3.0)
        self.tab_widget.addTab(comparison_tab, "⚡ Performance Comparison")

        # Neural Network Performance Table Tab
        performance_tab = QWidget()
        performance_layout = QVBoxLayout()
        performance_tab.setLayout(performance_layout)

        self.performance_table = QTableWidget()
        performance_layout.addWidget(self.performance_table)

        self.tab_widget.addTab(performance_tab, "📈 Model Performance")

        return self.tab_widget

    def log_message(self, message):
        """Add message to console output."""
        self.console_output.append(message)
        self.console_output.ensureCursorVisible()

    def clear_results(self):
        """Clear all results and plots."""
        # Clear traditional analysis plots
        for ax in self.traditional_axes:
            ax.clear()
        self.traditional_canvas.draw()

        # Clear neural network plots
        for ax in self.neural_axes:
            ax.clear()
        self.neural_canvas.draw()

        # Clear comparison plot
        self.comparison_ax.clear()
        self.comparison_canvas.draw()

        # Clear performance table
        self.performance_table.clear()

        # Reset labels
        self.weights_label.setText("Portfolio Weights: None")
        self.performance_label.setText("Performance: None")
        self.neural_status_label.setText("Neural Network: Not trained")
        self.console_output.clear()

        # Reset data
        self.current_data = None
        self.current_signals = None
        self.neural_signals = None
        self.portfolio_returns = None
        self.neural_strategy = None
        self.neural_performance = None

    def run_traditional_analysis(self):
        """Run traditional analysis without neural networks."""
        self.enable_lstm.setChecked(False)
        self.run_enhanced_analysis()

    def run_enhanced_analysis(self):
        """Run the complete enhanced analysis pipeline."""
        try:
            self.log_message("=== Starting Enhanced Analysis ===")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)

            # Get input parameters
            tickers = [t.strip().upper() for t in self.tickers_input.text().split(',')]
            start_date = self.start_date.date().toString("yyyy-MM-dd")
            end_date = self.end_date.date().toString("yyyy-MM-dd")
            opt_method = self.optimization_method.currentText()

            self.log_message(f"Analyzing tickers: {tickers}")
            self.log_message(f"Date range: {start_date} to {end_date}")

            # Step 1-5: Traditional analysis
            self.progress_bar.setValue(10)
            self.log_message("Step 1: Fetching stock data...")
            df = self.data_pipeline.fetch_data(tickers, start_date, end_date)

            self.progress_bar.setValue(20)
            self.log_message("Step 2: Cleaning data...")
            df = self.data_pipeline.clean_data(df)

            self.progress_bar.setValue(30)
            self.log_message("Step 3: Engineering features...")
            df = self.data_pipeline.engineer_features(df)

            self.current_data = df

            self.progress_bar.setValue(40)
            self.log_message("Step 4: Generating traditional trading signals...")
            signals = self.strategy.generate_signals(df)
            self.current_signals = signals

            self.progress_bar.setValue(50)
            self.log_message("Step 5: Backtesting traditional strategy...")
            portfolio_returns = self.strategy.backtest(df, signals)
            self.portfolio_returns = portfolio_returns

            self.progress_bar.setValue(60)
            self.log_message(f"Step 6: Optimizing portfolio using {opt_method}...")
            weights = self.optimizer.optimize(df, method=opt_method)

            self.progress_bar.setValue(70)
            self.log_message("Step 7: Calculating risk metrics...")
            risk_metrics = self.risk_analyzer.calculate_risk(portfolio_returns)

            # Update traditional analysis visualization
            self.update_traditional_plots(df, portfolio_returns, risk_metrics, weights, signals)

            if self.enable_lstm.isChecked():
                self.progress_bar.setValue(75)
                self.log_message("Step 8: Training neural network models...")

                # Start LSTM training in background thread
                self.training_thread = LSTMTrainingThread(
                    df,
                    self.model_type.currentText(),
                    tickers,
                    self.sequence_length.value(),
                    self.epochs.value()
                )

                self.training_thread.progress_update.connect(self.log_message)
                self.training_thread.training_complete.connect(self.on_training_complete)
                self.training_thread.training_error.connect(self.on_training_error)
                self.training_thread.start()

                self.neural_status_label.setText("Neural Network: Training in progress...")
            else:
                self.progress_bar.setValue(100)
                self.log_message("=== Traditional Analysis Complete ===")
                self.neural_status_label.setText("Neural Network: Disabled")

            # Update results display
            weights_text = ", ".join([f"{k}: {v:.3f}" for k, v in weights.items()])
            self.weights_label.setText(f"Portfolio Weights: {weights_text}")

            performance_text = f"Total Return: {risk_metrics.get('Mean_Return', 0):.2%}, " \
                               f"Volatility: {risk_metrics.get('Volatility', 0):.2%}, " \
                               f"Sharpe: {risk_metrics.get('Sharpe_Ratio', 0):.2f}"
            self.performance_label.setText(f"Performance: {performance_text}")

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.log_message(error_msg)
            self.progress_bar.setVisible(False)
            self.on_training_error(error_msg)

    def on_training_complete(self, neural_strategy, performance_metrics):
        """Handle completion of neural network training."""
        try:
            self.progress_bar.setValue(90)
            self.neural_strategy = neural_strategy
            self.neural_performance = performance_metrics

            # Generate neural signals
            self.neural_signals = neural_strategy.generate_neural_signals(self.current_data)

            self.progress_bar.setValue(95)
            self.log_message("Step 9: Updating neural network visualizations...")

            # Update neural network visualizations
            self.update_neural_plots()
            self.update_comparison_plots()
            self.update_performance_table()

            self.progress_bar.setValue(100)
            self.progress_bar.setVisible(False)

            # Update neural status
            trained_models = len([m for m in neural_strategy.models.values() if m.is_trained])
            total_models = len(neural_strategy.models)
            self.neural_status_label.setText(
                f"Neural Network: {trained_models}/{total_models} models trained successfully")

            self.log_message("=== Enhanced Analysis Complete ===")

        except Exception as e:
            self.on_training_error(f"Error in training completion: {str(e)}")

    def on_training_error(self, error_message):
        """Handle neural network training errors."""
        self.log_message(f"Neural network error: {error_message}")
        self.progress_bar.setVisible(False)
        self.neural_status_label.setText("Neural Network: Training failed")

    def update_traditional_plots(self, df, returns, risk_metrics, weights, signals):
        """Update traditional analysis plots."""
        # Clear all axes
        for ax in self.traditional_axes:
            ax.clear()

        try:
            # Plot 1: Cumulative Returns
            if returns is not None and not returns.empty:
                cumulative_returns = (1 + returns).cumprod()
                cumulative_returns.plot(ax=self.traditional_axes[0], linewidth=2, color='blue')
                self.traditional_axes[0].set_title("Traditional Strategy: Cumulative Returns", fontsize=12,
                                                   fontweight='bold')
                self.traditional_axes[0].set_xlabel("Date")
                self.traditional_axes[0].set_ylabel("Cumulative Return")
                self.traditional_axes[0].grid(True, alpha=0.3)

            # Plot 2: Risk Metrics Bar Chart
            if risk_metrics:
                key_metrics = {k: v for k, v in risk_metrics.items()
                               if k in ['Volatility', 'Sharpe_Ratio', 'Max_Drawdown', 'VaR_95']}

                if key_metrics:
                    metrics_names = list(key_metrics.keys())
                    metrics_values = list(key_metrics.values())

                    bars = self.traditional_axes[1].bar(metrics_names, metrics_values,
                                                        color=['red' if v < 0 else 'green' for v in metrics_values])
                    self.traditional_axes[1].set_title("Traditional Strategy: Risk Metrics", fontsize=12,
                                                       fontweight='bold')
                    self.traditional_axes[1].set_ylabel("Value")
                    self.traditional_axes[1].tick_params(axis='x', rotation=45)

                    for bar, value in zip(bars, metrics_values):
                        height = bar.get_height()
                        self.traditional_axes[1].text(bar.get_x() + bar.get_width() / 2., height,
                                                      f'{value:.3f}', ha='center',
                                                      va='bottom' if height >= 0 else 'top')

            # Plot 3: Technical Indicator (RSI)
            rsi_cols = [col for col in df.columns if 'RSI' in col]
            if rsi_cols:
                df[rsi_cols[0]].plot(ax=self.traditional_axes[2], color='purple', linewidth=1)
                self.traditional_axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
                self.traditional_axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
                self.traditional_axes[2].axhline(y=50, color='black', linestyle='-', alpha=0.5, label='Neutral (50)')
                self.traditional_axes[2].set_title(f"RSI: {rsi_cols[0].replace('_RSI', '')}", fontsize=12,
                                                   fontweight='bold')
                self.traditional_axes[2].set_ylabel("RSI Value")
                self.traditional_axes[2].set_ylim(0, 100)
                self.traditional_axes[2].legend(fontsize=8)
                self.traditional_axes[2].grid(True, alpha=0.3)

            # Plot 4: Portfolio Weights
            if weights is not None and not weights.empty:
                weights_plot = weights[weights > 0.001]
                if not weights_plot.empty:
                    wedges, texts, autotexts = self.traditional_axes[3].pie(weights_plot.values,
                                                                            labels=weights_plot.index,
                                                                            autopct='%1.1f%%',
                                                                            startangle=90)
                    self.traditional_axes[3].set_title("Portfolio Weights", fontsize=12, fontweight='bold')

                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')

        except Exception as e:
            self.log_message(f"Error updating traditional plots: {e}")

        self.traditional_figure.tight_layout()
        self.traditional_canvas.draw()

    # Complete the missing parts from your enhanced_lstm_complete.py

    def update_neural_plots(self):
        """Update neural network analysis plots."""
        if not self.neural_strategy or self.neural_signals is None:
            return

        # Clear all axes
        for ax in self.neural_axes:
            ax.clear()

        try:
            # Plot 1: Neural Network Signals Heatmap
            if not self.neural_signals.empty:
                # Create a heatmap of signals over time
                recent_signals = self.neural_signals.tail(100)  # Last 100 days

                if not recent_signals.empty and len(recent_signals.columns) > 0:
                    im = self.neural_axes[0].imshow(recent_signals.T.values, cmap='RdYlGn', aspect='auto',
                                                    interpolation='nearest')
                    self.neural_axes[0].set_title("Neural Network Signals (Recent 100 Days)", fontsize=12,
                                                  fontweight='bold')
                    self.neural_axes[0].set_xlabel("Time Steps")
                    self.neural_axes[0].set_ylabel("Tickers")
                    self.neural_axes[0].set_yticks(range(len(recent_signals.columns)))
                    self.neural_axes[0].set_yticklabels(recent_signals.columns)

                    # Add colorbar
                    from matplotlib import cm
                    cbar = self.neural_figure.colorbar(im, ax=self.neural_axes[0], shrink=0.6)
                    cbar.set_label('Signal (-1: Sell, 0: Hold, 1: Buy)')

            # Plot 2: Model Performance Metrics
            if self.neural_performance:
                tickers = list(self.neural_performance.keys())
                accuracies = []
                valid_tickers = []

                for ticker in tickers:
                    if isinstance(self.neural_performance[ticker], dict) and 'Direction_Accuracy' in \
                            self.neural_performance[ticker]:
                        accuracies.append(self.neural_performance[ticker]['Direction_Accuracy'])
                        valid_tickers.append(ticker)

                if accuracies and valid_tickers:
                    bars = self.neural_axes[1].bar(valid_tickers, accuracies, color='skyblue')
                    self.neural_axes[1].set_title("Neural Network Direction Accuracy", fontsize=12, fontweight='bold')
                    self.neural_axes[1].set_ylabel("Accuracy (%)")
                    self.neural_axes[1].set_ylim(0, 100)
                    self.neural_axes[1].tick_params(axis='x', rotation=45)

                    # Add value labels on bars
                    for bar, acc in zip(bars, accuracies):
                        height = bar.get_height()
                        self.neural_axes[1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                                                 f'{acc:.1f}%', ha='center', va='bottom')
                else:
                    self.neural_axes[1].text(0.5, 0.5, 'No valid accuracy data', ha='center', va='center',
                                             transform=self.neural_axes[1].transAxes)
                    self.neural_axes[1].set_title("Neural Network Direction Accuracy", fontsize=12, fontweight='bold')

            # Plot 3: Signal Distribution
            if not self.neural_signals.empty:
                # Count signals for each ticker
                signal_data = {}
                for col in self.neural_signals.columns:
                    value_counts = self.neural_signals[col].value_counts()
                    signal_data[col] = {
                        'Sell (-1)': value_counts.get(-1, 0),
                        'Hold (0)': value_counts.get(0, 0),
                        'Buy (1)': value_counts.get(1, 0)
                    }

                if signal_data:
                    signal_df = pd.DataFrame(signal_data).T
                    signal_df.plot(kind='bar', ax=self.neural_axes[2], stacked=True,
                                   color=['red', 'gray', 'green'], alpha=0.7)
                    self.neural_axes[2].set_title("Neural Network Signal Distribution", fontsize=12, fontweight='bold')
                    self.neural_axes[2].set_ylabel("Count")
                    self.neural_axes[2].legend(['Sell (-1)', 'Hold (0)', 'Buy (1)'])
                    self.neural_axes[2].tick_params(axis='x', rotation=45)

            # Plot 4: Prediction vs Actual (for first ticker with valid model)
            if self.neural_strategy and self.neural_strategy.models:
                valid_ticker = None
                for ticker, model in self.neural_strategy.models.items():
                    if model.is_trained:
                        valid_ticker = ticker
                        break

                if valid_ticker:
                    model = self.neural_strategy.models[valid_ticker]
                    target_col = f'{valid_ticker}_Adj Close' if f'{valid_ticker}_Adj Close' in self.current_data.columns else 'Adj Close'

                    if target_col in self.current_data.columns:
                        try:
                            # Get recent actual prices
                            actual_prices = self.current_data[target_col].tail(50).dropna()

                            if len(actual_prices) > 10:
                                # Plot actual prices
                                actual_prices.plot(ax=self.neural_axes[3], label='Actual', linewidth=2, color='blue')

                                # Generate simple trend prediction for visualization
                                feature_cols = model.feature_columns
                                if all(col in self.current_data.columns for col in feature_cols):
                                    recent_features = self.current_data[feature_cols].tail(50).dropna()

                                    if len(recent_features) >= model.sequence_length:
                                        try:
                                            X_scaled = model.scaler_features.transform(recent_features)

                                            # Create sequences for prediction
                                            predictions = []
                                            for i in range(model.sequence_length, len(X_scaled)):
                                                X_seq = X_scaled[i - model.sequence_length:i].reshape(1,
                                                                                                      model.sequence_length,
                                                                                                      -1)
                                                pred = model.predict(X_seq)[0]
                                                predictions.append(pred)

                                            if predictions:
                                                pred_index = actual_prices.index[-len(predictions):]
                                                pred_series = pd.Series(predictions, index=pred_index)
                                                pred_series.plot(ax=self.neural_axes[3], label='Predicted', linewidth=2,
                                                                 color='red', alpha=0.7)

                                        except Exception as pred_error:
                                            self.log_message(f"Prediction visualization error: {pred_error}")

                                self.neural_axes[3].set_title(f"Prediction vs Actual: {valid_ticker}", fontsize=12,
                                                              fontweight='bold')
                                self.neural_axes[3].set_ylabel("Price")
                                self.neural_axes[3].legend()
                                self.neural_axes[3].grid(True, alpha=0.3)
                            else:
                                self.neural_axes[3].text(0.5, 0.5, 'Insufficient data for visualization',
                                                         ha='center', va='center',
                                                         transform=self.neural_axes[3].transAxes)
                                self.neural_axes[3].set_title("Prediction vs Actual", fontsize=12, fontweight='bold')

                        except Exception as e:
                            self.log_message(f"Error in prediction plot: {e}")
                            self.neural_axes[3].text(0.5, 0.5, f'Error: {str(e)}',
                                                     ha='center', va='center', transform=self.neural_axes[3].transAxes)
                            self.neural_axes[3].set_title("Prediction vs Actual", fontsize=12, fontweight='bold')
                else:
                    self.neural_axes[3].text(0.5, 0.5, 'No trained models available',
                                             ha='center', va='center', transform=self.neural_axes[3].transAxes)
                    self.neural_axes[3].set_title("Prediction vs Actual", fontsize=12, fontweight='bold')

        except Exception as e:
            self.log_message(f"Error updating neural plots: {e}")
            for i, ax in enumerate(self.neural_axes):
                ax.text(0.5, 0.5, f'Plot {i + 1} Error: {str(e)[:50]}...',
                        ha='center', va='center', transform=ax.transAxes)

        self.neural_figure.tight_layout()
        self.neural_canvas.draw()

    def update_comparison_plots(self):
        """Update performance comparison plots."""
        if not self.portfolio_returns or not self.neural_strategy:
            return

        self.comparison_ax.clear()

        try:
            # Plot traditional strategy returns
            if self.portfolio_returns is not None and not self.portfolio_returns.empty:
                traditional_cumulative = (1 + self.portfolio_returns).cumprod()
                traditional_cumulative.plot(ax=self.comparison_ax, label='Traditional Strategy',
                                            linewidth=2, color='blue')

            # Plot neural network strategy returns (if available)
            if self.neural_signals is not None and not self.neural_signals.empty and self.current_data is not None:
                try:
                    # Calculate neural strategy returns
                    neural_returns = self.calculate_neural_strategy_returns()
                    if neural_returns is not None and not neural_returns.empty:
                        neural_cumulative = (1 + neural_returns).cumprod()
                        neural_cumulative.plot(ax=self.comparison_ax, label='Neural Network Strategy',
                                               linewidth=2, color='red')

                except Exception as e:
                    self.log_message(f"Error calculating neural returns: {e}")

            # Add benchmark (buy and hold first ticker)
            if self.current_data is not None:
                price_cols = [col for col in self.current_data.columns if 'Adj Close' in col]
                if price_cols:
                    first_price_col = price_cols[0]
                    benchmark_returns = self.current_data[first_price_col].pct_change().dropna()
                    benchmark_cumulative = (1 + benchmark_returns).cumprod()
                    benchmark_cumulative.plot(ax=self.comparison_ax, label='Buy & Hold Benchmark',
                                              linewidth=2, color='green', alpha=0.7)

            self.comparison_ax.set_title("Strategy Performance Comparison", fontsize=14, fontweight='bold')
            self.comparison_ax.set_xlabel("Date")
            self.comparison_ax.set_ylabel("Cumulative Return")
            self.comparison_ax.legend()
            self.comparison_ax.grid(True, alpha=0.3)

        except Exception as e:
            self.log_message(f"Error updating comparison plots: {e}")
            self.comparison_ax.text(0.5, 0.5, f'Comparison Error: {str(e)}',
                                    ha='center', va='center', transform=self.comparison_ax.transAxes)

        self.comparison_figure.tight_layout()
        self.comparison_canvas.draw()

    def calculate_neural_strategy_returns(self):
        """Calculate returns for neural network strategy."""
        try:
            if self.neural_signals is None or self.current_data is None:
                return None

            # Get price data
            price_cols = [col for col in self.current_data.columns if 'Adj Close' in col]
            if not price_cols:
                return None

            # Calculate returns for each asset
            returns_data = {}
            for col in price_cols:
                ticker = col.replace('_Adj Close', '')
                if ticker in self.neural_signals.columns:
                    price_returns = self.current_data[col].pct_change().dropna()
                    signal_col = self.neural_signals[ticker]

                    # Align signals with returns
                    aligned_signals = signal_col.reindex(price_returns.index).fillna(0)

                    # Calculate strategy returns (signal * return)
                    strategy_returns = aligned_signals.shift(1) * price_returns
                    returns_data[ticker] = strategy_returns

            if returns_data:
                returns_df = pd.DataFrame(returns_data).dropna()
                # Equal weight portfolio returns
                portfolio_returns = returns_df.mean(axis=1)
                return portfolio_returns

        except Exception as e:
            self.log_message(f"Error in neural strategy returns calculation: {e}")

        return None

    def update_performance_table(self):
        """Update the performance metrics table."""
        if not self.neural_performance:
            return

        try:
            # Prepare data for table
            table_data = []
            headers = ['Ticker', 'MSE', 'MAE', 'Direction Accuracy (%)', 'Predictions Count', 'Status']

            for ticker, metrics in self.neural_performance.items():
                if isinstance(metrics, dict):
                    if 'Error' in metrics:
                        row = [ticker, 'N/A', 'N/A', 'N/A', 'N/A', f"Error: {metrics['Error'][:30]}..."]
                    else:
                        row = [
                            ticker,
                            f"{metrics.get('MSE', 0):.4f}",
                            f"{metrics.get('MAE', 0):.4f}",
                            f"{metrics.get('Direction_Accuracy', 0):.2f}",
                            str(metrics.get('Predictions_Count', 0)),
                            'Trained Successfully'
                        ]
                    table_data.append(row)

            # Set up table
            self.performance_table.setRowCount(len(table_data))
            self.performance_table.setColumnCount(len(headers))
            self.performance_table.setHorizontalHeaderLabels(headers)

            # Fill table
            for i, row in enumerate(table_data):
                for j, item in enumerate(row):
                    self.performance_table.setItem(i, j, QTableWidgetItem(str(item)))

            # Adjust column widths
            self.performance_table.resizeColumnsToContents()

        except Exception as e:
            self.log_message(f"Error updating performance table: {e}")

    def closeEvent(self, event):
        """Handle application close event."""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.quit()
            self.training_thread.wait()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Set application properties
    app.setApplicationName("Enhanced Quantitative Research Framework")
    app.setApplicationVersion("2.0")

    # Create and show main window
    window = EnhancedQuantWindow()
    window.show()

    sys.exit(app.exec_())
