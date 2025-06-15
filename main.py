import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QDateEdit
)
from PyQt5.QtCore import QDate
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import numpy as np
from data_pipeline import DataPipeline
from strategy import Strategy
from portfolio import PortfolioOptimizer
from risk import RiskAnalyzer


class QuantWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quant Research Framework")
        self.setGeometry(100, 100, 1200, 800)

        # Initialize framework components
        self.data_pipeline = DataPipeline()
        self.strategy = Strategy()
        self.optimizer = PortfolioOptimizer()
        self.risk_analyzer = RiskAnalyzer()

        # Setup UI
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel: Inputs
        input_layout = QVBoxLayout()
        self.tickers_input = QLineEdit("AAPL,MSFT,GOOGL")
        self.start_date = QDateEdit(QDate(2023, 1, 1))
        self.end_date = QDateEdit(QDate(2025, 6, 15))
        run_button = QPushButton("Run Analysis")
        run_button.clicked.connect(self.run_analysis)
        self.weights_label = QLabel("Portfolio Weights: None")

        input_layout.addWidget(QLabel("Tickers (comma-separated):"))
        input_layout.addWidget(self.tickers_input)
        input_layout.addWidget(QLabel("Start Date:"))
        input_layout.addWidget(self.start_date)
        input_layout.addWidget(QLabel("End Date:"))
        input_layout.addWidget(self.end_date)
        input_layout.addWidget(run_button)
        input_layout.addWidget(self.weights_label)
        input_layout.addStretch()

        # Right panel: Plots
        self.figure, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 12))
        self.canvas = FigureCanvas(self.figure)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.canvas)

        main_layout.addLayout(input_layout)
        main_layout.addLayout(plot_layout)

    def run_analysis(self):
        tickers = [t.strip() for t in self.tickers_input.text().split(',')]
        start_date = self.start_date.date().toString("yyyy-MM-dd")
        end_date = self.end_date.date().toString("yyyy-MM-dd")

        try:
            # Fetch and process stock data
            df = self.data_pipeline.fetch_data(tickers, start_date, end_date)
            print(f"Post-fetch DataFrame columns: {df.columns}")
            df = self.data_pipeline.clean_data(df)
            print(f"Post-clean DataFrame columns: {df.columns}")
            df = self.data_pipeline.engineer_features(df)
            print(f"Post-features DataFrame columns: {df.columns}")

            # Fetch Fama-French factor data
            factor_data = self.data_pipeline.fetch_factor_data(start_date, end_date)
            if factor_data is not None:
                # Align stock returns with factor data
                return_cols = [col for col in df.columns if 'Return_1m' in col]
                if return_cols:
                    stock_returns = df[return_cols].resample('M').last()
                    stock_returns.index = stock_returns.index.tz_localize(None)
                    factor_data = factor_data.join(stock_returns, how='inner')
                    print(f"Aligned factor data shape: {factor_data.shape}, columns: {factor_data.columns}")

            # Generate signals and backtest
            signals = self.strategy.generate_signals(df)
            returns = self.strategy.backtest(df, signals)

            # Optimize portfolio
            weights = self.optimizer.optimize(df)

            # Calculate risk metrics
            risk_metrics = self.risk_analyzer.calculate_risk(returns, weights)

            # Update plots
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()

            # Plot returns
            returns.cumsum().plot(ax=self.ax1)
            self.ax1.set_title("Portfolio Cumulative Returns")
            self.ax1.set_xlabel("Date")
            self.ax1.set_ylabel("Returns")

            # Plot risk metrics
            self.ax2.bar(risk_metrics.keys(), risk_metrics.values())
            self.ax2.set_title("Risk Metrics (VaR, Volatility)")
            self.ax2.set_ylabel("Value")

            # Plot RSI for the first ticker as an example
            rsi_cols = [col for col in df.columns if 'RSI' in col]
            if rsi_cols:
                df[rsi_cols[0]].plot(ax=self.ax3)
                self.ax3.set_title(f"RSI for {rsi_cols[0].split('_')[0]}")
                self.ax3.set_xlabel("Date")
                self.ax3.set_ylabel("RSI")

            self.canvas.draw()

            # Display weights in GUI
            self.weights_label.setText(f"Portfolio Weights: {weights.to_dict()}")

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            self.ax1.clear()
            self.ax1.set_title(f"Error: {str(e)}")
            self.ax2.clear()
            self.ax3.clear()
            self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QuantWindow()
    window.show()
    sys.exit(app.exec_())
