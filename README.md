# Quantitative Research Framework

A modular, Python-based quantitative finance platform with a PyQt5 GUI for financial data analysis, strategy backtesting, portfolio optimization, risk management, and visualization. Designed for prototyping trading strategies and conducting alpha research, this framework integrates technical indicators, return calculations, and Fama-French factor data, making it a robust tool for quantitative analysts and researchers.

## Table of Contents
- [Overview](#overview)
- [Features and Capabilities](#features-and-capabilities)
  - [Data Ingestion and Cleaning](#data-ingestion-and-cleaning)
  - [Feature Engineering and Technical Indicators](#feature-engineering-and-technical-indicators)
  - [Strategy Development and Backtesting](#strategy-development-and-backtesting)
  - [Portfolio Optimization](#portfolio-optimization)
  - [Risk Analysis](#risk-analysis)
  - [GUI and Visualization](#gui-and-visualization)
  - [Fama-French Factor Integration](#fama-french-factor-integration)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Example Workflow](#example-workflow)
- [Value for Quantitative Finance](#value-for-quantitative-finance)
- [Limitations and Future Enhancements](#limitations-and-future-enhancements)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview
This quantitative research framework is a Python-based desktop application built with PyQt5, designed to support financial analysis and trading strategy development. It provides a modular pipeline for:
- Fetching and cleaning historical stock data from Yahoo Finance.
- Computing technical indicators and multi-period returns for alpha generation.
- Developing and backtesting trading strategies (e.g., momentum-based).
- Optimizing portfolio weights using mean-variance optimization.
- Calculating risk metrics like volatility and Value at Risk (VaR).
- Integrating Fama-French 5-factor data for factor-based analysis.
- Visualizing results through an interactive GUI with plots for returns, risk metrics, and indicators.

The framework is ideal for quantitative researchers, traders, or students aiming to prototype strategies or conduct financial analysis, with extensibility for advanced applications such as machine learning or live trading.

## Features and Capabilities

### Data Ingestion and Cleaning
- **Source**: Fetches OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance via `yfinance`.
- **Cleaning**: Handles missing values with forward/backward filling and removes outliers (values beyond 5 standard deviations).
- **Flexibility**: Supports single or multiple tickers with standardized column naming (e.g., `AAPL_Adj Close`).
- **Error Handling**: Raises descriptive errors for empty datasets or missing price columns, with debug logging.
- **Example**: Fetches data for `AAPL,MSFT` from 2023-01-01 to 2025-06-15, producing a DataFrame with columns like `['AAPL_Adj Close', 'AAPL_High', ...]`.

### Feature Engineering and Technical Indicators
- **Indicators**:
  - **Moving Average (MA_20)**: 20-day simple moving average.
  - **Momentum**: 10-day percentage change.
  - **Garman-Klass Volatility**: `σ² = (1/n) * Σ [0.5 * ln(H/L)² - (2 * ln(2) - 1) * ln(C/O)²]` (20-day rolling mean).
  - **Relative Strength Index (RSI)**: 20-period RSI for overbought/oversold signals.
  - **Bollinger Bands**: 20-period bands (low, mid, high) on log-adjusted prices.
  - **Average True Range (ATR)**: 14-period ATR, standardized for volatility.
  - **MACD**: 12-26-9 MACD, standardized for trend detection.
  - **Dollar Volume**: `(Adj Close * Volume) / 1e6` for trading activity.
- **Returns**: Calculates 1, 2, 3, 6, 9, and 12-month returns, clipped at 0.5% quantiles, using 21 trading days per month.
- **Output**: Enriches DataFrame with columns like `['AAPL_RSI', 'AAPL_MACD', 'AAPL_Return_1m', ...]`.

### Strategy Development and Backtesting
- **Signal Generation**: Implements a momentum-based strategy (long if momentum > 0, short otherwise).
- **Backtesting**: Computes daily portfolio returns by multiplying asset returns with lagged signals, averaging across tickers.
- **Extensibility**: Easily modified to incorporate other indicators (e.g., RSI, MACD) or machine learning models.
- **Output**: Signals DataFrame (e.g., `['AAPL': 1, 'MSFT': -1]`) and portfolio returns Series.

### Portfolio Optimization
- **Method**: Mean-variance optimization via `cvxpy`, maximizing return minus 0.5 * risk (w^T * Σ * w).
- **Constraints**: Weights sum to 1, non-negative (long-only).
- **Inputs**: Daily returns and covariance matrix from adjusted close prices.
- **Output**: Optimal weights, e.g., `{'AAPL': 0.4, 'MSFT': 0.35, 'GOOGL': 0.25}`.

### Risk Analysis
- **Metrics**:
  - **Volatility**: Annualized standard deviation (√252 * daily std).
  - **Value at Risk (VaR)**: 95% VaR, annualized (5th percentile).
- **Input**: Portfolio returns weighted by optimized weights.
- **Output**: `{'Volatility': 0.15, 'VaR_95': -0.02}`.

### GUI and Visualization
- **Interface**: PyQt5 desktop app with:
  - Input fields for tickers and date range.
  - "Run Analysis" button to execute the pipeline.
  - Label displaying portfolio weights.
  - Three embedded Matplotlib plots:
    1. Cumulative portfolio returns (line plot).
    2. Risk metrics (bar plot).
    3. RSI for the first ticker (line plot).
- **Error Handling**: Displays errors in plots and console for debugging.
- **Output**: Visualizes key results and weights, e.g., `Portfolio Weights: {'AAPL': 0.4, 'MSFT': 0.35, 'GOOGL': 0.25}`.

### Fama-French Factor Integration
- **Source**: Fetches 5-factor data (Mkt-RF, SMB, HML, RMW, CMA) via `pandas_datareader`.
- **Processing**: Resamples to monthly frequency, aligns with 1-month stock returns.
- **Use Case**: Supports factor-based analysis, e.g., regressing stock returns against factors.
- **Output**: DataFrame with columns like `['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'AAPL_Return_1m']`.

## Project Structure
```plaintext
quant_framework/
├── main.py              # GUI and main application logic
├── data_pipeline.py     # Data fetching, cleaning, and feature engineering
├── strategy.py          # Signal generation and backtesting
├── portfolio.py         # Portfolio optimization
├── risk.py              # Risk metrics calculation
├── requirements.txt     # Python dependencies
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Suad0/quant_framework.git
   cd quant_framework
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python main.py
   ```
2. In the GUI:
   - Enter tickers (e.g., `AAPL,MSFT,GOOGL`).
   - Select start and end dates (e.g., 2023-01-01 to 2025-06-15).
   - Click "Run Analysis" to:
     - Fetch and clean stock data.
     - Compute indicators, returns, and Fama-French factors.
     - Generate signals and backtest.
     - Optimize portfolio weights.
     - Calculate risk metrics.
     - Display plots and weights.
3. Review console output for debug logs (DataFrame shapes, columns).

## Example Workflow
For `AAPL,MSFT,GOOGL` from 2023-01-01 to 2025-06-15:
1. **Input**: Enter tickers and dates, click "Run Analysis".
2. **Pipeline**:
   - Fetches OHLCV data, standardizes columns.
   - Cleans outliers, computes indicators (RSI, MACD, etc.), and returns.
   - Aligns 1-month returns with Fama-French factors.
3. **Strategy**: Generates momentum signals, backtests portfolio returns.
4. **Optimization**: Allocates weights, e.g., `{'AAPL': 0.4, 'MSFT': 0.35, 'GOOGL': 0.25}`.
5. **Risk**: Computes `{'Volatility': 0.15, 'VaR_95': -0.02}`.
6. **Output**: Displays plots for returns, risk, `AAPL_RSI`, and weights in GUI.

## Value for Quantitative Finance
This framework demonstrates skills valued at top-tier quant funds (e.g., Two Sigma, Citadel):
- **Data Pipeline**: Robust handling of financial data, critical for alpha research.
- **Feature Engineering**: Comprehensive indicator set, showcasing technical analysis expertise.
- **Strategy and Backtesting**: Practical experience with signal generation and evaluation.
- **Optimization and Risk**: Knowledge of portfolio construction and risk management.
- **Software Engineering**: Modular, well-documented code with a GUI.
- **Factor Models**: Integration of Fama-French factors, relevant for multi-factor strategies.

## Limitations and Future Enhancements
- **Limitations**:
  - Basic momentum strategy; lacks advanced models (e.g., machine learning).
  - Relies on `yfinance`, which may have reliability issues compared to premium data sources.
  - Limited risk metrics (volatility, VaR); could include Sharpe ratio, max drawdown.
  - GUI displays only RSI; could support multiple indicators.
  - Designed for backtesting, not live trading.
- **Future Enhancements**:
  - Add strategy selector (e.g., RSI, MACD-based) via GUI dropdown.
  - Visualize additional indicators (e.g., Bollinger Bands, MACD).
  - Incorporate Fama-French factors into strategies (e.g., factor-based signals).
  - Export results to CSV.
  - Integrate live trading APIs (e.g., Alpaca).
  - Optimize for large datasets using `dask` or parallel processing.

## Dependencies
Listed in `requirements.txt`:
- Python 3.12
- PyQt5==5.15.9
- matplotlib==3.7.1
- pandas==1.5.3
- yfinance==0.2.28
- numpy==1.24.3
- cvxpy==1.3.2
- pandas_ta==0.3.14b0
- pandas_datareader==0.10.0

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a pull request.

Please ensure code follows PEP 8 and includes tests where applicable.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
