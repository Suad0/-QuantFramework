import yfinance as yf
import pandas as pd
import numpy as np


class DataPipeline:
    def fetch_data(self, tickers, start_date, end_date):
        """Fetch historical stock data from Yahoo Finance."""
        df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
        if len(tickers) == 1:
            # For single ticker, ensure flat column structure
            df.columns = ['Adj Close' if col == 'Close' or col == 'Adj Close' else col for col in df.columns]
        else:
            # For multiple tickers, flatten multi-level columns
            df.columns = [f"{col[1]}_{col[0]}" if isinstance(col, tuple) else col for col in df.columns]
            df.columns = [col.replace('Close', 'Adj Close') if 'Close' in col else col for col in df.columns]
        print(f"Fetched data shape: {df.shape}, columns: {df.columns}")
        return df

    def clean_data(self, df):
        """Clean data by filling missing values and removing outliers."""
        if df.empty:
            raise ValueError("Empty DataFrame after fetching data.")

        df = df.ffill().bfill()
        close_cols = [col for col in df.columns if 'Adj Close' in col or col == 'Adj Close']
        if not close_cols:
            raise ValueError(f"No 'Adj Close' or 'Close' columns found in DataFrame. Columns: {df.columns}")

        # Relaxed outlier removal: apply per ticker and preserve data
        for col in close_cols:
            mask = np.abs(df[col] - df[col].mean()) <= (5 * df[col].std())
            df = df[mask | df[col].isna()]  # Keep NaNs to avoid dropping all rows
        df = df.dropna()  # Drop rows with NaN after outlier filtering
        print(f"Cleaned data shape: {df.shape}, columns: {df.columns}")

        if df.empty:
            raise ValueError("DataFrame is empty after cleaning.")
        return df

    def engineer_features(self, df):
        """Engineer alpha features (e.g., momentum, moving averages)."""
        close_cols = [col for col in df.columns if 'Adj Close' in col or col == 'Adj Close']
        for col in close_cols:
            ticker = col.split('_')[0] if '_' in col else 'Ticker'
            df[f'{ticker}_MA_20'] = df[col].rolling(window=20).mean()
            df[f'{ticker}_Momentum'] = df[col].pct_change(10)
        print(f"Engineered features shape: {df.shape}, columns: {df.columns}")
        return df
