import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataPipeline:
    def fetch_data(self, tickers, start_date, end_date):
        """Fetch historical stock data from Yahoo Finance."""
        try:
            df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

            if df.empty:
                raise ValueError(f"No data fetched for tickers: {tickers}")

            if len(tickers) == 1:
                # For single ticker, ensure flat column structure
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]
                df.columns = ['Adj Close' if col == 'Close' else col for col in df.columns]
            else:
                # For multiple tickers, flatten multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
                df.columns = [col.replace('Close', 'Adj Close') if 'Close' in col else col for col in df.columns]

            print(f"Fetched data shape: {df.shape}, columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise

    def clean_data(self, df):
        """Clean data by filling missing values and removing outliers."""
        if df.empty:
            raise ValueError("Empty DataFrame after fetching data.")

        # Forward fill then backward fill
        df = df.ffill().bfill()

        # Find close columns
        close_cols = [col for col in df.columns if 'Adj Close' in col or col == 'Adj Close']
        if not close_cols:
            raise ValueError(f"No 'Adj Close' columns found in DataFrame. Columns: {list(df.columns)}")

        # Remove outliers - be more conservative
        for col in close_cols:
            if col in df.columns:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df = df[(df[col] >= q1) & (df[col] <= q99)]

        # Drop any remaining NaN rows
        df = df.dropna()
        print(f"Cleaned data shape: {df.shape}")

        if df.empty:
            raise ValueError("DataFrame is empty after cleaning.")
        return df

    def engineer_features(self, df):
        """Engineer alpha features (e.g., momentum, moving averages, technical indicators)."""
        close_cols = [col for col in df.columns if 'Adj Close' in col or col == 'Adj Close']

        for col in close_cols:
            ticker = col.split('_')[0] if '_' in col else 'Stock'

            # Basic features
            df[f'{ticker}_MA_20'] = df[col].rolling(window=20).mean()
            df[f'{ticker}_Momentum'] = df[col].pct_change(10)

            # RSI
            df[f'{ticker}_RSI'] = self._calculate_rsi(df[col], 14)

            # Bollinger Bands
            ma_20 = df[col].rolling(window=20).mean()
            std_20 = df[col].rolling(window=20).std()
            df[f'{ticker}_BB_Upper'] = ma_20 + (std_20 * 2)
            df[f'{ticker}_BB_Lower'] = ma_20 - (std_20 * 2)

            # MACD
            exp1 = df[col].ewm(span=12).mean()
            exp2 = df[col].ewm(span=26).mean()
            df[f'{ticker}_MACD'] = exp1 - exp2
            df[f'{ticker}_MACD_Signal'] = df[f'{ticker}_MACD'].ewm(span=9).mean()

            # Volatility
            df[f'{ticker}_Volatility'] = df[col].pct_change().rolling(window=20).std()

            # Returns for different periods
            for period in [1, 5, 10, 20]:
                df[f'{ticker}_Return_{period}d'] = df[col].pct_change(period)

        print(f"Engineered features shape: {df.shape}")
        return df

    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def fetch_factor_data(self, start_date, end_date):
        """Fetch Fama-French factor data."""
        try:
            # Convert string dates to datetime if needed
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')

            # Fetch Fama-French 3-factor data (more reliable than 5-factor)
            ff_data = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start_date, end_date)[0]
            ff_data.index = pd.to_datetime(ff_data.index, format='%Y%m')
            ff_data = ff_data / 100  # Convert from percentage to decimal

            print(f"Fetched factor data shape: {ff_data.shape}")
            return ff_data
        except Exception as e:
            print(f"Warning: Could not fetch Fama-French data: {e}")
            return None
