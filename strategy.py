import pandas as pd
import numpy as np


class Strategy:
    def generate_signals(self, df):
        """Generate trading signals based on momentum."""
        momentum_cols = [col for col in df.columns if 'Momentum' in col]
        if not momentum_cols:
            raise ValueError("No momentum columns found in DataFrame.")

        signals = pd.DataFrame(index=df.index)
        for col in momentum_cols:
            ticker = col.split('_')[0]
            signals[ticker] = np.where(df[col].fillna(0) > 0, 1, -1)
        return signals

    def backtest(self, df, signals):
        """Backtest strategy using signals."""
        returns = pd.DataFrame(index=df.index)
        close_cols = [col for col in df.columns if 'Adj Close' in col]
        if not close_cols:
            raise ValueError(f"No 'Adj Close' columns found for backtesting. Columns: {df.columns}")

        for ticker in signals.columns:
            close_col = next((col for col in close_cols if col.startswith(ticker)), None)
            if close_col:
                returns[ticker] = df[close_col].pct_change() * signals[ticker].shift(1)
        portfolio_returns = returns.mean(axis=1)
        portfolio_returns.name = 'Portfolio'
        if portfolio_returns.empty:
            raise ValueError("No portfolio returns generated.")
        return portfolio_returns
