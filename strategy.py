import pandas as pd
import numpy as np


class Strategy:
    def generate_signals(self, df):
        """Generate trading signals based on multiple indicators."""
        signals = pd.DataFrame(index=df.index)

        # Get all tickers from the dataframe
        tickers = self._get_tickers_from_columns(df)

        for ticker in tickers:
            # Initialize signal as neutral
            signal = pd.Series(0, index=df.index)

            # Momentum signal
            momentum_col = f'{ticker}_Momentum'
            if momentum_col in df.columns:
                momentum_signal = np.where(df[momentum_col] > 0, 1, -1)
                signal += momentum_signal * 0.4

            # RSI signal
            rsi_col = f'{ticker}_RSI'
            if rsi_col in df.columns:
                rsi_signal = np.where(df[rsi_col] < 30, 1,
                                      np.where(df[rsi_col] > 70, -1, 0))
                signal += rsi_signal * 0.3

            # MACD signal
            macd_col = f'{ticker}_MACD'
            macd_signal_col = f'{ticker}_MACD_Signal'
            if macd_col in df.columns and macd_signal_col in df.columns:
                macd_signal = np.where(df[macd_col] > df[macd_signal_col], 1, -1)
                signal += macd_signal * 0.3

            # Normalize signal to -1, 0, 1
            signals[ticker] = np.where(signal > 0.5, 1,
                                       np.where(signal < -0.5, -1, 0))

        # Fill NaN values with 0 (neutral)
        signals = signals.fillna(0)

        print(f"Generated signals for tickers: {list(signals.columns)}")
        return signals

    def backtest(self, df, signals):
        """Backtest strategy using signals."""
        returns = pd.DataFrame(index=df.index)

        # Get tickers and their corresponding close columns
        tickers = list(signals.columns)

        for ticker in tickers:
            close_col = f'{ticker}_Adj Close' if f'{ticker}_Adj Close' in df.columns else 'Adj Close'

            if close_col in df.columns:
                # Calculate daily returns
                daily_returns = df[close_col].pct_change()

                # Apply signals with 1-day lag (to avoid look-ahead bias)
                returns[ticker] = daily_returns * signals[ticker].shift(1)

        # Calculate portfolio returns (equal weighted)
        if not returns.empty:
            portfolio_returns = returns.mean(axis=1)
            portfolio_returns.name = 'Portfolio'

            # Calculate performance metrics
            total_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)

            print(f"Backtest Results:")
            print(f"Total Return: {total_return:.2%}")
            print(f"Volatility: {volatility:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

            return portfolio_returns
        else:
            raise ValueError("No portfolio returns generated.")

    def _get_tickers_from_columns(self, df):
        """Extract unique tickers from dataframe columns."""
        tickers = set()
        for col in df.columns:
            if '_' in col:
                ticker = col.split('_')[0]
                tickers.add(ticker)
            elif col == 'Adj Close':
                tickers.add('Stock')
        return list(tickers)
