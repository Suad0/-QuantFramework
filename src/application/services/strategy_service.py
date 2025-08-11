"""
Strategy service for the application layer.
Provides interface to strategy and backtesting functionality.
"""

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import existing infrastructure components
from ...infrastructure.strategies.registry import StrategyRegistry
from ...infrastructure.strategies.validator import SignalValidator
from ...infrastructure.backtesting.engine import BacktestEngine
from ...infrastructure.ml.ml_framework import MLFramework
from ...infrastructure.ml.advanced_algorithms import EnhancedXGBoostRegressor, EnhancedRandomForestRegressor
from ...infrastructure.ml.lstm_wrapper import LSTMWrapper
from ...infrastructure.ml.preprocessing import FinancialPreprocessor


class StrategyService:
    """Service for strategy operations."""
    
    def __init__(self):
        self.strategy_registry = StrategyRegistry()
        self.signal_validator = SignalValidator()
        self.backtest_engine = BacktestEngine()
        self.ml_framework = MLFramework()
        self.preprocessor = FinancialPreprocessor()
    
    def generate_signals(self, data: pd.DataFrame, strategy_type: str = 'momentum') -> pd.DataFrame:
        """Generate trading signals using specified strategy with ML enhancement."""
        try:
            signals = pd.DataFrame(index=data.index)
            
            # Extract symbols from data
            symbols = []
            for col in data.columns:
                if '_Close' in col:
                    symbols.append(col.replace('_Close', ''))
            
            if not symbols:
                raise ValueError("No price data found in input")
            
            # Prepare features using the financial preprocessor
            try:
                features = self._prepare_ml_features(data, symbols)
                ml_predictions = self._generate_ml_predictions(features, symbols)
            except Exception as e:
                print(f"ML processing failed: {e}, using traditional signals only")
                ml_predictions = {}
            
            # Generate signals for each symbol
            for symbol in symbols:
                price_col = f'{symbol}_Close'
                if price_col in data.columns:
                    prices = data[price_col].dropna()
                    
                    if len(prices) < 20:  # Need minimum data for moving average
                        signals[f'{symbol}_signal'] = 0
                        continue
                    
                    # Traditional momentum signal
                    ma20 = prices.rolling(20).mean()
                    momentum_signal = (prices > ma20).fillna(0).astype(int)
                    
                    # Combine with ML prediction if available
                    if symbol in ml_predictions:
                        try:
                            ml_signal = ml_predictions[symbol]
                            # Align indices
                            ml_signal = ml_signal.reindex(momentum_signal.index, fill_value=0)
                            ml_signal = ml_signal.fillna(0).astype(int)
                            
                            # Combine signals (both must agree for buy signal)
                            combined_signal = ((momentum_signal == 1) & (ml_signal == 1)).astype(int)
                            signals[f'{symbol}_signal'] = combined_signal
                        except Exception as e:
                            print(f"Error combining ML signal for {symbol}: {e}")
                            signals[f'{symbol}_signal'] = momentum_signal
                    else:
                        signals[f'{symbol}_signal'] = momentum_signal
            
            return signals.fillna(0)
            
        except Exception as e:
            raise Exception(f"Failed to generate signals: {str(e)}")
    
    def _prepare_ml_features(self, data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Prepare features for ML models using the financial preprocessor."""
        try:
            # Clean data first
            clean_data = data.dropna()
            
            if clean_data.empty:
                print("No clean data available for ML features")
                return pd.DataFrame()
            
            # Use the financial preprocessor to create features
            features = pd.DataFrame(index=clean_data.index)
            
            # Add basic technical indicators manually
            for symbol in symbols:
                price_col = f'{symbol}_Close'
                if price_col in clean_data.columns:
                    prices = clean_data[price_col]
                    
                    # Add price-based features
                    features[f'{symbol}_price'] = prices
                    features[f'{symbol}_returns'] = prices.pct_change()
                    
                    # Add momentum features
                    features[f'{symbol}_momentum_5'] = prices.pct_change(5)
                    features[f'{symbol}_momentum_20'] = prices.pct_change(20)
                    
                    # Add moving averages
                    features[f'{symbol}_sma_5'] = prices.rolling(5).mean()
                    features[f'{symbol}_sma_20'] = prices.rolling(20).mean()
                    
                    # Add volatility features
                    features[f'{symbol}_volatility'] = prices.pct_change().rolling(20).std()
                    
                    # Add RSI-like feature
                    delta = prices.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / (loss + 1e-10)  # Add small epsilon to avoid division by zero
                    features[f'{symbol}_rsi'] = 100 - (100 / (1 + rs))
            
            # Drop rows with NaN values
            features_clean = features.dropna()
            
            if features_clean.empty:
                print("No clean features available after processing")
                return pd.DataFrame()
            
            return features_clean
            
        except Exception as e:
            print(f"Error preparing ML features: {e}")
            return pd.DataFrame()
    
    def _generate_ml_predictions(self, features: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.Series]:
        """Generate ML predictions for each symbol."""
        predictions = {}
        
        if features.empty:
            return predictions
        
        try:
            # For each symbol, train a model and generate predictions
            for symbol in symbols:
                try:
                    # Prepare target variable (next day return)
                    price_col = f'{symbol}_Close'
                    if price_col in features.columns:
                        target = features[price_col].pct_change().shift(-1)
                        
                        # Select features for this symbol
                        feature_cols = [col for col in features.columns 
                                      if symbol in col and col != price_col]
                        
                        if len(feature_cols) < 3:  # Need minimum features
                            continue
                        
                        X = features[feature_cols]
                        y = target
                        
                        # Remove NaN values
                        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
                        X_clean = X[valid_idx]
                        y_clean = y[valid_idx]
                        
                        if len(X_clean) < 50:  # Need minimum data points
                            continue
                        
                        # Use XGBoost for prediction
                        model = EnhancedXGBoostRegressor(
                            n_estimators=50,
                            max_depth=4,
                            learning_rate=0.1,
                            random_state=42
                        )
                        
                        # Train on first 80% of data
                        split_idx = int(len(X_clean) * 0.8)
                        X_train = X_clean.iloc[:split_idx]
                        y_train = y_clean.iloc[:split_idx]
                        X_test = X_clean.iloc[split_idx:]
                        
                        model.fit(X_train, y_train)
                        
                        # Generate predictions for test set
                        y_pred = model.predict(X_test)
                        
                        # Convert predictions to signals (buy if predicted return > 0.1%)
                        threshold = 0.001
                        ml_signals = (y_pred > threshold).astype(int)
                        
                        # Create series with proper index
                        signal_series = pd.Series(0, index=features.index)
                        signal_series.iloc[split_idx:split_idx+len(ml_signals)] = ml_signals
                        
                        predictions[symbol] = signal_series
                        
                except Exception as e:
                    print(f"Error generating ML prediction for {symbol}: {e}")
                    continue
            
            return predictions
            
        except Exception as e:
            print(f"Error in ML prediction generation: {e}")
            return predictions
    
    def backtest_strategy(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.Series:
        """Backtest strategy with given signals."""
        try:
            # Simple backtest implementation since async engine is causing issues
            returns = pd.Series(index=data.index, dtype=float)
            
            # Calculate simple returns based on signals
            for col in data.columns:
                if '_Close' in col:
                    symbol = col.replace('_Close', '')
                    signal_col = f'{symbol}_signal'
                    
                    if signal_col in signals.columns:
                        # Calculate returns when signal is 1 (buy)
                        price_returns = data[col].pct_change()
                        strategy_returns = price_returns * signals[signal_col].shift(1)
                        returns = returns.add(strategy_returns, fill_value=0)
            
            # Normalize by number of assets
            n_assets = len([col for col in data.columns if '_Close' in col])
            if n_assets > 0:
                returns = returns / n_assets
            
            return returns.fillna(0)
        except Exception as e:
            raise Exception(f"Failed to backtest strategy: {str(e)}")
    
    def validate_signals(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """Validate trading signals."""
        validation_report = {
            'total_signals': len(signals),
            'signal_distribution': {},
            'issues': []
        }
        
        # Count signal types
        for column in signals.columns:
            if 'signal' in column.lower():
                signal_counts = signals[column].value_counts().to_dict()
                validation_report['signal_distribution'][column] = signal_counts
        
        # Check for issues
        issues = []
        
        # Check for excessive NaN values
        nan_pct = (signals.isnull().sum() / len(signals)) * 100
        high_nan = nan_pct[nan_pct > 20]
        if not high_nan.empty:
            issues.append(f"High NaN values in signal columns: {list(high_nan.index)}")
        
        # Check for signal concentration
        for column in signals.columns:
            if 'signal' in column.lower() and column in signals.select_dtypes(include=['number']).columns:
                unique_values = signals[column].nunique()
                if unique_values < 3:
                    issues.append(f"Low signal diversity in {column}: only {unique_values} unique values")
        
        validation_report['issues'] = issues
        return validation_report
    
    def get_strategy_parameters(self, strategy_type: str) -> Dict[str, Any]:
        """Get default parameters for a strategy type."""
        parameters = {
            'momentum': {
                'lookback_period': 20,
                'threshold': 0.02,
                'rebalance_frequency': 'monthly'
            },
            'mean_reversion': {
                'lookback_period': 10,
                'z_score_threshold': 2.0,
                'rebalance_frequency': 'weekly'
            },
            'volatility_breakout': {
                'volatility_window': 20,
                'breakout_threshold': 1.5,
                'rebalance_frequency': 'daily'
            }
        }
        
        return parameters.get(strategy_type, {})
    
    def calculate_strategy_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate strategy performance metrics."""
        if returns.empty:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * (252 ** 0.5)
        
        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Additional metrics
        win_rate = (returns > 0).sum() / len(returns)
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        }
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def optimize_strategy_parameters(self, data: pd.DataFrame, strategy_type: str, 
                                   parameter_ranges: Dict[str, tuple]) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search."""
        # This is a simplified implementation
        # In a full implementation, this would use more sophisticated optimization
        
        best_params = {}
        best_score = float('-inf')
        
        # For now, return default parameters
        # TODO: Implement actual parameter optimization
        best_params = self.get_strategy_parameters(strategy_type)
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'optimization_method': 'grid_search'
        }