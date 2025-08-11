"""
Financial-specific evaluation metrics for ML models.

This module provides metrics specifically designed for evaluating machine learning
models in financial applications, including directional accuracy, profit-based
metrics, and risk-adjusted performance measures.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import warnings
from datetime import datetime

from src.domain.exceptions import ValidationError


class FinancialMetricsCalculator:
    """Calculator for financial-specific ML model evaluation metrics."""
    
    def __init__(self, transaction_cost: float = 0.001, risk_free_rate: float = 0.02):
        """
        Initialize financial metrics calculator.
        
        Args:
            transaction_cost: Transaction cost as fraction of trade value
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
    
    def calculate_directional_accuracy(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        method: str = 'returns'
    ) -> Dict[str, float]:
        """
        Calculate directional accuracy metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            method: Method for calculating direction ('returns', 'levels', 'changes')
            
        Returns:
            Dictionary with directional accuracy metrics
        """
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)
        
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValidationError("y_true and y_pred must have the same length")
        
        if method == 'returns':
            # Calculate direction based on returns (differences)
            true_direction = np.sign(y_true.diff().dropna())
            pred_direction = np.sign(y_pred.diff().dropna())
        elif method == 'levels':
            # Calculate direction based on levels (sign)
            true_direction = np.sign(y_true)
            pred_direction = np.sign(y_pred)
        elif method == 'changes':
            # Calculate direction based on changes from mean
            true_mean = y_true.mean()
            pred_mean = y_pred.mean()
            true_direction = np.sign(y_true - true_mean)
            pred_direction = np.sign(y_pred - pred_mean)
        else:
            raise ValidationError(f"Unknown method: {method}")
        
        # Align series (in case of different lengths due to diff())
        min_len = min(len(true_direction), len(pred_direction))
        true_direction = true_direction.iloc[-min_len:]
        pred_direction = pred_direction.iloc[-min_len:]
        
        # Calculate directional accuracy
        correct_directions = (true_direction == pred_direction).sum()
        total_predictions = len(true_direction)
        directional_accuracy = correct_directions / total_predictions if total_predictions > 0 else 0.0
        
        # Calculate hit rate (percentage of correct sign predictions)
        hit_rate = directional_accuracy
        
        # Calculate up/down capture ratios
        up_mask = true_direction > 0
        down_mask = true_direction < 0
        
        up_accuracy = (true_direction[up_mask] == pred_direction[up_mask]).mean() if up_mask.sum() > 0 else 0.0
        down_accuracy = (true_direction[down_mask] == pred_direction[down_mask]).mean() if down_mask.sum() > 0 else 0.0
        
        return {
            'directional_accuracy': directional_accuracy,
            'hit_rate': hit_rate,
            'up_accuracy': up_accuracy,
            'down_accuracy': down_accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_directions,
            'method': method
        }
    
    def calculate_profit_based_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        prices: Optional[pd.Series] = None,
        initial_capital: float = 100000.0
    ) -> Dict[str, float]:
        """
        Calculate profit-based evaluation metrics.
        
        Args:
            y_true: True returns or price changes
            y_pred: Predicted returns or price changes
            prices: Price series (optional, for position sizing)
            initial_capital: Initial capital for backtesting
            
        Returns:
            Dictionary with profit-based metrics
        """
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)
        
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValidationError("y_true and y_pred must have the same length")
        
        # Generate trading signals based on predictions
        signals = np.sign(y_pred)
        
        # Calculate strategy returns
        strategy_returns = signals * y_true
        
        # Apply transaction costs
        position_changes = signals.diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        net_returns = strategy_returns - transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + net_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Calculate annualized return (assuming daily data)
        n_periods = len(net_returns)
        periods_per_year = 252  # Trading days per year
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # Calculate volatility
        volatility = net_returns.std() * np.sqrt(periods_per_year)
        
        # Calculate Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        
        # Calculate maximum drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate win rate
        winning_trades = (net_returns > 0).sum()
        total_trades = (signals != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate profit factor
        gross_profit = net_returns[net_returns > 0].sum()
        gross_loss = abs(net_returns[net_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'transaction_costs_total': transaction_costs.sum()
        }
    
    def calculate_risk_adjusted_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted performance metrics.
        
        Args:
            y_true: True returns
            y_pred: Predicted returns
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary with risk-adjusted metrics
        """
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)
        
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred)
        
        # Generate strategy returns
        signals = np.sign(y_pred)
        strategy_returns = signals * y_true
        
        # Calculate basic statistics
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0.0
        sortino_ratio = (mean_return - self.risk_free_rate / 252) / downside_deviation if downside_deviation > 0 else 0.0
        
        # Calculate Information Ratio (if benchmark provided)
        information_ratio = 0.0
        tracking_error = 0.0
        if benchmark_returns is not None:
            if len(benchmark_returns) == len(strategy_returns):
                excess_returns = strategy_returns - benchmark_returns
                tracking_error = excess_returns.std()
                information_ratio = excess_returns.mean() / tracking_error if tracking_error > 0 else 0.0
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(strategy_returns, 5)
        var_99 = np.percentile(strategy_returns, 1)
        
        # CVaR (Expected Shortfall)
        cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
        cvar_99 = strategy_returns[strategy_returns <= var_99].mean()
        
        # Calculate skewness and kurtosis
        skewness = stats.skew(strategy_returns)
        kurtosis = stats.kurtosis(strategy_returns)
        
        return {
            'mean_return': mean_return,
            'std_return': std_return,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def calculate_prediction_quality_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        confidence_intervals: Optional[Tuple[pd.Series, pd.Series]] = None
    ) -> Dict[str, float]:
        """
        Calculate prediction quality metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence_intervals: Tuple of (lower_bound, upper_bound) series
            
        Returns:
            Dictionary with prediction quality metrics
        """
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)
        
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred)
        
        # Basic accuracy metrics
        mse = ((y_true - y_pred) ** 2).mean()
        mae = abs(y_true - y_pred).mean()
        rmse = np.sqrt(mse)
        
        # Correlation metrics
        correlation = y_true.corr(y_pred)
        rank_correlation = y_true.corr(y_pred, method='spearman')
        
        # Prediction bias
        bias = (y_pred - y_true).mean()
        
        # Theil's U statistic (forecast accuracy)
        numerator = np.sqrt(((y_pred - y_true) ** 2).mean())
        denominator = np.sqrt((y_true ** 2).mean()) + np.sqrt((y_pred ** 2).mean())
        theil_u = numerator / denominator if denominator > 0 else np.inf
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'rank_correlation': rank_correlation,
            'bias': bias,
            'theil_u': theil_u
        }
        
        # Confidence interval metrics (if provided)
        if confidence_intervals is not None:
            lower_bound, upper_bound = confidence_intervals
            
            # Coverage probability
            within_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
            coverage_probability = within_interval.mean()
            
            # Average interval width
            interval_width = (upper_bound - lower_bound).mean()
            
            # Interval score (lower is better)
            alpha = 0.05  # For 95% confidence interval
            interval_score = self._calculate_interval_score(y_true, lower_bound, upper_bound, alpha)
            
            metrics.update({
                'coverage_probability': coverage_probability,
                'interval_width': interval_width,
                'interval_score': interval_score
            })
        
        return metrics
    
    def _calculate_interval_score(
        self,
        y_true: pd.Series,
        lower: pd.Series,
        upper: pd.Series,
        alpha: float
    ) -> float:
        """
        Calculate interval score for prediction intervals.
        
        Args:
            y_true: True values
            lower: Lower bound of prediction interval
            upper: Upper bound of prediction interval
            alpha: Significance level (e.g., 0.05 for 95% interval)
            
        Returns:
            Interval score (lower is better)
        """
        width = upper - lower
        
        # Penalties for being outside the interval
        lower_penalty = 2 * alpha * np.maximum(lower - y_true, 0)
        upper_penalty = 2 * alpha * np.maximum(y_true - upper, 0)
        
        # Total interval score
        interval_score = width + lower_penalty + upper_penalty
        
        return interval_score.mean()
    
    def calculate_comprehensive_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        prices: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None,
        confidence_intervals: Optional[Tuple[pd.Series, pd.Series]] = None,
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive financial metrics for model evaluation.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            prices: Price series (optional)
            benchmark_returns: Benchmark returns (optional)
            confidence_intervals: Prediction intervals (optional)
            initial_capital: Initial capital for backtesting
            
        Returns:
            Dictionary with all calculated metrics
        """
        results = {
            'timestamp': datetime.now(),
            'n_observations': len(y_true)
        }
        
        try:
            # Directional accuracy metrics
            directional_metrics = self.calculate_directional_accuracy(y_true, y_pred)
            results['directional'] = directional_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate directional metrics: {e}")
            results['directional'] = {}
        
        try:
            # Profit-based metrics
            profit_metrics = self.calculate_profit_based_metrics(
                y_true, y_pred, prices, initial_capital
            )
            results['profit'] = profit_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate profit metrics: {e}")
            results['profit'] = {}
        
        try:
            # Risk-adjusted metrics
            risk_metrics = self.calculate_risk_adjusted_metrics(
                y_true, y_pred, benchmark_returns
            )
            results['risk_adjusted'] = risk_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate risk-adjusted metrics: {e}")
            results['risk_adjusted'] = {}
        
        try:
            # Prediction quality metrics
            quality_metrics = self.calculate_prediction_quality_metrics(
                y_true, y_pred, confidence_intervals
            )
            results['prediction_quality'] = quality_metrics
        except Exception as e:
            warnings.warn(f"Failed to calculate prediction quality metrics: {e}")
            results['prediction_quality'] = {}
        
        return results