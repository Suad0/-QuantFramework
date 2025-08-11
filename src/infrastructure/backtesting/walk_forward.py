"""
Walk-Forward Analysis and Out-of-Sample Testing.

Implements sophisticated validation techniques to prevent overfitting and ensure
robust strategy performance evaluation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...domain.exceptions import ValidationError, BacktestError
from ...domain.interfaces import IStrategy
from ...domain.value_objects import PerformanceMetrics


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward analysis window."""
    
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Results
    optimal_parameters: Optional[Dict[str, Any]] = None
    in_sample_performance: Optional[PerformanceMetrics] = None
    out_of_sample_performance: Optional[PerformanceMetrics] = None
    
    @property
    def train_period_days(self) -> int:
        """Get training period length in days."""
        return (self.train_end - self.train_start).days
    
    @property
    def test_period_days(self) -> int:
        """Get testing period length in days."""
        return (self.test_end - self.test_start).days


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    
    # Window configuration
    train_period_months: int = 12  # Training period length
    test_period_months: int = 3    # Out-of-sample test period length
    step_months: int = 1           # Step size between windows
    
    # Optimization configuration
    optimization_metric: str = 'sharpe_ratio'  # Metric to optimize
    parameter_ranges: Dict[str, List] = None   # Parameter ranges to test
    max_iterations: int = 100                  # Maximum optimization iterations
    
    # Validation configuration
    min_trades_per_window: int = 10           # Minimum trades required
    min_train_period_days: int = 252          # Minimum training period
    
    # Performance configuration
    use_parallel_processing: bool = True       # Enable parallel processing
    max_workers: int = 4                      # Maximum worker threads
    
    def __post_init__(self):
        if self.parameter_ranges is None:
            self.parameter_ranges = {}


class WalkForwardAnalyzer:
    """Performs walk-forward analysis and out-of-sample testing."""
    
    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
        self.windows: List[WalkForwardWindow] = []
        self.results: Dict[str, Any] = {}
    
    def run_walk_forward_analysis(
        self,
        strategy: IStrategy,
        market_data: pd.DataFrame,
        backtest_engine: 'IBacktestEngine',
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Run complete walk-forward analysis."""
        
        # Generate walk-forward windows
        self.windows = self._generate_windows(start_date, end_date)
        
        if len(self.windows) == 0:
            raise ValidationError("No valid walk-forward windows generated")
        
        # Run analysis for each window
        if self.config.use_parallel_processing:
            self._run_parallel_analysis(strategy, market_data, backtest_engine)
        else:
            self._run_sequential_analysis(strategy, market_data, backtest_engine)
        
        # Aggregate and analyze results
        self.results = self._aggregate_results()
        
        return self.results
    
    def _generate_windows(self, start_date: datetime, end_date: datetime) -> List[WalkForwardWindow]:
        """Generate walk-forward analysis windows."""
        
        windows = []
        window_id = 0
        
        current_date = start_date
        
        while current_date < end_date:
            # Calculate training period
            train_start = current_date
            train_end = train_start + timedelta(days=self.config.train_period_months * 30)
            
            # Calculate test period
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.config.test_period_months * 30)
            
            # Check if we have enough data
            if test_end > end_date:
                break
            
            # Validate minimum training period
            if (train_end - train_start).days < self.config.min_train_period_days:
                current_date += timedelta(days=self.config.step_months * 30)
                continue
            
            window = WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end
            )
            
            windows.append(window)
            window_id += 1
            
            # Move to next window
            current_date += timedelta(days=self.config.step_months * 30)
        
        return windows
    
    def _run_parallel_analysis(
        self,
        strategy: IStrategy,
        market_data: pd.DataFrame,
        backtest_engine: 'IBacktestEngine'
    ) -> None:
        """Run walk-forward analysis in parallel."""
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all window analyses
            future_to_window = {
                executor.submit(
                    self._analyze_window,
                    window,
                    strategy,
                    market_data,
                    backtest_engine
                ): window
                for window in self.windows
            }
            
            # Collect results
            for future in as_completed(future_to_window):
                window = future_to_window[future]
                try:
                    result = future.result()
                    # Update window with results
                    window.optimal_parameters = result['optimal_parameters']
                    window.in_sample_performance = result['in_sample_performance']
                    window.out_of_sample_performance = result['out_of_sample_performance']
                except Exception as e:
                    print(f"Error analyzing window {window.window_id}: {str(e)}")
    
    def _run_sequential_analysis(
        self,
        strategy: IStrategy,
        market_data: pd.DataFrame,
        backtest_engine: 'IBacktestEngine'
    ) -> None:
        """Run walk-forward analysis sequentially."""
        
        for window in self.windows:
            try:
                result = self._analyze_window(window, strategy, market_data, backtest_engine)
                window.optimal_parameters = result['optimal_parameters']
                window.in_sample_performance = result['in_sample_performance']
                window.out_of_sample_performance = result['out_of_sample_performance']
            except Exception as e:
                print(f"Error analyzing window {window.window_id}: {str(e)}")
    
    def _analyze_window(
        self,
        window: WalkForwardWindow,
        strategy: IStrategy,
        market_data: pd.DataFrame,
        backtest_engine: 'IBacktestEngine'
    ) -> Dict[str, Any]:
        """Analyze a single walk-forward window."""
        
        # Extract training and testing data
        train_data = market_data[
            (market_data.index >= window.train_start) & 
            (market_data.index <= window.train_end)
        ]
        
        test_data = market_data[
            (market_data.index >= window.test_start) & 
            (market_data.index <= window.test_end)
        ]
        
        if len(train_data) == 0 or len(test_data) == 0:
            raise BacktestError(f"Insufficient data for window {window.window_id}")
        
        # Optimize parameters on training data
        optimal_params = self._optimize_parameters(strategy, train_data, backtest_engine)
        
        # Test in-sample performance
        strategy.update_parameters(optimal_params)
        in_sample_config = {
            'start_date': window.train_start,
            'end_date': window.train_end,
            'initial_capital': 100000,
            'market_data': train_data,
            'enable_corporate_actions': False,
            'enable_bias_detection': False
        }
        
        # Use asyncio to run the backtest
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        in_sample_result = loop.run_until_complete(
            backtest_engine.run_backtest(strategy, in_sample_config)
        )
        in_sample_performance = in_sample_result.get('performance')
        
        # Test out-of-sample performance
        out_of_sample_config = {
            'start_date': window.test_start,
            'end_date': window.test_end,
            'initial_capital': 100000,
            'market_data': test_data,
            'enable_corporate_actions': False,
            'enable_bias_detection': False
        }
        
        out_of_sample_result = loop.run_until_complete(
            backtest_engine.run_backtest(strategy, out_of_sample_config)
        )
        out_of_sample_performance = out_of_sample_result.get('performance')
        
        # Validate minimum trades requirement
        in_sample_trades = in_sample_result.get('trading', {}).get('total_trades', 0)
        out_of_sample_trades = out_of_sample_result.get('trading', {}).get('total_trades', 0)
        
        if in_sample_trades < self.config.min_trades_per_window:
            print(f"Warning: Window {window.window_id} has insufficient in-sample trades ({in_sample_trades})")
        
        if out_of_sample_trades < self.config.min_trades_per_window:
            print(f"Warning: Window {window.window_id} has insufficient out-of-sample trades ({out_of_sample_trades})")
        
        return {
            'optimal_parameters': optimal_params,
            'in_sample_performance': in_sample_performance,
            'out_of_sample_performance': out_of_sample_performance,
            'in_sample_trades': in_sample_trades,
            'out_of_sample_trades': out_of_sample_trades
        }
    
    def _optimize_parameters(
        self,
        strategy: IStrategy,
        train_data: pd.DataFrame,
        backtest_engine: 'IBacktestEngine'
    ) -> Dict[str, Any]:
        """Optimize strategy parameters on training data."""
        
        if not self.config.parameter_ranges:
            return strategy.get_parameters()
        
        best_params = strategy.get_parameters()
        best_score = float('-inf')
        
        # Simple grid search optimization
        # In practice, you might want to use more sophisticated optimization
        param_combinations = self._generate_parameter_combinations()
        
        for params in param_combinations[:self.config.max_iterations]:
            try:
                # Update strategy parameters
                strategy.update_parameters(params)
                
                # Run backtest on training data
                config = {
                    'start_date': train_data.index[0],
                    'end_date': train_data.index[-1],
                    'initial_capital': 100000
                }
                
                result = backtest_engine.run_backtest(strategy, config)
                performance = result.get('performance_metrics')
                
                if performance and hasattr(performance, self.config.optimization_metric):
                    score = getattr(performance, self.config.optimization_metric)
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                
            except Exception as e:
                # Skip this parameter combination if it fails
                continue
        
        return best_params
    
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for optimization."""
        
        import itertools
        
        param_names = list(self.config.parameter_ranges.keys())
        param_values = list(self.config.parameter_ranges.values())
        
        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results from all walk-forward windows."""
        
        valid_windows = [w for w in self.windows if w.out_of_sample_performance is not None]
        
        if len(valid_windows) == 0:
            return {'error': 'No valid windows with results'}
        
        # Collect out-of-sample performance metrics
        oos_returns = []
        oos_sharpe_ratios = []
        oos_max_drawdowns = []
        oos_win_rates = []
        
        for window in valid_windows:
            perf = window.out_of_sample_performance
            oos_returns.append(perf.annualized_return)
            oos_sharpe_ratios.append(perf.sharpe_ratio)
            oos_max_drawdowns.append(perf.max_drawdown)
            oos_win_rates.append(perf.win_rate)
        
        # Calculate aggregate statistics
        results = {
            'total_windows': len(self.windows),
            'valid_windows': len(valid_windows),
            'out_of_sample_stats': {
                'mean_return': np.mean(oos_returns),
                'std_return': np.std(oos_returns),
                'mean_sharpe': np.mean(oos_sharpe_ratios),
                'std_sharpe': np.std(oos_sharpe_ratios),
                'mean_max_drawdown': np.mean(oos_max_drawdowns),
                'mean_win_rate': np.mean(oos_win_rates),
                'consistency_ratio': len([r for r in oos_returns if r > 0]) / len(oos_returns)
            },
            'parameter_stability': self._analyze_parameter_stability(),
            'performance_degradation': self._analyze_performance_degradation(),
            'windows': [
                {
                    'window_id': w.window_id,
                    'train_period': f"{w.train_start.date()} to {w.train_end.date()}",
                    'test_period': f"{w.test_start.date()} to {w.test_end.date()}",
                    'optimal_parameters': w.optimal_parameters,
                    'in_sample_sharpe': w.in_sample_performance.sharpe_ratio if w.in_sample_performance else None,
                    'out_of_sample_sharpe': w.out_of_sample_performance.sharpe_ratio if w.out_of_sample_performance else None
                }
                for w in valid_windows
            ]
        }
        
        return results
    
    def _analyze_parameter_stability(self) -> Dict[str, Any]:
        """Analyze stability of optimal parameters across windows."""
        
        valid_windows = [w for w in self.windows if w.optimal_parameters is not None]
        
        if len(valid_windows) < 2:
            return {'error': 'Insufficient windows for parameter stability analysis'}
        
        # Collect parameter values across windows
        param_stability = {}
        
        # Get all parameter names
        all_param_names = set()
        for window in valid_windows:
            all_param_names.update(window.optimal_parameters.keys())
        
        for param_name in all_param_names:
            param_values = []
            for window in valid_windows:
                if param_name in window.optimal_parameters:
                    param_values.append(window.optimal_parameters[param_name])
            
            if len(param_values) > 1:
                param_stability[param_name] = {
                    'mean': np.mean(param_values),
                    'std': np.std(param_values),
                    'coefficient_of_variation': np.std(param_values) / np.mean(param_values) if np.mean(param_values) != 0 else float('inf')
                }
        
        return param_stability
    
    def _analyze_performance_degradation(self) -> Dict[str, Any]:
        """Analyze performance degradation from in-sample to out-of-sample."""
        
        valid_windows = [
            w for w in self.windows 
            if w.in_sample_performance is not None and w.out_of_sample_performance is not None
        ]
        
        if len(valid_windows) == 0:
            return {'error': 'No windows with both in-sample and out-of-sample results'}
        
        degradation_metrics = []
        
        for window in valid_windows:
            is_perf = window.in_sample_performance
            oos_perf = window.out_of_sample_performance
            
            degradation = {
                'window_id': window.window_id,
                'return_degradation': is_perf.annualized_return - oos_perf.annualized_return,
                'sharpe_degradation': is_perf.sharpe_ratio - oos_perf.sharpe_ratio,
                'relative_return_degradation': (is_perf.annualized_return - oos_perf.annualized_return) / abs(is_perf.annualized_return) if is_perf.annualized_return != 0 else 0,
                'relative_sharpe_degradation': (is_perf.sharpe_ratio - oos_perf.sharpe_ratio) / abs(is_perf.sharpe_ratio) if is_perf.sharpe_ratio != 0 else 0
            }
            degradation_metrics.append(degradation)
        
        # Calculate aggregate degradation statistics
        return_degradations = [d['return_degradation'] for d in degradation_metrics]
        sharpe_degradations = [d['sharpe_degradation'] for d in degradation_metrics]
        
        return {
            'mean_return_degradation': np.mean(return_degradations),
            'mean_sharpe_degradation': np.mean(sharpe_degradations),
            'degradation_consistency': len([d for d in return_degradations if d > 0]) / len(return_degradations),
            'severe_degradation_windows': len([d for d in sharpe_degradations if d > 1.0]),
            'window_details': degradation_metrics
        }
    
    def get_summary_report(self) -> str:
        """Generate a summary report of the walk-forward analysis."""
        
        if not self.results:
            return "No walk-forward analysis results available."
        
        report = []
        report.append("=== Walk-Forward Analysis Summary ===")
        report.append(f"Total Windows: {self.results['total_windows']}")
        report.append(f"Valid Windows: {self.results['valid_windows']}")
        report.append("")
        
        if 'out_of_sample_stats' in self.results:
            oos_stats = self.results['out_of_sample_stats']
            report.append("Out-of-Sample Performance:")
            report.append(f"  Mean Return: {oos_stats['mean_return']:.2%}")
            report.append(f"  Return Std Dev: {oos_stats['std_return']:.2%}")
            report.append(f"  Mean Sharpe Ratio: {oos_stats['mean_sharpe']:.2f}")
            report.append(f"  Sharpe Std Dev: {oos_stats['std_sharpe']:.2f}")
            report.append(f"  Consistency Ratio: {oos_stats['consistency_ratio']:.2%}")
            report.append("")
        
        if 'performance_degradation' in self.results and 'error' not in self.results['performance_degradation']:
            deg_stats = self.results['performance_degradation']
            report.append("Performance Degradation (In-Sample vs Out-of-Sample):")
            report.append(f"  Mean Return Degradation: {deg_stats['mean_return_degradation']:.2%}")
            report.append(f"  Mean Sharpe Degradation: {deg_stats['mean_sharpe_degradation']:.2f}")
            report.append(f"  Degradation Consistency: {deg_stats['degradation_consistency']:.2%}")
            report.append("")
        
        return "\n".join(report)