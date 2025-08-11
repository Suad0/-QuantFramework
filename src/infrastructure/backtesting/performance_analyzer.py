"""
Performance Analyzer for comprehensive backtesting metrics.

Calculates 20+ performance and risk metrics for strategy evaluation.
Includes performance attribution, benchmark comparison, and statistical significance testing.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_1samp, jarque_bera, normaltest

from ...domain.exceptions import ValidationError
from ...domain.value_objects import PerformanceMetrics, RiskMetrics
from .performance_attribution import PerformanceAttributionAnalyzer, AttributionResult
from .benchmark_comparison import BenchmarkComparator, BenchmarkComparisonResult


@dataclass
class DetailedPerformanceMetrics:
    """Extended performance metrics for comprehensive analysis."""
    
    # Basic return metrics
    total_return: float
    annualized_return: float
    volatility: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration: int
    average_drawdown: float
    recovery_factor: float
    
    # Distribution metrics
    skewness: float
    kurtosis: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Additional metrics
    tail_ratio: float
    common_sense_ratio: float
    stability_ratio: float
    
    # Statistical significance
    return_tstat: float
    return_pvalue: float
    sharpe_tstat: float
    sharpe_pvalue: float
    normality_pvalue: float
    
    # Additional advanced metrics
    omega_ratio: float
    kappa_3: float
    gain_loss_ratio: float
    upside_potential_ratio: float
    
    timestamp: datetime


@dataclass
class ComprehensivePerformanceReport:
    """Comprehensive performance analysis including attribution and benchmark comparison."""
    
    performance_metrics: DetailedPerformanceMetrics
    attribution_result: Optional[AttributionResult]
    benchmark_comparison: Optional[BenchmarkComparisonResult]
    statistical_tests: Dict[str, Any]
    
    timestamp: datetime


class PerformanceAnalyzer:
    """Calculates comprehensive performance metrics for backtesting."""
    
    def __init__(self, benchmark_returns: Optional[pd.Series] = None, risk_free_rate: float = 0.02):
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.attribution_analyzer = PerformanceAttributionAnalyzer()
        self.benchmark_comparator = BenchmarkComparator(risk_free_rate)
    
    def calculate_performance_metrics(
        self,
        returns: pd.Series,
        trades: Optional[List[Dict[str, Any]]] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> DetailedPerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if len(returns) == 0:
            raise ValidationError("Cannot calculate metrics for empty returns series")
        
        # Use provided benchmark or default
        benchmark = benchmark_returns if benchmark_returns is not None else self.benchmark_returns
        
        # Calculate basic metrics
        total_return = self._calculate_total_return(returns)
        annualized_return = self._calculate_annualized_return(returns)
        volatility = self._calculate_volatility(returns)
        
        # Calculate risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        information_ratio = self._calculate_information_ratio(returns, benchmark)
        treynor_ratio = self._calculate_treynor_ratio(returns, benchmark)
        
        # Calculate drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(returns)
        
        # Calculate distribution metrics
        distribution_metrics = self._calculate_distribution_metrics(returns)
        
        # Calculate trading metrics
        trading_metrics = self._calculate_trading_metrics(returns, trades)
        
        # Calculate additional metrics
        tail_ratio = self._calculate_tail_ratio(returns)
        common_sense_ratio = self._calculate_common_sense_ratio(returns)
        stability_ratio = self._calculate_stability_ratio(returns)
        
        # Calculate statistical significance
        statistical_metrics = self._calculate_statistical_significance(returns)
        
        # Calculate additional advanced metrics
        omega_ratio = self._calculate_omega_ratio(returns)
        kappa_3 = self._calculate_kappa_3(returns)
        gain_loss_ratio = self._calculate_gain_loss_ratio(returns)
        upside_potential_ratio = self._calculate_upside_potential_ratio(returns)
        
        return DetailedPerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            max_drawdown=drawdown_metrics['max_drawdown'],
            max_drawdown_duration=drawdown_metrics['max_drawdown_duration'],
            average_drawdown=drawdown_metrics['average_drawdown'],
            recovery_factor=drawdown_metrics['recovery_factor'],
            skewness=distribution_metrics['skewness'],
            kurtosis=distribution_metrics['kurtosis'],
            var_95=distribution_metrics['var_95'],
            var_99=distribution_metrics['var_99'],
            cvar_95=distribution_metrics['cvar_95'],
            cvar_99=distribution_metrics['cvar_99'],
            win_rate=trading_metrics['win_rate'],
            profit_factor=trading_metrics['profit_factor'],
            average_win=trading_metrics['average_win'],
            average_loss=trading_metrics['average_loss'],
            largest_win=trading_metrics['largest_win'],
            largest_loss=trading_metrics['largest_loss'],
            tail_ratio=tail_ratio,
            common_sense_ratio=common_sense_ratio,
            stability_ratio=stability_ratio,
            return_tstat=statistical_metrics['return_tstat'],
            return_pvalue=statistical_metrics['return_pvalue'],
            sharpe_tstat=statistical_metrics['sharpe_tstat'],
            sharpe_pvalue=statistical_metrics['sharpe_pvalue'],
            normality_pvalue=statistical_metrics['normality_pvalue'],
            omega_ratio=omega_ratio,
            kappa_3=kappa_3,
            gain_loss_ratio=gain_loss_ratio,
            upside_potential_ratio=upside_potential_ratio,
            timestamp=datetime.now()
        )
    
    def _calculate_total_return(self, returns: pd.Series) -> float:
        """Calculate total return."""
        return float((1 + returns).prod() - 1)
    
    def _calculate_annualized_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        
        total_return = self._calculate_total_return(returns)
        years = len(returns) / periods_per_year
        
        if years <= 0:
            return 0.0
        
        return float((1 + total_return) ** (1 / years) - 1)
    
    def _calculate_volatility(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility."""
        if len(returns) <= 1:
            return 0.0
        
        return float(returns.std() * np.sqrt(periods_per_year))
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / periods_per_year
        
        if excess_returns.std() == 0:
            return 0.0
        
        return float(excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year))
    
    def _calculate_sortino_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_deviation = downside_returns.std()
        return float(excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year))
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        annualized_return = self._calculate_annualized_return(returns)
        max_drawdown = abs(self._calculate_max_drawdown(returns))
        
        if max_drawdown == 0:
            return float('inf') if annualized_return > 0 else 0.0
        
        return annualized_return / max_drawdown
    
    def _calculate_information_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series]
    ) -> float:
        """Calculate Information ratio."""
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns with benchmark
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) <= 1:
            return 0.0
        
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return float(excess_returns.mean() / tracking_error * np.sqrt(252))
    
    def _calculate_treynor_ratio(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series]
    ) -> float:
        """Calculate Treynor ratio."""
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns with benchmark
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) <= 1:
            return 0.0
        
        # Calculate beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        
        if benchmark_variance == 0:
            return 0.0
        
        beta = covariance / benchmark_variance
        
        if beta == 0:
            return 0.0
        
        excess_return = self._calculate_annualized_return(aligned_returns) - self.risk_free_rate
        return excess_return / beta
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive drawdown metrics."""
        if len(returns) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'average_drawdown': 0.0,
                'recovery_factor': 0.0
            }
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Max drawdown
        max_drawdown = float(drawdown.min())
        
        # Max drawdown duration
        max_dd_duration = self._calculate_max_drawdown_duration(drawdown)
        
        # Average drawdown
        negative_drawdowns = drawdown[drawdown < 0]
        average_drawdown = float(negative_drawdowns.mean()) if len(negative_drawdowns) > 0 else 0.0
        
        # Recovery factor
        total_return = self._calculate_total_return(returns)
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'average_drawdown': average_drawdown,
            'recovery_factor': recovery_factor
        }
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        if len(drawdown) == 0:
            return 0
        
        # Find periods where we're in drawdown
        in_drawdown = drawdown < 0
        
        # Find the start and end of each drawdown period
        drawdown_periods = []
        start = None
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start is None:
                start = i
            elif not is_dd and start is not None:
                drawdown_periods.append(i - start)
                start = None
        
        # Handle case where drawdown continues to the end
        if start is not None:
            drawdown_periods.append(len(drawdown) - start)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return float(drawdown.min())
    
    def _calculate_distribution_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate distribution-related metrics."""
        if len(returns) <= 1:
            return {
                'skewness': 0.0,
                'kurtosis': 0.0,
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0
            }
        
        # Skewness and kurtosis
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns))
        
        # Value at Risk (VaR)
        var_95 = float(np.percentile(returns, 5))  # 5th percentile
        var_99 = float(np.percentile(returns, 1))  # 1st percentile
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = float(returns[returns <= var_99].mean()) if len(returns[returns <= var_99]) > 0 else var_99
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99
        }
    
    def _calculate_trading_metrics(
        self,
        returns: pd.Series,
        trades: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        
        default_metrics = {
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }
        
        if not trades:
            # Calculate from returns if trades not available
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0.0
            
            average_win = float(positive_returns.mean()) if len(positive_returns) > 0 else 0.0
            average_loss = float(negative_returns.mean()) if len(negative_returns) > 0 else 0.0
            
            largest_win = float(positive_returns.max()) if len(positive_returns) > 0 else 0.0
            largest_loss = float(negative_returns.min()) if len(negative_returns) > 0 else 0.0
            
            # Profit factor
            total_wins = float(positive_returns.sum()) if len(positive_returns) > 0 else 0.0
            total_losses = abs(float(negative_returns.sum())) if len(negative_returns) > 0 else 0.0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_win': average_win,
                'average_loss': average_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss
            }
        
        # Calculate from trade data
        trade_pnls = []
        for trade in trades:
            pnl = trade.get('pnl', 0)
            if pnl != 0:
                trade_pnls.append(pnl)
        
        if not trade_pnls:
            return default_metrics
        
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(trade_pnls)
        
        average_win = np.mean(winning_trades) if winning_trades else 0.0
        average_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        largest_win = max(winning_trades) if winning_trades else 0.0
        largest_loss = min(losing_trades) if losing_trades else 0.0
        
        total_wins = sum(winning_trades) if winning_trades else 0.0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0.0
        
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'average_win': average_win,
            'average_loss': average_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        if len(returns) <= 1:
            return 0.0
        
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 == 0:
            return float('inf') if p95 > 0 else 0.0
        
        return abs(p95 / p5)
    
    def _calculate_common_sense_ratio(self, returns: pd.Series) -> float:
        """Calculate common sense ratio (tail ratio * profit factor)."""
        if len(returns) <= 1:
            return 0.0
        
        tail_ratio = self._calculate_tail_ratio(returns)
        trading_metrics = self._calculate_trading_metrics(returns, None)
        profit_factor = trading_metrics['profit_factor']
        
        if profit_factor == float('inf'):
            return float('inf')
        
        return tail_ratio * profit_factor
    
    def _calculate_stability_ratio(self, returns: pd.Series, window: int = 252) -> float:
        """Calculate stability ratio (consistency of performance)."""
        if len(returns) < window * 2:
            return 0.0
        
        # Calculate rolling Sharpe ratios
        rolling_sharpe = returns.rolling(window=window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        ).dropna()
        
        if len(rolling_sharpe) <= 1:
            return 0.0
        
        # Stability is measured as 1 - coefficient of variation of rolling Sharpe ratios
        cv = rolling_sharpe.std() / abs(rolling_sharpe.mean()) if rolling_sharpe.mean() != 0 else float('inf')
        
        return max(0.0, 1.0 - cv)
    
    def generate_performance_report(self, metrics: DetailedPerformanceMetrics) -> str:
        """Generate a comprehensive performance report."""
        
        report = []
        report.append("=== PERFORMANCE ANALYSIS REPORT ===")
        report.append(f"Analysis Date: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Return metrics
        report.append("=== RETURN METRICS ===")
        report.append(f"Total Return:        {metrics.total_return:>10.2%}")
        report.append(f"Annualized Return:   {metrics.annualized_return:>10.2%}")
        report.append(f"Volatility:          {metrics.volatility:>10.2%}")
        report.append("")
        
        # Risk-adjusted metrics
        report.append("=== RISK-ADJUSTED METRICS ===")
        report.append(f"Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}")
        report.append(f"Sortino Ratio:       {metrics.sortino_ratio:>10.2f}")
        report.append(f"Calmar Ratio:        {metrics.calmar_ratio:>10.2f}")
        report.append(f"Information Ratio:   {metrics.information_ratio:>10.2f}")
        report.append(f"Treynor Ratio:       {metrics.treynor_ratio:>10.2f}")
        report.append("")
        
        # Drawdown metrics
        report.append("=== DRAWDOWN METRICS ===")
        report.append(f"Max Drawdown:        {metrics.max_drawdown:>10.2%}")
        report.append(f"Max DD Duration:     {metrics.max_drawdown_duration:>10d} periods")
        report.append(f"Average Drawdown:    {metrics.average_drawdown:>10.2%}")
        report.append(f"Recovery Factor:     {metrics.recovery_factor:>10.2f}")
        report.append("")
        
        # Risk metrics
        report.append("=== RISK METRICS ===")
        report.append(f"VaR (95%):           {metrics.var_95:>10.2%}")
        report.append(f"VaR (99%):           {metrics.var_99:>10.2%}")
        report.append(f"CVaR (95%):          {metrics.cvar_95:>10.2%}")
        report.append(f"CVaR (99%):          {metrics.cvar_99:>10.2%}")
        report.append(f"Skewness:            {metrics.skewness:>10.2f}")
        report.append(f"Kurtosis:            {metrics.kurtosis:>10.2f}")
        report.append("")
        
        # Trading metrics
        report.append("=== TRADING METRICS ===")
        report.append(f"Win Rate:            {metrics.win_rate:>10.2%}")
        report.append(f"Profit Factor:       {metrics.profit_factor:>10.2f}")
        report.append(f"Average Win:         {metrics.average_win:>10.2%}")
        report.append(f"Average Loss:        {metrics.average_loss:>10.2%}")
        report.append(f"Largest Win:         {metrics.largest_win:>10.2%}")
        report.append(f"Largest Loss:        {metrics.largest_loss:>10.2%}")
        report.append("")
        
        # Advanced metrics
        report.append("=== ADVANCED METRICS ===")
        report.append(f"Tail Ratio:          {metrics.tail_ratio:>10.2f}")
        report.append(f"Common Sense Ratio:  {metrics.common_sense_ratio:>10.2f}")
        report.append(f"Stability Ratio:     {metrics.stability_ratio:>10.2f}")
        
        return "\n".join(report)
    
    def comprehensive_analysis(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> ComprehensivePerformanceReport:
        """
        Perform comprehensive performance analysis including attribution and benchmark comparison.
        
        Args:
            returns: Portfolio return series
            benchmark_returns: Optional benchmark return series
            factor_returns: Optional factor return DataFrame for attribution
            trades: Optional trade data for detailed analysis
            
        Returns:
            ComprehensivePerformanceReport with all analysis results
        """
        # Calculate basic performance metrics
        performance_metrics = self.calculate_performance_metrics(returns, trades, benchmark_returns)
        
        # Perform attribution analysis if factor returns provided
        attribution_result = None
        if factor_returns is not None and not factor_returns.empty:
            try:
                attribution_result = self.attribution_analyzer.factor_attribution(
                    returns, factor_returns, benchmark_returns
                )
            except Exception as e:
                print(f"Attribution analysis failed: {e}")
        
        # Perform benchmark comparison if benchmark provided
        benchmark_comparison = None
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            try:
                benchmark_comparison = self.benchmark_comparator.compare_performance(
                    returns, benchmark_returns
                )
            except Exception as e:
                print(f"Benchmark comparison failed: {e}")
        
        # Perform additional statistical tests
        statistical_tests = self._perform_comprehensive_statistical_tests(returns, benchmark_returns)
        
        return ComprehensivePerformanceReport(
            performance_metrics=performance_metrics,
            attribution_result=attribution_result,
            benchmark_comparison=benchmark_comparison,
            statistical_tests=statistical_tests,
            timestamp=datetime.now()
        )
    
    def _calculate_statistical_significance(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate statistical significance of returns and Sharpe ratio."""
        if len(returns) <= 1:
            return {
                'return_tstat': 0.0,
                'return_pvalue': 1.0,
                'sharpe_tstat': 0.0,
                'sharpe_pvalue': 1.0,
                'normality_pvalue': 1.0
            }
        
        # Test if mean return is significantly different from zero
        try:
            return_tstat, return_pvalue = ttest_1samp(returns, 0)
        except:
            return_tstat, return_pvalue = 0.0, 1.0
        
        # Test if Sharpe ratio is significantly different from zero
        excess_returns = returns - self.risk_free_rate / 252
        try:
            sharpe_tstat, sharpe_pvalue = ttest_1samp(excess_returns, 0)
        except:
            sharpe_tstat, sharpe_pvalue = 0.0, 1.0
        
        # Test for normality
        try:
            _, normality_pvalue = normaltest(returns)
        except:
            normality_pvalue = 1.0
        
        return {
            'return_tstat': float(return_tstat),
            'return_pvalue': float(return_pvalue),
            'sharpe_tstat': float(sharpe_tstat),
            'sharpe_pvalue': float(sharpe_pvalue),
            'normality_pvalue': float(normality_pvalue)
        }
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0]
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf') if len(positive_returns) > 0 else 0.0
        
        upside = positive_returns.sum()
        downside = abs(negative_returns.sum())
        
        return upside / downside if downside > 0 else float('inf')
    
    def _calculate_kappa_3(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Kappa 3 (third-order lower partial moment)."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - threshold
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        # Third-order lower partial moment
        lpm3 = (downside_returns ** 3).mean()
        
        if lpm3 == 0:
            return float('inf')
        
        return (returns.mean() - threshold) / (abs(lpm3) ** (1/3))
    
    def _calculate_gain_loss_ratio(self, returns: pd.Series) -> float:
        """Calculate gain-to-loss ratio."""
        if len(returns) <= 1:
            return 0.0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf') if len(positive_returns) > 0 else 0.0
        
        average_gain = positive_returns.mean() if len(positive_returns) > 0 else 0.0
        average_loss = abs(negative_returns.mean())
        
        return average_gain / average_loss if average_loss > 0 else float('inf')
    
    def _calculate_upside_potential_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate upside potential ratio."""
        if len(returns) <= 1:
            return 0.0
        
        excess_returns = returns - threshold
        upside_returns = excess_returns[excess_returns > 0]
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if len(upside_returns) > 0 else 0.0
        
        upside_potential = upside_returns.mean() if len(upside_returns) > 0 else 0.0
        downside_deviation = np.sqrt((downside_returns ** 2).mean())
        
        return upside_potential / downside_deviation if downside_deviation > 0 else float('inf')
    
    def _perform_comprehensive_statistical_tests(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive statistical tests."""
        tests = {}
        
        if len(returns) <= 1:
            return tests
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = jarque_bera(returns)
            tests['jarque_bera'] = {
                'statistic': float(jb_stat),
                'pvalue': float(jb_pvalue),
                'interpretation': 'Normal' if jb_pvalue > 0.05 else 'Non-normal'
            }
        except:
            tests['jarque_bera'] = {'statistic': 0.0, 'pvalue': 1.0, 'interpretation': 'Test failed'}
        
        # Ljung-Box test for autocorrelation
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(returns, lags=10, return_df=True)
            tests['ljung_box'] = {
                'statistic': float(lb_result['lb_stat'].iloc[-1]),
                'pvalue': float(lb_result['lb_pvalue'].iloc[-1]),
                'interpretation': 'No autocorrelation' if lb_result['lb_pvalue'].iloc[-1] > 0.05 else 'Autocorrelation present'
            }
        except:
            tests['ljung_box'] = {'statistic': 0.0, 'pvalue': 1.0, 'interpretation': 'Test failed'}
        
        # ARCH test for heteroscedasticity
        try:
            from statsmodels.stats.diagnostic import het_arch
            arch_stat, arch_pvalue, _, _ = het_arch(returns, nlags=5)
            tests['arch'] = {
                'statistic': float(arch_stat),
                'pvalue': float(arch_pvalue),
                'interpretation': 'Homoscedastic' if arch_pvalue > 0.05 else 'Heteroscedastic'
            }
        except:
            tests['arch'] = {'statistic': 0.0, 'pvalue': 1.0, 'interpretation': 'Test failed'}
        
        # If benchmark provided, test for structural breaks
        if benchmark_returns is not None and len(benchmark_returns) > 0:
            try:
                aligned_returns = pd.concat([
                    returns.rename('portfolio'),
                    benchmark_returns.rename('benchmark')
                ], axis=1).dropna()
                
                if len(aligned_returns) > 30:
                    # Simple Chow test approximation
                    mid_point = len(aligned_returns) // 2
                    first_half = aligned_returns.iloc[:mid_point]
                    second_half = aligned_returns.iloc[mid_point:]
                    
                    # Compare correlations
                    corr1 = first_half['portfolio'].corr(first_half['benchmark'])
                    corr2 = second_half['portfolio'].corr(second_half['benchmark'])
                    
                    tests['structural_break'] = {
                        'first_half_correlation': float(corr1) if not pd.isna(corr1) else 0.0,
                        'second_half_correlation': float(corr2) if not pd.isna(corr2) else 0.0,
                        'correlation_change': float(abs(corr2 - corr1)) if not pd.isna(corr1) and not pd.isna(corr2) else 0.0
                    }
            except:
                pass
        
        return tests
    
    def generate_comprehensive_report(self, analysis: ComprehensivePerformanceReport) -> str:
        """Generate a comprehensive performance analysis report."""
        
        report = []
        report.append("=" * 60)
        report.append("COMPREHENSIVE PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Date: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic performance metrics
        report.append(self.generate_performance_report(analysis.performance_metrics))
        report.append("")
        
        # Statistical significance
        report.append("=== STATISTICAL SIGNIFICANCE ===")
        metrics = analysis.performance_metrics
        report.append(f"Return t-statistic:   {metrics.return_tstat:>10.2f}")
        report.append(f"Return p-value:       {metrics.return_pvalue:>10.3f}")
        report.append(f"Sharpe t-statistic:   {metrics.sharpe_tstat:>10.2f}")
        report.append(f"Sharpe p-value:       {metrics.sharpe_pvalue:>10.3f}")
        report.append(f"Normality p-value:    {metrics.normality_pvalue:>10.3f}")
        report.append("")
        
        # Additional advanced metrics
        report.append("=== ADDITIONAL ADVANCED METRICS ===")
        report.append(f"Omega Ratio:          {metrics.omega_ratio:>10.2f}")
        report.append(f"Kappa 3:              {metrics.kappa_3:>10.2f}")
        report.append(f"Gain/Loss Ratio:      {metrics.gain_loss_ratio:>10.2f}")
        report.append(f"Upside Potential:     {metrics.upside_potential_ratio:>10.2f}")
        report.append("")
        
        # Attribution analysis
        if analysis.attribution_result:
            report.append(self.attribution_analyzer.generate_attribution_report(analysis.attribution_result))
            report.append("")
        
        # Benchmark comparison
        if analysis.benchmark_comparison:
            report.append(self.benchmark_comparator.generate_comparison_report(analysis.benchmark_comparison))
            report.append("")
        
        # Statistical tests
        if analysis.statistical_tests:
            report.append("=== STATISTICAL TESTS ===")
            
            for test_name, test_result in analysis.statistical_tests.items():
                if isinstance(test_result, dict) and 'interpretation' in test_result:
                    report.append(f"{test_name.replace('_', ' ').title()}:")
                    report.append(f"  Statistic: {test_result.get('statistic', 0.0):>8.3f}")
                    report.append(f"  P-value:   {test_result.get('pvalue', 1.0):>8.3f}")
                    report.append(f"  Result:    {test_result['interpretation']}")
                    report.append("")
        
        return "\n".join(report)