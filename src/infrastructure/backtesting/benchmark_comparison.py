"""
Benchmark Comparison Analysis for quantitative strategies.

Provides comprehensive comparison against benchmarks including relative performance,
risk-adjusted metrics, and statistical significance testing.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_1samp, jarque_bera, normaltest

from ...domain.exceptions import ValidationError


@dataclass
class BenchmarkComparisonResult:
    """Results from benchmark comparison analysis."""
    
    # Basic comparison metrics
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    
    # Risk metrics comparison
    portfolio_volatility: float
    benchmark_volatility: float
    tracking_error: float
    
    # Risk-adjusted metrics
    portfolio_sharpe: float
    benchmark_sharpe: float
    information_ratio: float
    treynor_ratio: float
    
    # Relative performance metrics
    beta: float
    alpha: float
    correlation: float
    r_squared: float
    
    # Drawdown comparison
    portfolio_max_drawdown: float
    benchmark_max_drawdown: float
    relative_max_drawdown: float
    
    # Statistical significance
    excess_return_tstat: float
    excess_return_pvalue: float
    alpha_tstat: float
    alpha_pvalue: float
    
    # Distribution analysis
    excess_return_skewness: float
    excess_return_kurtosis: float
    normality_test_pvalue: float
    
    # Performance periods
    outperformance_periods: int
    total_periods: int
    hit_ratio: float
    
    # Up/down market analysis
    up_market_beta: float
    down_market_beta: float
    up_capture_ratio: float
    down_capture_ratio: float
    
    timestamp: datetime


@dataclass
class RelativePerformanceMetrics:
    """Relative performance metrics over different time periods."""
    
    period: str
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    tracking_error: float
    information_ratio: float
    hit_ratio: float


class BenchmarkComparator:
    """Compares portfolio performance against benchmarks."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def compare_performance(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        confidence_level: float = 0.05
    ) -> BenchmarkComparisonResult:
        """
        Perform comprehensive benchmark comparison analysis.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            confidence_level: Confidence level for statistical tests
            
        Returns:
            BenchmarkComparisonResult with comprehensive comparison metrics
        """
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            raise ValidationError("Portfolio and benchmark returns cannot be empty")
        
        # Align returns
        aligned_returns = pd.concat([
            portfolio_returns.rename('portfolio'),
            benchmark_returns.rename('benchmark')
        ], axis=1).dropna()
        
        if len(aligned_returns) < 10:
            raise ValidationError("Insufficient overlapping data for comparison")
        
        portfolio_ret = aligned_returns['portfolio']
        benchmark_ret = aligned_returns['benchmark']
        excess_returns = portfolio_ret - benchmark_ret
        
        # Basic return metrics
        portfolio_return = self._calculate_annualized_return(portfolio_ret)
        benchmark_return = self._calculate_annualized_return(benchmark_ret)
        excess_return = portfolio_return - benchmark_return
        
        # Risk metrics
        portfolio_volatility = self._calculate_volatility(portfolio_ret)
        benchmark_volatility = self._calculate_volatility(benchmark_ret)
        tracking_error = self._calculate_volatility(excess_returns)
        
        # Risk-adjusted metrics
        portfolio_sharpe = self._calculate_sharpe_ratio(portfolio_ret)
        benchmark_sharpe = self._calculate_sharpe_ratio(benchmark_ret)
        information_ratio = self._calculate_information_ratio(excess_returns)
        
        # Regression analysis
        beta, alpha, correlation, r_squared = self._calculate_regression_metrics(
            portfolio_ret, benchmark_ret
        )
        
        treynor_ratio = (portfolio_return - self.risk_free_rate) / beta if beta != 0 else 0.0
        
        # Drawdown analysis
        portfolio_max_drawdown = self._calculate_max_drawdown(portfolio_ret)
        benchmark_max_drawdown = self._calculate_max_drawdown(benchmark_ret)
        
        # Calculate relative drawdown
        portfolio_cumulative = (1 + portfolio_ret).cumprod()
        benchmark_cumulative = (1 + benchmark_ret).cumprod()
        relative_performance = portfolio_cumulative / benchmark_cumulative
        relative_max_drawdown = self._calculate_max_drawdown(relative_performance.pct_change().dropna())
        
        # Statistical significance testing
        excess_return_tstat, excess_return_pvalue = self._test_excess_return_significance(excess_returns)
        alpha_tstat, alpha_pvalue = self._test_alpha_significance(portfolio_ret, benchmark_ret)
        
        # Distribution analysis
        excess_return_skewness = float(stats.skew(excess_returns))
        excess_return_kurtosis = float(stats.kurtosis(excess_returns))
        
        # Normality test
        try:
            _, normality_pvalue = normaltest(excess_returns)
        except:
            normality_pvalue = 0.0
        
        # Performance periods analysis
        outperformance_periods = (excess_returns > 0).sum()
        total_periods = len(excess_returns)
        hit_ratio = outperformance_periods / total_periods
        
        # Up/down market analysis
        up_down_metrics = self._calculate_up_down_capture(portfolio_ret, benchmark_ret)
        
        return BenchmarkComparisonResult(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            excess_return=excess_return,
            portfolio_volatility=portfolio_volatility,
            benchmark_volatility=benchmark_volatility,
            tracking_error=tracking_error,
            portfolio_sharpe=portfolio_sharpe,
            benchmark_sharpe=benchmark_sharpe,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            beta=beta,
            alpha=alpha,
            correlation=correlation,
            r_squared=r_squared,
            portfolio_max_drawdown=portfolio_max_drawdown,
            benchmark_max_drawdown=benchmark_max_drawdown,
            relative_max_drawdown=relative_max_drawdown,
            excess_return_tstat=excess_return_tstat,
            excess_return_pvalue=excess_return_pvalue,
            alpha_tstat=alpha_tstat,
            alpha_pvalue=alpha_pvalue,
            excess_return_skewness=excess_return_skewness,
            excess_return_kurtosis=excess_return_kurtosis,
            normality_test_pvalue=normality_pvalue,
            outperformance_periods=outperformance_periods,
            total_periods=total_periods,
            hit_ratio=hit_ratio,
            up_market_beta=up_down_metrics['up_beta'],
            down_market_beta=up_down_metrics['down_beta'],
            up_capture_ratio=up_down_metrics['up_capture'],
            down_capture_ratio=up_down_metrics['down_capture'],
            timestamp=datetime.now()
        )
    
    def rolling_comparison(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Perform rolling benchmark comparison analysis.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            window: Rolling window size
            
        Returns:
            DataFrame with rolling comparison metrics
        """
        # Align returns
        aligned_returns = pd.concat([
            portfolio_returns.rename('portfolio'),
            benchmark_returns.rename('benchmark')
        ], axis=1).dropna()
        
        if len(aligned_returns) < window * 2:
            raise ValidationError(f"Insufficient data for rolling comparison (need at least {window * 2} periods)")
        
        results = []
        
        for i in range(window, len(aligned_returns)):
            window_data = aligned_returns.iloc[i-window:i]
            portfolio_ret = window_data['portfolio']
            benchmark_ret = window_data['benchmark']
            excess_returns = portfolio_ret - benchmark_ret
            
            # Calculate metrics for this window
            result = {
                'date': aligned_returns.index[i],
                'portfolio_return': self._calculate_annualized_return(portfolio_ret),
                'benchmark_return': self._calculate_annualized_return(benchmark_ret),
                'excess_return': self._calculate_annualized_return(excess_returns),
                'tracking_error': self._calculate_volatility(excess_returns),
                'information_ratio': self._calculate_information_ratio(excess_returns),
                'hit_ratio': (excess_returns > 0).mean()
            }
            
            # Add regression metrics
            beta, alpha, correlation, r_squared = self._calculate_regression_metrics(
                portfolio_ret, benchmark_ret
            )
            result.update({
                'beta': beta,
                'alpha': alpha,
                'correlation': correlation,
                'r_squared': r_squared
            })
            
            results.append(result)
        
        return pd.DataFrame(results).set_index('date')
    
    def multi_period_analysis(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        periods: List[str] = ['1M', '3M', '6M', '1Y', '2Y', '3Y']
    ) -> List[RelativePerformanceMetrics]:
        """
        Analyze relative performance over multiple time periods.
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            periods: List of periods to analyze
            
        Returns:
            List of RelativePerformanceMetrics for each period
        """
        # Align returns
        aligned_returns = pd.concat([
            portfolio_returns.rename('portfolio'),
            benchmark_returns.rename('benchmark')
        ], axis=1).dropna()
        
        results = []
        
        for period in periods:
            try:
                # Get data for the specified period
                if period == '1M':
                    period_data = aligned_returns.tail(21)  # ~1 month
                elif period == '3M':
                    period_data = aligned_returns.tail(63)  # ~3 months
                elif period == '6M':
                    period_data = aligned_returns.tail(126)  # ~6 months
                elif period == '1Y':
                    period_data = aligned_returns.tail(252)  # ~1 year
                elif period == '2Y':
                    period_data = aligned_returns.tail(504)  # ~2 years
                elif period == '3Y':
                    period_data = aligned_returns.tail(756)  # ~3 years
                else:
                    continue
                
                if len(period_data) < 10:
                    continue
                
                portfolio_ret = period_data['portfolio']
                benchmark_ret = period_data['benchmark']
                excess_returns = portfolio_ret - benchmark_ret
                
                # Calculate metrics
                portfolio_return = self._calculate_annualized_return(portfolio_ret)
                benchmark_return = self._calculate_annualized_return(benchmark_ret)
                excess_return = portfolio_return - benchmark_return
                tracking_error = self._calculate_volatility(excess_returns)
                information_ratio = self._calculate_information_ratio(excess_returns)
                hit_ratio = (excess_returns > 0).mean()
                
                results.append(RelativePerformanceMetrics(
                    period=period,
                    portfolio_return=portfolio_return,
                    benchmark_return=benchmark_return,
                    excess_return=excess_return,
                    tracking_error=tracking_error,
                    information_ratio=information_ratio,
                    hit_ratio=hit_ratio
                ))
                
            except Exception:
                continue
        
        return results
    
    def _calculate_annualized_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
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
    
    def _calculate_information_ratio(self, excess_returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Information ratio."""
        if len(excess_returns) <= 1:
            return 0.0
        
        tracking_error = excess_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return float(excess_returns.mean() / tracking_error * np.sqrt(periods_per_year))
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return float(drawdown.min())
    
    def _calculate_regression_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float, float, float]:
        """Calculate regression-based metrics (beta, alpha, correlation, R²)."""
        if len(portfolio_returns) <= 1 or len(benchmark_returns) <= 1:
            return 0.0, 0.0, 0.0, 0.0
        
        # Calculate beta and alpha using linear regression
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        
        if benchmark_variance == 0:
            beta = 0.0
            alpha = portfolio_returns.mean()
        else:
            beta = covariance / benchmark_variance
            alpha = portfolio_returns.mean() - beta * benchmark_returns.mean()
        
        # Calculate correlation
        correlation = portfolio_returns.corr(benchmark_returns)
        if pd.isna(correlation):
            correlation = 0.0
        
        # Calculate R-squared
        r_squared = correlation ** 2
        
        # Annualize alpha
        alpha_annualized = alpha * 252
        
        return float(beta), float(alpha_annualized), float(correlation), float(r_squared)
    
    def _test_excess_return_significance(self, excess_returns: pd.Series) -> Tuple[float, float]:
        """Test statistical significance of excess returns."""
        if len(excess_returns) <= 1:
            return 0.0, 1.0
        
        try:
            t_stat, p_value = ttest_1samp(excess_returns, 0)
            return float(t_stat), float(p_value)
        except:
            return 0.0, 1.0
    
    def _test_alpha_significance(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Test statistical significance of alpha."""
        if len(portfolio_returns) <= 2 or len(benchmark_returns) <= 2:
            return 0.0, 1.0
        
        try:
            # Perform regression
            from sklearn.linear_model import LinearRegression
            import numpy as np
            
            X = benchmark_returns.values.reshape(-1, 1)
            y = portfolio_returns.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate residuals and standard error
            predictions = model.predict(X)
            residuals = y - predictions
            mse = np.mean(residuals ** 2)
            
            # Standard error of alpha (intercept)
            n = len(portfolio_returns)
            x_mean = benchmark_returns.mean()
            x_var = benchmark_returns.var()
            
            se_alpha = np.sqrt(mse * (1/n + x_mean**2 / (n * x_var)))
            
            # T-statistic for alpha
            alpha = model.intercept_
            t_stat = alpha / se_alpha if se_alpha > 0 else 0.0
            
            # P-value (two-tailed test)
            from scipy.stats import t
            p_value = 2 * (1 - t.cdf(abs(t_stat), n - 2))
            
            return float(t_stat), float(p_value)
            
        except:
            return 0.0, 1.0
    
    def _calculate_up_down_capture(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate up and down market capture ratios and betas."""
        
        # Separate up and down market periods
        up_market = benchmark_returns > 0
        down_market = benchmark_returns < 0
        
        up_portfolio = portfolio_returns[up_market]
        up_benchmark = benchmark_returns[up_market]
        down_portfolio = portfolio_returns[down_market]
        down_benchmark = benchmark_returns[down_market]
        
        # Calculate up market beta
        if len(up_portfolio) > 1 and up_benchmark.var() > 0:
            up_beta = np.cov(up_portfolio, up_benchmark)[0, 1] / up_benchmark.var()
        else:
            up_beta = 0.0
        
        # Calculate down market beta
        if len(down_portfolio) > 1 and down_benchmark.var() > 0:
            down_beta = np.cov(down_portfolio, down_benchmark)[0, 1] / down_benchmark.var()
        else:
            down_beta = 0.0
        
        # Calculate capture ratios
        if len(up_portfolio) > 0 and up_benchmark.mean() != 0:
            up_capture = up_portfolio.mean() / up_benchmark.mean()
        else:
            up_capture = 0.0
        
        if len(down_portfolio) > 0 and down_benchmark.mean() != 0:
            down_capture = down_portfolio.mean() / down_benchmark.mean()
        else:
            down_capture = 0.0
        
        return {
            'up_beta': float(up_beta),
            'down_beta': float(down_beta),
            'up_capture': float(up_capture),
            'down_capture': float(down_capture)
        }
    
    def generate_comparison_report(self, comparison: BenchmarkComparisonResult) -> str:
        """Generate a comprehensive benchmark comparison report."""
        
        report = []
        report.append("=== BENCHMARK COMPARISON REPORT ===")
        report.append(f"Analysis Date: {comparison.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Performance comparison
        report.append("=== PERFORMANCE COMPARISON ===")
        report.append(f"Portfolio Return:    {comparison.portfolio_return:>10.2%}")
        report.append(f"Benchmark Return:    {comparison.benchmark_return:>10.2%}")
        report.append(f"Excess Return:       {comparison.excess_return:>10.2%}")
        report.append("")
        
        # Risk comparison
        report.append("=== RISK COMPARISON ===")
        report.append(f"Portfolio Volatility: {comparison.portfolio_volatility:>9.2%}")
        report.append(f"Benchmark Volatility: {comparison.benchmark_volatility:>9.2%}")
        report.append(f"Tracking Error:       {comparison.tracking_error:>9.2%}")
        report.append("")
        
        # Risk-adjusted metrics
        report.append("=== RISK-ADJUSTED METRICS ===")
        report.append(f"Portfolio Sharpe:    {comparison.portfolio_sharpe:>10.2f}")
        report.append(f"Benchmark Sharpe:    {comparison.benchmark_sharpe:>10.2f}")
        report.append(f"Information Ratio:   {comparison.information_ratio:>10.2f}")
        report.append(f"Treynor Ratio:       {comparison.treynor_ratio:>10.2f}")
        report.append("")
        
        # Regression analysis
        report.append("=== REGRESSION ANALYSIS ===")
        report.append(f"Beta:                {comparison.beta:>10.2f}")
        report.append(f"Alpha:               {comparison.alpha:>10.2%}")
        report.append(f"Correlation:         {comparison.correlation:>10.2f}")
        report.append(f"R-Squared:           {comparison.r_squared:>10.2%}")
        report.append("")
        
        # Statistical significance
        report.append("=== STATISTICAL SIGNIFICANCE ===")
        report.append(f"Excess Return t-stat: {comparison.excess_return_tstat:>9.2f}")
        report.append(f"Excess Return p-val:  {comparison.excess_return_pvalue:>9.3f}")
        report.append(f"Alpha t-stat:         {comparison.alpha_tstat:>9.2f}")
        report.append(f"Alpha p-value:        {comparison.alpha_pvalue:>9.3f}")
        report.append("")
        
        # Performance periods
        report.append("=== PERFORMANCE PERIODS ===")
        report.append(f"Outperformance Periods: {comparison.outperformance_periods:>7d}")
        report.append(f"Total Periods:          {comparison.total_periods:>7d}")
        report.append(f"Hit Ratio:              {comparison.hit_ratio:>7.2%}")
        report.append("")
        
        # Up/Down market analysis
        report.append("=== UP/DOWN MARKET ANALYSIS ===")
        report.append(f"Up Market Beta:      {comparison.up_market_beta:>10.2f}")
        report.append(f"Down Market Beta:    {comparison.down_market_beta:>10.2f}")
        report.append(f"Up Capture Ratio:    {comparison.up_capture_ratio:>10.2f}")
        report.append(f"Down Capture Ratio:  {comparison.down_capture_ratio:>10.2f}")
        
        return "\n".join(report)