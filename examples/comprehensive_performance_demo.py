"""
Demonstration of comprehensive performance analysis capabilities.

Shows performance metrics, attribution analysis, benchmark comparison,
and statistical significance testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.backtesting.performance_analyzer import PerformanceAnalyzer
from src.infrastructure.backtesting.performance_attribution import PerformanceAttributionAnalyzer
from src.infrastructure.backtesting.benchmark_comparison import BenchmarkComparator


def generate_sample_data():
    """Generate sample portfolio, benchmark, and factor return data."""
    np.random.seed(42)
    
    # Generate 3 years of daily data
    dates = pd.date_range('2021-01-01', '2023-12-31', freq='D')
    n_periods = len(dates)
    
    # Create factor returns (market, value, momentum, size, quality)
    factor_returns = pd.DataFrame({
        'market': np.random.normal(0.0005, 0.015, n_periods),
        'value': np.random.normal(0.0002, 0.01, n_periods),
        'momentum': np.random.normal(0.0001, 0.012, n_periods),
        'size': np.random.normal(-0.0001, 0.008, n_periods),
        'quality': np.random.normal(0.0003, 0.009, n_periods)
    }, index=dates)
    
    # Create benchmark returns (market-like)
    benchmark_returns = pd.Series(
        factor_returns['market'] + np.random.normal(0, 0.003, n_periods),
        index=dates,
        name='benchmark'
    )
    
    # Create portfolio returns with factor exposures
    portfolio_exposures = {
        'market': 1.2,    # Higher market exposure
        'value': 0.3,     # Value tilt
        'momentum': 0.2,  # Momentum tilt
        'size': -0.1,     # Large cap bias
        'quality': 0.4    # Quality tilt
    }
    
    # Generate portfolio returns based on factor model
    portfolio_returns = pd.Series(0.0, index=dates)
    for factor, exposure in portfolio_exposures.items():
        portfolio_returns += exposure * factor_returns[factor]
    
    # Add alpha and idiosyncratic risk
    alpha = 0.0003  # 30 bps daily alpha
    idiosyncratic_risk = np.random.normal(0, 0.008, n_periods)
    portfolio_returns += alpha + idiosyncratic_risk
    
    portfolio_returns.name = 'portfolio'
    
    # Generate sample trades
    trades = []
    trade_dates = pd.date_range('2021-01-01', '2023-12-31', freq='W')
    
    for i, date in enumerate(trade_dates):
        # Simulate some winning and losing trades
        pnl = np.random.normal(0.002, 0.05)  # 20 bps average with 5% volatility
        trades.append({
            'date': date,
            'symbol': f'STOCK_{i % 50}',
            'pnl': pnl,
            'quantity': np.random.randint(10, 1000),
            'entry_price': 100 + np.random.normal(0, 20),
            'exit_price': 100 + np.random.normal(0, 20) + pnl * 100
        })
    
    return portfolio_returns, benchmark_returns, factor_returns, trades


def demonstrate_basic_performance_analysis():
    """Demonstrate basic performance metrics calculation."""
    print("=" * 60)
    print("BASIC PERFORMANCE ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    portfolio_returns, benchmark_returns, factor_returns, trades = generate_sample_data()
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02
    )
    
    # Calculate performance metrics
    metrics = analyzer.calculate_performance_metrics(
        portfolio_returns,
        trades,
        benchmark_returns
    )
    
    # Generate and print report
    report = analyzer.generate_performance_report(metrics)
    print(report)
    
    return metrics


def demonstrate_attribution_analysis():
    """Demonstrate performance attribution analysis."""
    print("\n" + "=" * 60)
    print("PERFORMANCE ATTRIBUTION ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    portfolio_returns, benchmark_returns, factor_returns, trades = generate_sample_data()
    
    # Initialize attribution analyzer
    attribution_analyzer = PerformanceAttributionAnalyzer()
    
    # Perform factor attribution
    attribution_result = attribution_analyzer.factor_attribution(
        portfolio_returns,
        factor_returns,
        benchmark_returns
    )
    
    # Generate and print attribution report
    attribution_report = attribution_analyzer.generate_attribution_report(attribution_result)
    print(attribution_report)
    
    # Demonstrate Brinson attribution
    print("\n" + "=" * 40)
    print("BRINSON ATTRIBUTION EXAMPLE")
    print("=" * 40)
    
    # Sample sector data
    portfolio_weights = {
        'Technology': 0.35,
        'Healthcare': 0.25,
        'Financials': 0.20,
        'Consumer': 0.15,
        'Energy': 0.05
    }
    
    benchmark_weights = {
        'Technology': 0.30,
        'Healthcare': 0.20,
        'Financials': 0.25,
        'Consumer': 0.20,
        'Energy': 0.05
    }
    
    portfolio_sector_returns = {
        'Technology': 0.12,
        'Healthcare': 0.08,
        'Financials': 0.06,
        'Consumer': 0.07,
        'Energy': -0.02
    }
    
    benchmark_sector_returns = {
        'Technology': 0.10,
        'Healthcare': 0.09,
        'Financials': 0.05,
        'Consumer': 0.06,
        'Energy': -0.01
    }
    
    brinson_result = attribution_analyzer.brinson_attribution(
        portfolio_weights,
        benchmark_weights,
        portfolio_sector_returns,
        benchmark_sector_returns
    )
    
    print(f"Total Allocation Effect: {brinson_result.total_allocation:.4f}")
    print(f"Total Selection Effect:  {brinson_result.total_selection:.4f}")
    print(f"Total Interaction Effect: {brinson_result.total_interaction:.4f}")
    print(f"Total Active Return:     {brinson_result.total_active_return:.4f}")
    
    return attribution_result


def demonstrate_benchmark_comparison():
    """Demonstrate benchmark comparison analysis."""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    portfolio_returns, benchmark_returns, factor_returns, trades = generate_sample_data()
    
    # Initialize benchmark comparator
    comparator = BenchmarkComparator(risk_free_rate=0.02)
    
    # Perform benchmark comparison
    comparison_result = comparator.compare_performance(
        portfolio_returns,
        benchmark_returns
    )
    
    # Generate and print comparison report
    comparison_report = comparator.generate_comparison_report(comparison_result)
    print(comparison_report)
    
    # Demonstrate multi-period analysis
    print("\n" + "=" * 40)
    print("MULTI-PERIOD ANALYSIS")
    print("=" * 40)
    
    multi_period_results = comparator.multi_period_analysis(
        portfolio_returns,
        benchmark_returns
    )
    
    print(f"{'Period':<8} {'Portfolio':<10} {'Benchmark':<10} {'Excess':<10} {'Hit Ratio':<10}")
    print("-" * 50)
    
    for result in multi_period_results:
        print(f"{result.period:<8} {result.portfolio_return:>9.2%} "
              f"{result.benchmark_return:>9.2%} {result.excess_return:>9.2%} "
              f"{result.hit_ratio:>9.2%}")
    
    return comparison_result


def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive performance analysis."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE PERFORMANCE ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    portfolio_returns, benchmark_returns, factor_returns, trades = generate_sample_data()
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer(
        benchmark_returns=benchmark_returns,
        risk_free_rate=0.02
    )
    
    # Perform comprehensive analysis
    comprehensive_analysis = analyzer.comprehensive_analysis(
        portfolio_returns,
        benchmark_returns,
        factor_returns,
        trades
    )
    
    # Generate and print comprehensive report
    comprehensive_report = analyzer.generate_comprehensive_report(comprehensive_analysis)
    print(comprehensive_report)
    
    return comprehensive_analysis


def demonstrate_rolling_analysis():
    """Demonstrate rolling performance analysis."""
    print("\n" + "=" * 60)
    print("ROLLING ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Generate sample data
    portfolio_returns, benchmark_returns, factor_returns, trades = generate_sample_data()
    
    # Initialize analyzers
    attribution_analyzer = PerformanceAttributionAnalyzer()
    comparator = BenchmarkComparator(risk_free_rate=0.02)
    
    # Perform rolling attribution analysis
    rolling_attribution = attribution_analyzer.rolling_attribution(
        portfolio_returns,
        factor_returns,
        window=252,  # 1-year rolling window
        benchmark_returns=benchmark_returns
    )
    
    # Perform rolling benchmark comparison
    rolling_comparison = comparator.rolling_comparison(
        portfolio_returns,
        benchmark_returns,
        window=252
    )
    
    print("Rolling Attribution Analysis (last 10 periods):")
    print(rolling_attribution.tail(10)[['alpha', 'r_squared', 'tracking_error', 'information_ratio']])
    
    print("\nRolling Benchmark Comparison (last 10 periods):")
    print(rolling_comparison.tail(10)[['excess_return', 'beta', 'alpha', 'information_ratio']])
    
    return rolling_attribution, rolling_comparison


def create_performance_visualizations():
    """Create visualizations for performance analysis."""
    print("\n" + "=" * 60)
    print("CREATING PERFORMANCE VISUALIZATIONS")
    print("=" * 60)
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping visualizations.")
        return None
    
    # Generate sample data
    portfolio_returns, benchmark_returns, factor_returns, trades = generate_sample_data()
    
    # Calculate cumulative returns
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    benchmark_cumulative = (1 + benchmark_returns).cumprod()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Performance Analysis', fontsize=16)
    
    # Plot 1: Cumulative returns
    axes[0, 0].plot(portfolio_cumulative.index, portfolio_cumulative.values, 
                    label='Portfolio', linewidth=2)
    axes[0, 0].plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                    label='Benchmark', linewidth=2)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Rolling Sharpe ratio
    rolling_sharpe_portfolio = portfolio_returns.rolling(252).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    rolling_sharpe_benchmark = benchmark_returns.rolling(252).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    
    axes[0, 1].plot(rolling_sharpe_portfolio.index, rolling_sharpe_portfolio.values, 
                    label='Portfolio', linewidth=2)
    axes[0, 1].plot(rolling_sharpe_benchmark.index, rolling_sharpe_benchmark.values, 
                    label='Benchmark', linewidth=2)
    axes[0, 1].set_title('Rolling Sharpe Ratio (1-Year)')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Return distribution
    axes[1, 0].hist(portfolio_returns.values, bins=50, alpha=0.7, 
                    label='Portfolio', density=True)
    axes[1, 0].hist(benchmark_returns.values, bins=50, alpha=0.7, 
                    label='Benchmark', density=True)
    axes[1, 0].set_title('Return Distribution')
    axes[1, 0].set_xlabel('Daily Return')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Drawdown
    portfolio_dd = (portfolio_cumulative / portfolio_cumulative.expanding().max() - 1)
    benchmark_dd = (benchmark_cumulative / benchmark_cumulative.expanding().max() - 1)
    
    axes[1, 1].fill_between(portfolio_dd.index, portfolio_dd.values, 0, 
                           alpha=0.7, label='Portfolio')
    axes[1, 1].fill_between(benchmark_dd.index, benchmark_dd.values, 0, 
                           alpha=0.7, label='Benchmark')
    axes[1, 1].set_title('Drawdown')
    axes[1, 1].set_ylabel('Drawdown')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_performance_analysis.png', dpi=300, bbox_inches='tight')
    print("Performance visualization saved as 'comprehensive_performance_analysis.png'")
    
    return fig


def main():
    """Run comprehensive performance analysis demonstration."""
    print("Starting Comprehensive Performance Analysis Demonstration...")
    print("This demo showcases 20+ performance metrics, attribution analysis,")
    print("benchmark comparison, and statistical significance testing.\n")
    
    try:
        # Run demonstrations
        basic_metrics = demonstrate_basic_performance_analysis()
        attribution_result = demonstrate_attribution_analysis()
        comparison_result = demonstrate_benchmark_comparison()
        comprehensive_result = demonstrate_comprehensive_analysis()
        rolling_attribution, rolling_comparison = demonstrate_rolling_analysis()
        
        # Create visualizations
        fig = create_performance_visualizations() if MATPLOTLIB_AVAILABLE else None
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Key features demonstrated:")
        print("✓ 20+ comprehensive performance metrics")
        print("✓ Factor-based performance attribution")
        print("✓ Brinson attribution analysis")
        print("✓ Comprehensive benchmark comparison")
        print("✓ Statistical significance testing")
        print("✓ Rolling analysis capabilities")
        print("✓ Advanced risk metrics (VaR, CVaR, etc.)")
        print("✓ Distribution analysis and normality tests")
        print("✓ Up/down market capture analysis")
        print("✓ Professional reporting capabilities")
        
        return {
            'basic_metrics': basic_metrics,
            'attribution_result': attribution_result,
            'comparison_result': comparison_result,
            'comprehensive_result': comprehensive_result,
            'rolling_attribution': rolling_attribution,
            'rolling_comparison': rolling_comparison
        }
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()