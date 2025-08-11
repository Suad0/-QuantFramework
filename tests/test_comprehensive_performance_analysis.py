"""
Tests for comprehensive performance analysis functionality.

Tests performance metrics, attribution analysis, benchmark comparison,
and statistical significance testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.infrastructure.backtesting.performance_analyzer import (
    PerformanceAnalyzer, DetailedPerformanceMetrics, ComprehensivePerformanceReport
)
from src.infrastructure.backtesting.performance_attribution import (
    PerformanceAttributionAnalyzer, AttributionResult
)
from src.infrastructure.backtesting.benchmark_comparison import (
    BenchmarkComparator, BenchmarkComparisonResult
)
from src.domain.exceptions import ValidationError


class TestPerformanceAnalyzer:
    """Test comprehensive performance analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample return data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        # Portfolio returns with slight positive drift
        self.portfolio_returns = pd.Series(
            np.random.normal(0.0008, 0.02, 252),
            index=dates,
            name='portfolio'
        )
        
        # Benchmark returns with lower drift
        self.benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=dates,
            name='benchmark'
        )
        
        # Factor returns
        self.factor_returns = pd.DataFrame({
            'market': np.random.normal(0.0005, 0.015, 252),
            'value': np.random.normal(0.0002, 0.01, 252),
            'momentum': np.random.normal(0.0001, 0.012, 252),
            'size': np.random.normal(0.0003, 0.008, 252)
        }, index=dates)
        
        # Sample trades
        self.trades = [
            {'symbol': 'AAPL', 'pnl': 0.05, 'quantity': 100},
            {'symbol': 'GOOGL', 'pnl': -0.02, 'quantity': 50},
            {'symbol': 'MSFT', 'pnl': 0.03, 'quantity': 75},
            {'symbol': 'TSLA', 'pnl': -0.01, 'quantity': 25},
            {'symbol': 'AMZN', 'pnl': 0.04, 'quantity': 60}
        ]
        
        self.analyzer = PerformanceAnalyzer(
            benchmark_returns=self.benchmark_returns,
            risk_free_rate=0.02
        )
    
    def test_calculate_performance_metrics(self):
        """Test basic performance metrics calculation."""
        metrics = self.analyzer.calculate_performance_metrics(
            self.portfolio_returns,
            self.trades,
            self.benchmark_returns
        )
        
        assert isinstance(metrics, DetailedPerformanceMetrics)
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.annualized_return, float)
        assert isinstance(metrics.volatility, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.sortino_ratio, float)
        assert isinstance(metrics.calmar_ratio, float)
        assert isinstance(metrics.information_ratio, float)
        assert isinstance(metrics.treynor_ratio, float)
        
        # Check that all metrics are reasonable
        assert -1.0 <= metrics.total_return <= 5.0
        assert -1.0 <= metrics.annualized_return <= 2.0
        assert 0.0 <= metrics.volatility <= 1.0
        assert -10.0 <= metrics.sharpe_ratio <= 10.0
        assert 0.0 <= metrics.win_rate <= 1.0
        assert metrics.profit_factor >= 0.0
    
    def test_statistical_significance_metrics(self):
        """Test statistical significance calculations."""
        metrics = self.analyzer.calculate_performance_metrics(self.portfolio_returns)
        
        assert isinstance(metrics.return_tstat, float)
        assert isinstance(metrics.return_pvalue, float)
        assert isinstance(metrics.sharpe_tstat, float)
        assert isinstance(metrics.sharpe_pvalue, float)
        assert isinstance(metrics.normality_pvalue, float)
        
        # P-values should be between 0 and 1
        assert 0.0 <= metrics.return_pvalue <= 1.0
        assert 0.0 <= metrics.sharpe_pvalue <= 1.0
        assert 0.0 <= metrics.normality_pvalue <= 1.0
    
    def test_advanced_metrics(self):
        """Test advanced performance metrics."""
        metrics = self.analyzer.calculate_performance_metrics(self.portfolio_returns)
        
        assert isinstance(metrics.omega_ratio, float)
        assert isinstance(metrics.kappa_3, float)
        assert isinstance(metrics.gain_loss_ratio, float)
        assert isinstance(metrics.upside_potential_ratio, float)
        
        # These metrics should be positive (or inf)
        assert metrics.omega_ratio >= 0.0
        assert metrics.gain_loss_ratio >= 0.0
        assert metrics.upside_potential_ratio >= 0.0
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis with all components."""
        analysis = self.analyzer.comprehensive_analysis(
            self.portfolio_returns,
            self.benchmark_returns,
            self.factor_returns,
            self.trades
        )
        
        assert isinstance(analysis, ComprehensivePerformanceReport)
        assert isinstance(analysis.performance_metrics, DetailedPerformanceMetrics)
        assert analysis.attribution_result is not None
        assert analysis.benchmark_comparison is not None
        assert isinstance(analysis.statistical_tests, dict)
        assert isinstance(analysis.timestamp, datetime)
    
    def test_empty_returns_handling(self):
        """Test handling of empty returns."""
        empty_returns = pd.Series([], dtype=float)
        
        with pytest.raises(ValidationError):
            self.analyzer.calculate_performance_metrics(empty_returns)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        short_returns = pd.Series([0.01, 0.02], index=pd.date_range('2020-01-01', periods=2))
        
        metrics = self.analyzer.calculate_performance_metrics(short_returns)
        
        # Should handle gracefully with default values
        assert isinstance(metrics, DetailedPerformanceMetrics)
        assert metrics.volatility >= 0.0
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        analysis = self.analyzer.comprehensive_analysis(
            self.portfolio_returns,
            self.benchmark_returns,
            self.factor_returns
        )
        
        report = self.analyzer.generate_comprehensive_report(analysis)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "COMPREHENSIVE PERFORMANCE ANALYSIS REPORT" in report
        assert "PERFORMANCE ANALYSIS REPORT" in report
        assert "STATISTICAL SIGNIFICANCE" in report
        assert "ADDITIONAL ADVANCED METRICS" in report


class TestPerformanceAttributionAnalyzer:
    """Test performance attribution analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        self.portfolio_returns = pd.Series(
            np.random.normal(0.0008, 0.02, 252),
            index=dates
        )
        
        self.factor_returns = pd.DataFrame({
            'market': np.random.normal(0.0005, 0.015, 252),
            'value': np.random.normal(0.0002, 0.01, 252),
            'momentum': np.random.normal(0.0001, 0.012, 252)
        }, index=dates)
        
        self.benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=dates
        )
        
        self.analyzer = PerformanceAttributionAnalyzer()
    
    def test_factor_attribution(self):
        """Test factor attribution analysis."""
        result = self.analyzer.factor_attribution(
            self.portfolio_returns,
            self.factor_returns,
            self.benchmark_returns
        )
        
        assert isinstance(result, AttributionResult)
        assert len(result.factor_returns) == len(self.factor_returns.columns)
        assert len(result.factor_exposures) == len(self.factor_returns.columns)
        assert len(result.factor_contributions) == len(self.factor_returns.columns)
        assert isinstance(result.alpha, float)
        assert isinstance(result.r_squared, float)
        assert 0.0 <= result.r_squared <= 1.0
    
    def test_brinson_attribution(self):
        """Test Brinson attribution analysis."""
        portfolio_weights = {'Tech': 0.4, 'Finance': 0.3, 'Healthcare': 0.3}
        benchmark_weights = {'Tech': 0.3, 'Finance': 0.4, 'Healthcare': 0.3}
        portfolio_returns = {'Tech': 0.05, 'Finance': 0.02, 'Healthcare': 0.03}
        benchmark_returns = {'Tech': 0.04, 'Finance': 0.025, 'Healthcare': 0.028}
        
        result = self.analyzer.brinson_attribution(
            portfolio_weights,
            benchmark_weights,
            portfolio_returns,
            benchmark_returns
        )
        
        assert len(result.allocation_effect) == 3
        assert len(result.selection_effect) == 3
        assert len(result.interaction_effect) == 3
        assert isinstance(result.total_allocation, float)
        assert isinstance(result.total_selection, float)
        assert isinstance(result.total_interaction, float)
    
    def test_rolling_attribution(self):
        """Test rolling attribution analysis."""
        result = self.analyzer.rolling_attribution(
            self.portfolio_returns,
            self.factor_returns,
            window=60,
            benchmark_returns=self.benchmark_returns
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'alpha' in result.columns
        assert 'r_squared' in result.columns
        assert 'tracking_error' in result.columns
        assert 'information_ratio' in result.columns
    
    def test_generate_attribution_report(self):
        """Test attribution report generation."""
        result = self.analyzer.factor_attribution(
            self.portfolio_returns,
            self.factor_returns,
            self.benchmark_returns
        )
        
        report = self.analyzer.generate_attribution_report(result)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "PERFORMANCE ATTRIBUTION REPORT" in report
        assert "FACTOR ATTRIBUTION" in report
        assert "ALPHA ANALYSIS" in report


class TestBenchmarkComparator:
    """Test benchmark comparison analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        
        self.portfolio_returns = pd.Series(
            np.random.normal(0.0008, 0.02, 252),
            index=dates
        )
        
        self.benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.015, 252),
            index=dates
        )
        
        self.comparator = BenchmarkComparator(risk_free_rate=0.02)
    
    def test_compare_performance(self):
        """Test comprehensive benchmark comparison."""
        result = self.comparator.compare_performance(
            self.portfolio_returns,
            self.benchmark_returns
        )
        
        assert isinstance(result, BenchmarkComparisonResult)
        assert isinstance(result.portfolio_return, float)
        assert isinstance(result.benchmark_return, float)
        assert isinstance(result.excess_return, float)
        assert isinstance(result.beta, float)
        assert isinstance(result.alpha, float)
        assert isinstance(result.correlation, float)
        assert isinstance(result.r_squared, float)
        assert isinstance(result.hit_ratio, float)
        
        # Check reasonable ranges
        assert -1.0 <= result.correlation <= 1.0
        assert 0.0 <= result.r_squared <= 1.0
        assert 0.0 <= result.hit_ratio <= 1.0
        assert 0.0 <= result.excess_return_pvalue <= 1.0
        assert 0.0 <= result.alpha_pvalue <= 1.0
    
    def test_rolling_comparison(self):
        """Test rolling benchmark comparison."""
        result = self.comparator.rolling_comparison(
            self.portfolio_returns,
            self.benchmark_returns,
            window=60
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'portfolio_return' in result.columns
        assert 'benchmark_return' in result.columns
        assert 'excess_return' in result.columns
        assert 'beta' in result.columns
        assert 'alpha' in result.columns
        assert 'information_ratio' in result.columns
    
    def test_multi_period_analysis(self):
        """Test multi-period performance analysis."""
        results = self.comparator.multi_period_analysis(
            self.portfolio_returns,
            self.benchmark_returns
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        for result in results:
            assert hasattr(result, 'period')
            assert hasattr(result, 'portfolio_return')
            assert hasattr(result, 'benchmark_return')
            assert hasattr(result, 'excess_return')
            assert hasattr(result, 'information_ratio')
            assert hasattr(result, 'hit_ratio')
    
    def test_generate_comparison_report(self):
        """Test benchmark comparison report generation."""
        result = self.comparator.compare_performance(
            self.portfolio_returns,
            self.benchmark_returns
        )
        
        report = self.comparator.generate_comparison_report(result)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "BENCHMARK COMPARISON REPORT" in report
        assert "PERFORMANCE COMPARISON" in report
        assert "RISK COMPARISON" in report
        assert "STATISTICAL SIGNIFICANCE" in report
    
    def test_empty_returns_handling(self):
        """Test handling of empty returns."""
        empty_returns = pd.Series([], dtype=float)
        
        with pytest.raises(ValidationError):
            self.comparator.compare_performance(empty_returns, self.benchmark_returns)
        
        with pytest.raises(ValidationError):
            self.comparator.compare_performance(self.portfolio_returns, empty_returns)
    
    def test_insufficient_overlapping_data(self):
        """Test handling of insufficient overlapping data."""
        # Create non-overlapping series
        dates1 = pd.date_range('2020-01-01', periods=5, freq='D')
        dates2 = pd.date_range('2020-02-01', periods=5, freq='D')
        
        returns1 = pd.Series(np.random.normal(0.001, 0.02, 5), index=dates1)
        returns2 = pd.Series(np.random.normal(0.001, 0.02, 5), index=dates2)
        
        with pytest.raises(ValidationError):
            self.comparator.compare_performance(returns1, returns2)


if __name__ == "__main__":
    pytest.main([__file__])