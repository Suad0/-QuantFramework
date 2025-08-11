"""
Bias Detection for Backtesting.

Detects and reports common biases that can invalidate backtesting results:
- Look-ahead bias
- Survivorship bias
- Data snooping bias
- Selection bias
- Overfitting
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
import numpy as np
from scipy import stats

from ...domain.exceptions import ValidationError, BacktestError


@dataclass
class BiasWarning:
    """Represents a detected bias warning."""
    
    bias_type: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    description: str
    recommendation: str
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BiasAnalysisResult:
    """Results of bias analysis."""
    
    warnings: List[BiasWarning]
    overall_risk_score: float  # 0.0 to 1.0
    is_reliable: bool
    summary: str
    
    @property
    def critical_warnings(self) -> List[BiasWarning]:
        """Get critical bias warnings."""
        return [w for w in self.warnings if w.severity == 'CRITICAL']
    
    @property
    def high_warnings(self) -> List[BiasWarning]:
        """Get high severity warnings."""
        return [w for w in self.warnings if w.severity == 'HIGH']


class BiasDetector:
    """Detects various biases in backtesting results and data."""
    
    def __init__(self):
        self.analysis_history: List[BiasAnalysisResult] = []
        self.symbol_universe_history: List[Set[str]] = []
        self.parameter_test_history: List[Dict[str, Any]] = []
    
    def analyze_backtest_for_bias(
        self,
        backtest_results: Dict[str, Any],
        market_data: pd.DataFrame,
        strategy_config: Dict[str, Any],
        universe_history: Optional[List[Set[str]]] = None
    ) -> BiasAnalysisResult:
        """Perform comprehensive bias analysis on backtest results."""
        
        warnings = []
        
        # Check for look-ahead bias
        warnings.extend(self._detect_look_ahead_bias(backtest_results, market_data, strategy_config))
        
        # Check for survivorship bias
        if universe_history:
            warnings.extend(self._detect_survivorship_bias(backtest_results, universe_history))
        
        # Check for data snooping bias
        warnings.extend(self._detect_data_snooping_bias(backtest_results, strategy_config))
        
        # Check for selection bias
        warnings.extend(self._detect_selection_bias(backtest_results, market_data))
        
        # Check for overfitting
        warnings.extend(self._detect_overfitting(backtest_results, strategy_config))
        
        # Check for statistical significance
        warnings.extend(self._check_statistical_significance(backtest_results))
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(warnings)
        
        # Determine reliability
        is_reliable = risk_score < 0.5 and len([w for w in warnings if w.severity in ['CRITICAL', 'HIGH']]) == 0
        
        # Generate summary
        summary = self._generate_summary(warnings, risk_score, is_reliable)
        
        result = BiasAnalysisResult(
            warnings=warnings,
            overall_risk_score=risk_score,
            is_reliable=is_reliable,
            summary=summary
        )
        
        self.analysis_history.append(result)
        return result
    
    def _detect_look_ahead_bias(
        self,
        backtest_results: Dict[str, Any],
        market_data: pd.DataFrame,
        strategy_config: Dict[str, Any]
    ) -> List[BiasWarning]:
        """Detect look-ahead bias in strategy implementation."""
        
        warnings = []
        
        # Check if strategy uses future data
        trades = backtest_results.get('trades', [])
        
        for trade in trades:
            trade_date = pd.to_datetime(trade.get('timestamp'))
            symbol = trade.get('symbol')
            
            if symbol in market_data.columns:
                # Check if trade decision could have been made with available data
                available_data = market_data[market_data.index <= trade_date][symbol]
                
                if len(available_data) == 0:
                    warnings.append(BiasWarning(
                        bias_type="LOOK_AHEAD_BIAS",
                        severity="CRITICAL",
                        description=f"Trade on {symbol} at {trade_date} uses future data",
                        recommendation="Ensure all trading decisions use only historical data available at decision time",
                        confidence=0.9,
                        metadata={'trade': trade}
                    ))
        
        # Check for suspicious performance patterns that might indicate look-ahead bias
        returns = backtest_results.get('returns', pd.Series())
        if len(returns) > 0:
            # Check for unrealistically consistent performance
            rolling_sharpe = returns.rolling(window=252).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            ).dropna()
            
            if len(rolling_sharpe) > 0:
                # If Sharpe ratio is consistently very high, it might indicate look-ahead bias
                high_sharpe_periods = (rolling_sharpe > 3.0).sum()
                if high_sharpe_periods / len(rolling_sharpe) > 0.5:
                    warnings.append(BiasWarning(
                        bias_type="LOOK_AHEAD_BIAS",
                        severity="HIGH",
                        description="Unrealistically consistent high performance may indicate look-ahead bias",
                        recommendation="Review strategy implementation for use of future information",
                        confidence=0.7,
                        metadata={'high_sharpe_ratio_periods': high_sharpe_periods}
                    ))
        
        return warnings
    
    def _detect_survivorship_bias(
        self,
        backtest_results: Dict[str, Any],
        universe_history: List[Set[str]]
    ) -> List[BiasWarning]:
        """Detect survivorship bias in symbol selection."""
        
        warnings = []
        
        if len(universe_history) < 2:
            return warnings
        
        # Analyze universe changes over time
        initial_universe = universe_history[0]
        final_universe = universe_history[-1]
        
        # Check for symbols that disappeared
        disappeared_symbols = initial_universe - final_universe
        disappeared_ratio = len(disappeared_symbols) / len(initial_universe) if initial_universe else 0
        
        if disappeared_ratio > 0.2:  # More than 20% of symbols disappeared
            warnings.append(BiasWarning(
                bias_type="SURVIVORSHIP_BIAS",
                severity="HIGH",
                description=f"{disappeared_ratio:.1%} of initial symbols disappeared from universe",
                recommendation="Include delisted/failed companies in backtest or use point-in-time universe",
                confidence=0.8,
                metadata={
                    'disappeared_symbols': list(disappeared_symbols),
                    'disappeared_ratio': disappeared_ratio
                }
            ))
        
        # Check if backtest only used symbols that survived to the end
        tested_symbols = set(backtest_results.get('symbols', []))
        survivor_overlap = len(tested_symbols & final_universe) / len(tested_symbols) if tested_symbols else 0
        
        if survivor_overlap > 0.9:  # More than 90% overlap with survivors
            warnings.append(BiasWarning(
                bias_type="SURVIVORSHIP_BIAS",
                severity="MEDIUM",
                description="Backtest heavily biased toward symbols that survived to present",
                recommendation="Use point-in-time universe selection to avoid survivorship bias",
                confidence=0.7,
                metadata={'survivor_overlap': survivor_overlap}
            ))
        
        return warnings
    
    def _detect_data_snooping_bias(
        self,
        backtest_results: Dict[str, Any],
        strategy_config: Dict[str, Any]
    ) -> List[BiasWarning]:
        """Detect data snooping bias from excessive parameter testing."""
        
        warnings = []
        
        # Track parameter testing history
        self.parameter_test_history.append(strategy_config)
        
        if len(self.parameter_test_history) > 50:  # Many parameter combinations tested
            warnings.append(BiasWarning(
                bias_type="DATA_SNOOPING_BIAS",
                severity="HIGH",
                description=f"Extensive parameter testing ({len(self.parameter_test_history)} combinations) increases data snooping risk",
                recommendation="Use out-of-sample testing and adjust for multiple testing",
                confidence=0.8,
                metadata={'parameter_tests': len(self.parameter_test_history)}
            ))
        
        # Check for suspiciously perfect parameter values
        params = strategy_config.get('parameters', {})
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float)):
                # Check if parameter is a "round number" which might indicate cherry-picking
                if param_value in [5, 10, 15, 20, 25, 30, 50, 100, 200]:
                    warnings.append(BiasWarning(
                        bias_type="DATA_SNOOPING_BIAS",
                        severity="LOW",
                        description=f"Parameter {param_name}={param_value} is a round number, possibly cherry-picked",
                        recommendation="Validate parameter selection with out-of-sample testing",
                        confidence=0.4,
                        metadata={'parameter': param_name, 'value': param_value}
                    ))
        
        return warnings
    
    def _detect_selection_bias(
        self,
        backtest_results: Dict[str, Any],
        market_data: pd.DataFrame
    ) -> List[BiasWarning]:
        """Detect selection bias in time period or market conditions."""
        
        warnings = []
        
        # Check for cherry-picked time periods
        start_date = backtest_results.get('start_date')
        end_date = backtest_results.get('end_date')
        
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Check if period is suspiciously short
            period_days = (end_date - start_date).days
            if period_days < 365:
                warnings.append(BiasWarning(
                    bias_type="SELECTION_BIAS",
                    severity="MEDIUM",
                    description=f"Backtest period is very short ({period_days} days)",
                    recommendation="Use longer backtest periods to ensure robustness",
                    confidence=0.6,
                    metadata={'period_days': period_days}
                ))
            
            # Check if period avoids major market downturns
            if len(market_data) > 0:
                period_data = market_data[
                    (market_data.index >= start_date) & 
                    (market_data.index <= end_date)
                ]
                
                if len(period_data) > 0:
                    # Calculate market returns during period
                    market_returns = period_data.pct_change().mean(axis=1).dropna()
                    
                    if len(market_returns) > 0:
                        # Check for absence of significant drawdowns
                        cumulative_returns = (1 + market_returns).cumprod()
                        max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
                        
                        if max_drawdown > -0.1:  # Less than 10% drawdown
                            warnings.append(BiasWarning(
                                bias_type="SELECTION_BIAS",
                                severity="MEDIUM",
                                description="Backtest period avoids significant market downturns",
                                recommendation="Include various market conditions in backtest period",
                                confidence=0.7,
                                metadata={'max_drawdown': max_drawdown}
                            ))
        
        return warnings
    
    def _detect_overfitting(
        self,
        backtest_results: Dict[str, Any],
        strategy_config: Dict[str, Any]
    ) -> List[BiasWarning]:
        """Detect signs of overfitting in strategy results."""
        
        warnings = []
        
        # Check for unrealistically high Sharpe ratio
        performance = backtest_results.get('performance_metrics', {})
        sharpe_ratio = performance.get('sharpe_ratio', 0)
        
        if sharpe_ratio > 3.0:
            warnings.append(BiasWarning(
                bias_type="OVERFITTING",
                severity="HIGH",
                description=f"Extremely high Sharpe ratio ({sharpe_ratio:.2f}) may indicate overfitting",
                recommendation="Validate results with out-of-sample testing and walk-forward analysis",
                confidence=0.8,
                metadata={'sharpe_ratio': sharpe_ratio}
            ))
        
        # Check for too many parameters relative to data
        params = strategy_config.get('parameters', {})
        num_params = len(params)
        
        returns = backtest_results.get('returns', pd.Series())
        num_observations = len(returns)
        
        if num_observations > 0:
            param_to_obs_ratio = num_params / num_observations
            
            if param_to_obs_ratio > 0.01:  # More than 1 parameter per 100 observations
                warnings.append(BiasWarning(
                    bias_type="OVERFITTING",
                    severity="MEDIUM",
                    description=f"High parameter-to-observation ratio ({param_to_obs_ratio:.4f})",
                    recommendation="Reduce model complexity or increase sample size",
                    confidence=0.6,
                    metadata={
                        'num_parameters': num_params,
                        'num_observations': num_observations,
                        'ratio': param_to_obs_ratio
                    }
                ))
        
        # Check for perfect or near-perfect win rate
        win_rate = performance.get('win_rate', 0)
        if win_rate > 0.95:
            warnings.append(BiasWarning(
                bias_type="OVERFITTING",
                severity="HIGH",
                description=f"Unrealistically high win rate ({win_rate:.1%}) suggests overfitting",
                recommendation="Validate with out-of-sample data and check for look-ahead bias",
                confidence=0.9,
                metadata={'win_rate': win_rate}
            ))
        
        # Check for excessive optimization (curve fitting)
        if len(returns) > 0:
            # Calculate rolling performance to detect inconsistency
            if len(returns) >= 252:  # At least 1 year of data
                rolling_returns = returns.rolling(window=63).mean()  # Quarterly rolling returns
                rolling_volatility = returns.rolling(window=63).std()
                
                # Check for excessive performance variation
                performance_cv = rolling_returns.std() / abs(rolling_returns.mean()) if rolling_returns.mean() != 0 else float('inf')
                
                if performance_cv > 2.0:  # High coefficient of variation
                    warnings.append(BiasWarning(
                        bias_type="OVERFITTING",
                        severity="MEDIUM",
                        description=f"High performance variability (CV: {performance_cv:.2f}) suggests overfitting",
                        recommendation="Strategy may be over-optimized to specific market conditions",
                        confidence=0.7,
                        metadata={'performance_cv': performance_cv}
                    ))
        
        return warnings
    
    def _check_statistical_significance(
        self,
        backtest_results: Dict[str, Any]
    ) -> List[BiasWarning]:
        """Check statistical significance of backtest results."""
        
        warnings = []
        
        returns = backtest_results.get('returns', pd.Series())
        
        if len(returns) == 0:
            return warnings
        
        # Perform t-test for mean return significance
        if len(returns) > 30:  # Sufficient sample size for t-test
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            
            if p_value > 0.05:  # Not statistically significant at 5% level
                warnings.append(BiasWarning(
                    bias_type="STATISTICAL_SIGNIFICANCE",
                    severity="MEDIUM",
                    description=f"Returns not statistically significant (p-value: {p_value:.3f})",
                    recommendation="Increase sample size or accept that results may be due to chance",
                    confidence=0.8,
                    metadata={'p_value': p_value, 't_statistic': t_stat}
                ))
        
        # Check for sufficient number of trades
        trades = backtest_results.get('trades', [])
        num_trades = len(trades)
        
        if num_trades < 30:
            warnings.append(BiasWarning(
                bias_type="STATISTICAL_SIGNIFICANCE",
                severity="MEDIUM",
                description=f"Insufficient number of trades ({num_trades}) for statistical reliability",
                recommendation="Increase trading frequency or extend backtest period",
                confidence=0.7,
                metadata={'num_trades': num_trades}
            ))
        
        return warnings
    
    def _calculate_risk_score(self, warnings: List[BiasWarning]) -> float:
        """Calculate overall bias risk score."""
        
        if not warnings:
            return 0.0
        
        severity_weights = {
            'LOW': 0.1,
            'MEDIUM': 0.3,
            'HIGH': 0.7,
            'CRITICAL': 1.0
        }
        
        total_score = 0.0
        for warning in warnings:
            weight = severity_weights.get(warning.severity, 0.5)
            total_score += weight * warning.confidence
        
        # Normalize to 0-1 range
        max_possible_score = len(warnings) * 1.0
        return min(total_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
    
    def _generate_summary(
        self,
        warnings: List[BiasWarning],
        risk_score: float,
        is_reliable: bool
    ) -> str:
        """Generate a summary of the bias analysis."""
        
        if not warnings:
            return "No significant biases detected. Backtest results appear reliable."
        
        summary_parts = []
        
        # Count warnings by severity
        severity_counts = {}
        for warning in warnings:
            severity_counts[warning.severity] = severity_counts.get(warning.severity, 0) + 1
        
        summary_parts.append(f"Bias Risk Score: {risk_score:.2f}/1.0")
        summary_parts.append(f"Reliability: {'RELIABLE' if is_reliable else 'QUESTIONABLE'}")
        
        if severity_counts:
            counts_str = ", ".join([f"{count} {severity}" for severity, count in severity_counts.items()])
            summary_parts.append(f"Warnings: {counts_str}")
        
        # Highlight most critical issues
        critical_warnings = [w for w in warnings if w.severity == 'CRITICAL']
        if critical_warnings:
            summary_parts.append("CRITICAL ISSUES:")
            for warning in critical_warnings[:3]:  # Show top 3
                summary_parts.append(f"  - {warning.description}")
        
        return "\n".join(summary_parts)
    
    def get_bias_report(self, analysis_result: BiasAnalysisResult) -> str:
        """Generate a detailed bias analysis report."""
        
        report = []
        report.append("=== BIAS ANALYSIS REPORT ===")
        report.append(f"Overall Risk Score: {analysis_result.overall_risk_score:.2f}/1.0")
        report.append(f"Reliability Assessment: {'RELIABLE' if analysis_result.is_reliable else 'QUESTIONABLE'}")
        report.append("")
        
        if not analysis_result.warnings:
            report.append("✓ No significant biases detected.")
            return "\n".join(report)
        
        # Group warnings by type
        warnings_by_type = {}
        for warning in analysis_result.warnings:
            if warning.bias_type not in warnings_by_type:
                warnings_by_type[warning.bias_type] = []
            warnings_by_type[warning.bias_type].append(warning)
        
        for bias_type, warnings in warnings_by_type.items():
            report.append(f"=== {bias_type.replace('_', ' ').title()} ===")
            
            for warning in warnings:
                severity_symbol = {
                    'LOW': '⚠️',
                    'MEDIUM': '⚠️',
                    'HIGH': '🚨',
                    'CRITICAL': '🔴'
                }.get(warning.severity, '⚠️')
                
                report.append(f"{severity_symbol} {warning.severity}: {warning.description}")
                report.append(f"   Recommendation: {warning.recommendation}")
                report.append(f"   Confidence: {warning.confidence:.1%}")
                report.append("")
        
        report.append("=== SUMMARY ===")
        report.append(analysis_result.summary)
        
        return "\n".join(report)
    
    def clear_history(self) -> None:
        """Clear analysis history."""
        self.analysis_history.clear()
        self.symbol_universe_history.clear()
        self.parameter_test_history.clear()