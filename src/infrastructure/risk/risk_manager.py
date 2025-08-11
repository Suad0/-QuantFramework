"""
Comprehensive risk management system orchestrator.

This module provides the main RiskManager class that coordinates all risk management
components including VaR calculation, risk monitoring, stress testing, factor models,
liquidity analysis, and concentration monitoring.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import warnings

from ...domain.entities import Portfolio
from ...domain.value_objects import RiskMetrics
from ...domain.interfaces import IRiskManager
from ...domain.exceptions import ValidationError

from .var_calculator import VaRCalculator, VaRMethod, VaRResult
from .risk_monitor import RiskMonitor, RiskAlert, AlertSeverity, RiskLimit, AlertType
from .stress_tester import StressTester, StressScenario, StressResult
from .factor_risk_model import FactorRiskModel, FactorExposure, RiskAttribution
from .liquidity_analyzer import LiquidityAnalyzer, LiquidityMetrics, PortfolioLiquidityProfile
from .concentration_monitor import ConcentrationMonitor, ConcentrationMetrics, ConcentrationLimits


@dataclass
class RiskManagerConfig:
    """Configuration for the risk management system."""
    var_lookback_days: int = 252
    var_confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    monitoring_interval: int = 60  # seconds
    enable_real_time_monitoring: bool = True
    enable_stress_testing: bool = True
    enable_factor_model: bool = True
    enable_liquidity_analysis: bool = True
    enable_concentration_monitoring: bool = True
    max_workers: int = 4


@dataclass
class ComprehensiveRiskReport:
    """Comprehensive risk analysis report."""
    portfolio_id: str
    timestamp: datetime
    
    # VaR and risk metrics
    var_result: VaRResult
    risk_metrics: RiskMetrics
    
    # Stress testing
    stress_results: Dict[str, StressResult]
    
    # Factor analysis
    factor_exposure: Optional[FactorExposure]
    risk_attribution: Optional[RiskAttribution]
    
    # Liquidity analysis
    liquidity_profile: PortfolioLiquidityProfile
    
    # Concentration analysis
    concentration_metrics: ConcentrationMetrics
    
    # Alerts and recommendations
    active_alerts: List[RiskAlert]
    risk_score: float  # Overall risk score (0-100)
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'portfolio_id': self.portfolio_id,
            'timestamp': self.timestamp.isoformat(),
            'var_95': self.var_result.var_95,
            'var_99': self.var_result.var_99,
            'cvar_95': self.var_result.cvar_95,
            'volatility': self.risk_metrics.volatility,
            'max_drawdown': self.risk_metrics.max_drawdown,
            'beta': self.risk_metrics.beta,
            'stress_scenarios': len(self.stress_results),
            'worst_stress_loss': min(result.pnl_percent for result in self.stress_results.values()) if self.stress_results else 0,
            'liquidity_score': self.liquidity_profile.get_liquidity_score(),
            'concentration_score': self.concentration_metrics.concentration_score,
            'active_alerts': len(self.active_alerts),
            'critical_alerts': len([a for a in self.active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'risk_score': self.risk_score,
            'recommendations_count': len(self.recommendations)
        }


class RiskManager(IRiskManager):
    """
    Comprehensive risk management system.
    
    This class orchestrates all risk management components to provide
    a unified interface for portfolio risk analysis and monitoring.
    """
    
    def __init__(self, config: Optional[RiskManagerConfig] = None):
        """
        Initialize the risk management system.
        
        Args:
            config: Risk manager configuration
        """
        self.config = config or RiskManagerConfig()
        
        # Initialize risk management components
        self.var_calculator = VaRCalculator(
            lookback_days=self.config.var_lookback_days
        )
        
        self.risk_monitor = RiskMonitor(
            var_calculator=self.var_calculator,
            monitoring_interval=self.config.monitoring_interval,
            max_workers=self.config.max_workers
        )
        
        self.stress_tester = StressTester(
            confidence_levels=self.config.var_confidence_levels
        )
        
        self.factor_model = FactorRiskModel(
            lookback_days=self.config.var_lookback_days
        ) if self.config.enable_factor_model else None
        
        self.liquidity_analyzer = LiquidityAnalyzer(
            lookback_days=min(60, self.config.var_lookback_days)
        ) if self.config.enable_liquidity_analysis else None
        
        self.concentration_monitor = ConcentrationMonitor(
            limits=ConcentrationLimits()
        ) if self.config.enable_concentration_monitoring else None
        
        # Risk calculation cache
        self.risk_cache: Dict[str, Dict[str, Any]] = {}
        
        # Setup default risk limits
        self._setup_default_risk_limits()
        
        # Start monitoring if enabled
        if self.config.enable_real_time_monitoring:
            self.start_monitoring()
    
    def _setup_default_risk_limits(self) -> None:
        """Setup default risk limits for monitoring."""
        default_limits = [
            RiskLimit(
                name="var_95_limit",
                limit_type=AlertType.VAR_BREACH,
                threshold=0.05,  # 5% VaR limit
                severity=AlertSeverity.HIGH
            ),
            RiskLimit(
                name="volatility_limit",
                limit_type=AlertType.VOLATILITY,
                threshold=0.25,  # 25% volatility limit
                severity=AlertSeverity.MEDIUM
            ),
            RiskLimit(
                name="drawdown_limit",
                limit_type=AlertType.DRAWDOWN,
                threshold=0.15,  # 15% drawdown limit
                severity=AlertSeverity.HIGH
            ),
            RiskLimit(
                name="concentration_limit",
                limit_type=AlertType.CONCENTRATION,
                threshold=0.10,  # 10% single position limit
                severity=AlertSeverity.MEDIUM
            )
        ]
        
        # Add limits to monitor (would typically be portfolio-specific)
        for limit in default_limits:
            self.risk_monitor.add_risk_limit("default_portfolio", limit)
    
    def calculate_portfolio_risk(self, portfolio: Portfolio) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        # Convert portfolio to data format
        portfolio_data = self._portfolio_to_data(portfolio)
        
        # Check cache
        cache_key = f"{portfolio.id}_{portfolio.updated_at.timestamp()}"
        if cache_key in self.risk_cache:
            cached_result = self.risk_cache[cache_key]
            if (datetime.now() - cached_result['timestamp']).seconds < 300:  # 5 min cache
                return cached_result['risk_metrics']
        
        # Calculate portfolio returns (simplified)
        returns = self._calculate_portfolio_returns(portfolio_data)
        
        # Calculate VaR
        var_result = self.var_calculator.calculate_var(
            returns,
            method=VaRMethod.HISTORICAL,
            portfolio_value=portfolio_data['total_value']
        )
        
        # Calculate additional risk metrics
        volatility = returns.std() * np.sqrt(252)  # Annualized
        max_drawdown = self._calculate_max_drawdown(returns)
        beta = self._calculate_beta(returns)
        tracking_error = self._calculate_tracking_error(returns)
        
        risk_metrics = RiskMetrics(
            var_95=var_result.var_95,
            var_99=var_result.var_99,
            cvar_95=var_result.cvar_95,
            volatility=volatility,
            max_drawdown=max_drawdown,
            beta=beta,
            tracking_error=tracking_error,
            timestamp=datetime.now()
        )
        
        # Cache result
        self.risk_cache[cache_key] = {
            'risk_metrics': risk_metrics,
            'timestamp': datetime.now()
        }
        
        return risk_metrics
    
    def monitor_risk_limits(self, portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Monitor portfolio against risk limits."""
        portfolio_data = self._portfolio_to_data(portfolio)
        
        # Get alerts from risk monitor
        alerts = self.risk_monitor.get_active_alerts(portfolio.id)
        
        # Add concentration alerts if enabled
        if self.concentration_monitor:
            concentration_alerts = self.concentration_monitor.check_concentration_limits(portfolio_data)
            # Convert to risk alerts format
            for alert in concentration_alerts:
                risk_alert = RiskAlert(
                    alert_id=alert.alert_id,
                    portfolio_id=alert.portfolio_id,
                    alert_type=AlertType.CONCENTRATION,
                    severity=AlertSeverity.MEDIUM,  # Map concentration severity
                    message=alert.message,
                    current_value=alert.current_exposure,
                    threshold=alert.limit,
                    timestamp=alert.timestamp
                )
                alerts.append(risk_alert)
        
        return [alert.to_dict() for alert in alerts]
    
    def stress_test(
        self,
        portfolio: Portfolio,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform comprehensive stress testing."""
        if not self.config.enable_stress_testing:
            return {'status': 'stress_testing_disabled'}
        
        portfolio_data = self._portfolio_to_data(portfolio)
        
        # Run predefined scenarios
        predefined_results = self.stress_tester.run_all_scenarios(portfolio_data)
        
        # Run custom scenarios if provided
        custom_results = {}
        for scenario_config in scenarios:
            scenario_name = scenario_config.get('name', f'custom_{len(custom_results)}')
            scenario = StressScenario(
                name=scenario_name,
                scenario_type=scenario_config.get('type', 'custom'),
                description=scenario_config.get('description', 'Custom scenario'),
                parameters=scenario_config.get('parameters', {})
            )
            
            self.stress_tester.add_scenario(scenario)
            result = self.stress_tester.run_stress_test(portfolio_data, scenario_name)
            custom_results[scenario_name] = result
        
        # Combine results
        all_results = {**predefined_results, **custom_results}
        
        # Calculate summary statistics
        worst_case = min(result.pnl_percent for result in all_results.values()) if all_results else 0
        average_loss = np.mean([result.pnl_percent for result in all_results.values()]) if all_results else 0
        
        return {
            'portfolio_id': portfolio.id,
            'timestamp': datetime.now(),
            'scenarios_tested': len(all_results),
            'worst_case_loss': worst_case,
            'average_loss': average_loss,
            'scenario_results': {name: result.__dict__ for name, result in all_results.items()},
            'recommendations': self._generate_stress_test_recommendations(all_results)
        }
    
    def generate_comprehensive_report(
        self,
        portfolio: Portfolio,
        include_stress_testing: bool = True,
        include_factor_analysis: bool = True
    ) -> ComprehensiveRiskReport:
        """Generate comprehensive risk analysis report."""
        portfolio_data = self._portfolio_to_data(portfolio)
        
        # Calculate VaR and basic risk metrics
        returns = self._calculate_portfolio_returns(portfolio_data)
        var_result = self.var_calculator.calculate_var(
            returns,
            method=VaRMethod.HISTORICAL,
            portfolio_value=portfolio_data['total_value']
        )
        
        risk_metrics = self.calculate_portfolio_risk(portfolio)
        
        # Stress testing
        stress_results = {}
        if include_stress_testing and self.config.enable_stress_testing:
            stress_results = self.stress_tester.run_all_scenarios(portfolio_data)
        
        # Factor analysis
        factor_exposure = None
        risk_attribution = None
        if include_factor_analysis and self.factor_model:
            try:
                weights = pd.Series({
                    pos.symbol: pos.quantity * pos.current_price / portfolio_data['total_value']
                    for pos in portfolio.positions.values()
                })
                factor_exposure = self.factor_model.calculate_portfolio_exposures(weights)
                risk_attribution = self.factor_model.calculate_risk_attribution(weights)
            except Exception as e:
                warnings.warn(f"Factor analysis failed: {e}")
        
        # Liquidity analysis
        liquidity_profile = PortfolioLiquidityProfile(
            portfolio_id=portfolio.id,
            total_value=portfolio_data['total_value'],
            liquid_assets_percent=0.7,  # Default
            weighted_avg_spread=0.01,
            weighted_time_to_liquidate=3.0,
            liquidity_concentration=0.3,
            stress_liquidation_cost=0.05,
            tier_distribution={'tier_1': 0.5, 'tier_2': 0.3, 'tier_3': 0.2},
            largest_position_liquidity_days=2.0,
            timestamp=datetime.now()
        )
        
        if self.liquidity_analyzer:
            liquidity_profile = self.liquidity_analyzer.calculate_portfolio_liquidity(portfolio_data)
        
        # Concentration analysis
        concentration_metrics = ConcentrationMetrics(
            portfolio_id=portfolio.id,
            herfindahl_index=0.1,
            effective_positions=10.0,
            top_5_concentration=0.4,
            top_10_concentration=0.6,
            max_position_weight=0.08,
            position_concentration={},
            sector_concentration={},
            geography_concentration={},
            currency_concentration={},
            concentration_score=30.0,
            timestamp=datetime.now()
        )
        
        if self.concentration_monitor:
            concentration_metrics = self.concentration_monitor.calculate_concentration_metrics(portfolio_data)
        
        # Get active alerts
        active_alerts = self.risk_monitor.get_active_alerts(portfolio.id)
        
        # Calculate overall risk score
        risk_score = self._calculate_overall_risk_score(
            var_result, risk_metrics, stress_results, liquidity_profile, concentration_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(
            var_result, risk_metrics, stress_results, liquidity_profile, 
            concentration_metrics, active_alerts
        )
        
        return ComprehensiveRiskReport(
            portfolio_id=portfolio.id,
            timestamp=datetime.now(),
            var_result=var_result,
            risk_metrics=risk_metrics,
            stress_results=stress_results,
            factor_exposure=factor_exposure,
            risk_attribution=risk_attribution,
            liquidity_profile=liquidity_profile,
            concentration_metrics=concentration_metrics,
            active_alerts=active_alerts,
            risk_score=risk_score,
            recommendations=recommendations
        )
    
    def _portfolio_to_data(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Convert Portfolio entity to data dictionary."""
        positions_data = {}
        total_value = 0
        
        for symbol, position in portfolio.positions.items():
            position_value = float(position.quantity * position.current_price)
            total_value += position_value
            
            positions_data[symbol] = {
                'quantity': float(position.quantity),
                'price': float(position.current_price),
                'value': position_value,
                'weight': 0  # Will be calculated below
            }
        
        # Calculate weights
        if total_value > 0:
            for symbol in positions_data:
                positions_data[symbol]['weight'] = positions_data[symbol]['value'] / total_value
        
        # Add sector exposures (simplified)
        sector_exposures = {
            'Technology': 0.4,
            'Healthcare': 0.2,
            'Financials': 0.2,
            'Consumer': 0.2
        }
        
        return {
            'portfolio_id': portfolio.id,
            'total_value': total_value,
            'positions': positions_data,
            'sector_exposures': sector_exposures,
            'timestamp': datetime.now()
        }
    
    def _calculate_portfolio_returns(self, portfolio_data: Dict[str, Any]) -> pd.Series:
        """Calculate portfolio returns (simplified implementation)."""
        # In practice, this would use historical price data
        # For demonstration, generate synthetic returns
        
        np.random.seed(hash(portfolio_data['portfolio_id']) % 2**32)
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=self.config.var_lookback_days),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate synthetic returns based on portfolio composition
        base_volatility = 0.15  # 15% annual volatility
        daily_vol = base_volatility / np.sqrt(252)
        
        returns = pd.Series(
            np.random.normal(0.0005, daily_vol, len(dates)),
            index=dates
        )
        
        return returns
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate portfolio beta (simplified)."""
        # In practice, this would use market benchmark returns
        # For demonstration, return a typical beta value
        return 1.1
    
    def _calculate_tracking_error(self, returns: pd.Series) -> float:
        """Calculate tracking error vs benchmark."""
        # In practice, this would compare to benchmark returns
        # For demonstration, return a typical tracking error
        return 0.05
    
    def _calculate_overall_risk_score(
        self,
        var_result: VaRResult,
        risk_metrics: RiskMetrics,
        stress_results: Dict[str, StressResult],
        liquidity_profile: PortfolioLiquidityProfile,
        concentration_metrics: ConcentrationMetrics
    ) -> float:
        """Calculate overall risk score (0-100, higher is riskier)."""
        # VaR component (0-25 points)
        var_score = min(abs(var_result.var_95) * 500, 25)
        
        # Volatility component (0-20 points)
        vol_score = min(risk_metrics.volatility * 80, 20)
        
        # Stress testing component (0-25 points)
        if stress_results:
            worst_stress = min(result.pnl_percent for result in stress_results.values())
            stress_score = min(abs(worst_stress) * 0.5, 25)
        else:
            stress_score = 10  # Default if no stress testing
        
        # Liquidity component (0-15 points)
        liquidity_score = max(0, 15 - liquidity_profile.get_liquidity_score() * 0.15)
        
        # Concentration component (0-15 points)
        concentration_score = min(concentration_metrics.concentration_score * 0.25, 15)
        
        total_score = var_score + vol_score + stress_score + liquidity_score + concentration_score
        return min(total_score, 100)
    
    def _generate_risk_recommendations(
        self,
        var_result: VaRResult,
        risk_metrics: RiskMetrics,
        stress_results: Dict[str, StressResult],
        liquidity_profile: PortfolioLiquidityProfile,
        concentration_metrics: ConcentrationMetrics,
        active_alerts: List[RiskAlert]
    ) -> List[str]:
        """Generate risk management recommendations."""
        recommendations = []
        
        # VaR-based recommendations
        if abs(var_result.var_95) > 0.05:  # 5% threshold
            recommendations.append(
                f"High VaR detected ({var_result.var_95:.2%}). Consider reducing position sizes or hedging."
            )
        
        # Volatility recommendations
        if risk_metrics.volatility > 0.25:  # 25% threshold
            recommendations.append(
                f"High portfolio volatility ({risk_metrics.volatility:.2%}). Consider diversification."
            )
        
        # Drawdown recommendations
        if abs(risk_metrics.max_drawdown) > 0.15:  # 15% threshold
            recommendations.append(
                f"Significant drawdown risk ({risk_metrics.max_drawdown:.2%}). Review risk management strategy."
            )
        
        # Stress testing recommendations
        if stress_results:
            worst_case = min(result.pnl_percent for result in stress_results.values())
            if worst_case < -0.20:  # 20% loss threshold
                recommendations.append(
                    f"Severe stress test losses ({worst_case:.2%}). Consider stress hedging strategies."
                )
        
        # Liquidity recommendations
        if liquidity_profile.get_liquidity_score() < 50:
            recommendations.append(
                "Low portfolio liquidity. Increase allocation to liquid assets."
            )
        
        # Concentration recommendations
        if concentration_metrics.concentration_score > 50:
            recommendations.append(
                "High concentration risk. Diversify across more positions and sectors."
            )
        
        # Alert-based recommendations
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            recommendations.append(
                f"{len(critical_alerts)} critical risk alerts require immediate attention."
            )
        
        return recommendations
    
    def _generate_stress_test_recommendations(
        self,
        stress_results: Dict[str, StressResult]
    ) -> List[str]:
        """Generate stress testing specific recommendations."""
        recommendations = []
        
        if not stress_results:
            return recommendations
        
        # Find worst performing scenarios
        worst_scenarios = sorted(
            stress_results.items(),
            key=lambda x: x[1].pnl_percent
        )[:3]
        
        for scenario_name, result in worst_scenarios:
            if result.pnl_percent < -0.10:  # 10% loss threshold
                recommendations.append(
                    f"Vulnerable to {scenario_name}: {result.pnl_percent:.2%} loss. "
                    f"Consider hedging strategies."
                )
        
        return recommendations
    
    def start_monitoring(self) -> None:
        """Start real-time risk monitoring."""
        if self.config.enable_real_time_monitoring:
            self.risk_monitor.start_monitoring()
    
    def stop_monitoring(self) -> None:
        """Stop real-time risk monitoring."""
        self.risk_monitor.stop_monitoring()
    
    def add_risk_limit(self, portfolio_id: str, risk_limit: RiskLimit) -> None:
        """Add a risk limit for monitoring."""
        self.risk_monitor.add_risk_limit(portfolio_id, risk_limit)
    
    def export_risk_report(
        self,
        report: ComprehensiveRiskReport,
        filename: str,
        format: str = 'excel'
    ) -> None:
        """Export comprehensive risk report."""
        if format.lower() == 'excel':
            with pd.ExcelWriter(filename) as writer:
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Risk Score',
                        'VaR 95%',
                        'VaR 99%',
                        'CVaR 95%',
                        'Volatility',
                        'Max Drawdown',
                        'Beta',
                        'Liquidity Score',
                        'Concentration Score'
                    ],
                    'Value': [
                        report.risk_score,
                        report.var_result.var_95,
                        report.var_result.var_99,
                        report.var_result.cvar_95,
                        report.risk_metrics.volatility,
                        report.risk_metrics.max_drawdown,
                        report.risk_metrics.beta,
                        report.liquidity_profile.get_liquidity_score(),
                        report.concentration_metrics.concentration_score
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Stress test results
                if report.stress_results:
                    stress_data = []
                    for scenario_name, result in report.stress_results.items():
                        stress_data.append({
                            'Scenario': scenario_name,
                            'PnL %': result.pnl_percent,
                            'PnL Amount': result.pnl,
                            'VaR 95%': result.var_95,
                            'Max Drawdown': result.max_drawdown
                        })
                    pd.DataFrame(stress_data).to_excel(writer, sheet_name='Stress_Tests', index=False)
                
                # Recommendations
                recommendations_data = [
                    {'Recommendation': rec} for rec in report.recommendations
                ]
                pd.DataFrame(recommendations_data).to_excel(writer, sheet_name='Recommendations', index=False)
        
        elif format.lower() == 'csv':
            # Export summary as CSV
            summary_dict = report.to_dict()
            pd.DataFrame([summary_dict]).to_csv(filename, index=False)
        
        else:
            raise ValidationError(f"Unsupported export format: {format}")
    
    def get_risk_dashboard_data(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Get risk dashboard data for UI display."""
        risk_metrics = self.calculate_portfolio_risk(portfolio)
        alerts = self.monitor_risk_limits(portfolio)
        
        # Get component-specific dashboard data
        portfolio_data = self._portfolio_to_data(portfolio)
        
        dashboard_data = {
            'portfolio_id': portfolio.id,
            'timestamp': datetime.now(),
            'risk_metrics': {
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'cvar_95': risk_metrics.cvar_95,
                'volatility': risk_metrics.volatility,
                'max_drawdown': risk_metrics.max_drawdown,
                'beta': risk_metrics.beta,
                'tracking_error': risk_metrics.tracking_error
            },
            'alerts': {
                'total': len(alerts),
                'critical': len([a for a in alerts if a.get('severity') == 'critical']),
                'high': len([a for a in alerts if a.get('severity') == 'high']),
                'medium': len([a for a in alerts if a.get('severity') == 'medium'])
            }
        }
        
        # Add component-specific data
        if self.liquidity_analyzer:
            liquidity_profile = self.liquidity_analyzer.calculate_portfolio_liquidity(portfolio_data)
            dashboard_data['liquidity'] = {
                'score': liquidity_profile.get_liquidity_score(),
                'liquid_assets_percent': liquidity_profile.liquid_assets_percent,
                'time_to_liquidate': liquidity_profile.weighted_time_to_liquidate
            }
        
        if self.concentration_monitor:
            concentration_metrics = self.concentration_monitor.calculate_concentration_metrics(portfolio_data)
            dashboard_data['concentration'] = {
                'score': concentration_metrics.concentration_score,
                'herfindahl_index': concentration_metrics.herfindahl_index,
                'effective_positions': concentration_metrics.effective_positions,
                'max_position_weight': concentration_metrics.max_position_weight
            }
        
        return dashboard_data
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.risk_cache.clear()
        if self.liquidity_analyzer:
            self.liquidity_analyzer.clear_cache()
    
    def __del__(self):
        """Cleanup when risk manager is destroyed."""
        try:
            self.stop_monitoring()
        except:
            pass