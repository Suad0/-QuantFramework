"""
Stress testing system with scenario analysis and Monte Carlo simulations.

This module provides comprehensive stress testing capabilities including:
- Historical scenario analysis
- Monte Carlo stress testing
- Custom scenario definition
- Stress test reporting and visualization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass, field
from scipy import stats
from scipy.optimize import minimize
import warnings

from ...domain.entities import Portfolio, Position
from ...domain.exceptions import ValidationError


class ScenarioType(Enum):
    """Types of stress test scenarios."""
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"
    CUSTOM = "custom"
    FACTOR_SHOCK = "factor_shock"


class ShockType(Enum):
    """Types of market shocks."""
    MARKET_CRASH = "market_crash"
    INTEREST_RATE = "interest_rate"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    CREDIT = "credit"
    CURRENCY = "currency"
    COMMODITY = "commodity"


@dataclass
class StressScenario:
    """Stress test scenario definition."""
    name: str
    scenario_type: ScenarioType
    description: str
    parameters: Dict[str, Any]
    enabled: bool = True
    created_date: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> None:
        """Validate scenario parameters."""
        if not self.name:
            raise ValidationError("Scenario name cannot be empty")
        if not self.parameters:
            raise ValidationError("Scenario parameters cannot be empty")


@dataclass
class StressResult:
    """Result of a stress test."""
    scenario_name: str
    portfolio_id: str
    base_value: float
    stressed_value: float
    pnl: float
    pnl_percent: float
    var_95: float
    var_99: float
    max_drawdown: float
    volatility: float
    position_impacts: Dict[str, float]
    sector_impacts: Dict[str, float]
    factor_contributions: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def loss_amount(self) -> float:
        """Get loss amount (positive value)."""
        return -min(0, self.pnl)
    
    @property
    def loss_percent(self) -> float:
        """Get loss percentage (positive value)."""
        return -min(0, self.pnl_percent)


class StressTester:
    """
    Comprehensive stress testing system.
    
    This class provides various stress testing methodologies including
    historical scenario analysis, Monte Carlo simulations, and custom scenarios.
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize stress tester.
        
        Args:
            confidence_levels: Confidence levels for VaR calculations
        """
        self.confidence_levels = confidence_levels
        self.scenarios: Dict[str, StressScenario] = {}
        self.results_history: List[StressResult] = []
        
        # Load predefined scenarios
        self._load_predefined_scenarios()
    
    def _load_predefined_scenarios(self) -> None:
        """Load predefined stress test scenarios."""
        # 2008 Financial Crisis
        self.add_scenario(StressScenario(
            name="2008_financial_crisis",
            scenario_type=ScenarioType.HISTORICAL,
            description="2008 Financial Crisis scenario based on historical data",
            parameters={
                "start_date": "2008-09-01",
                "end_date": "2009-03-31",
                "market_shock": -0.45,
                "volatility_multiplier": 2.5,
                "correlation_increase": 0.3
            }
        ))
        
        # COVID-19 Market Crash
        self.add_scenario(StressScenario(
            name="covid_crash",
            scenario_type=ScenarioType.HISTORICAL,
            description="COVID-19 market crash scenario",
            parameters={
                "start_date": "2020-02-20",
                "end_date": "2020-04-30",
                "market_shock": -0.35,
                "volatility_multiplier": 3.0,
                "correlation_increase": 0.4
            }
        ))
        
        # Interest Rate Shock
        self.add_scenario(StressScenario(
            name="interest_rate_shock",
            scenario_type=ScenarioType.FACTOR_SHOCK,
            description="Sudden interest rate increase",
            parameters={
                "rate_change": 0.02,  # 200 bps increase
                "duration_impact": -0.05,  # Duration-based impact
                "sector_impacts": {
                    "Financials": 0.1,
                    "Real Estate": -0.15,
                    "Utilities": -0.12,
                    "Technology": -0.08
                }
            }
        ))
        
        # Volatility Shock
        self.add_scenario(StressScenario(
            name="volatility_shock",
            scenario_type=ScenarioType.MONTE_CARLO,
            description="Extreme volatility increase",
            parameters={
                "volatility_multiplier": 2.0,
                "correlation_increase": 0.2,
                "n_simulations": 10000,
                "time_horizon": 30  # days
            }
        ))
        
        # Market Crash
        self.add_scenario(StressScenario(
            name="market_crash",
            scenario_type=ScenarioType.CUSTOM,
            description="Severe market crash scenario",
            parameters={
                "market_shock": -0.30,
                "volatility_multiplier": 2.5,
                "correlation_increase": 0.5,
                "liquidity_impact": 0.02  # 2% liquidity cost
            }
        ))
    
    def add_scenario(self, scenario: StressScenario) -> None:
        """Add a stress test scenario."""
        scenario.validate()
        self.scenarios[scenario.name] = scenario
    
    def remove_scenario(self, scenario_name: str) -> bool:
        """Remove a stress test scenario."""
        if scenario_name in self.scenarios:
            del self.scenarios[scenario_name]
            return True
        return False
    
    def run_stress_test(
        self,
        portfolio_data: Dict[str, Any],
        scenario_name: str,
        market_data: Optional[pd.DataFrame] = None
    ) -> StressResult:
        """
        Run stress test for a specific scenario.
        
        Args:
            portfolio_data: Portfolio data including positions and weights
            scenario_name: Name of the scenario to run
            market_data: Historical market data for scenario analysis
            
        Returns:
            StressResult with detailed impact analysis
        """
        if scenario_name not in self.scenarios:
            raise ValidationError(f"Scenario '{scenario_name}' not found")
        
        scenario = self.scenarios[scenario_name]
        
        if scenario.scenario_type == ScenarioType.HISTORICAL:
            return self._run_historical_scenario(portfolio_data, scenario, market_data)
        elif scenario.scenario_type == ScenarioType.MONTE_CARLO:
            return self._run_monte_carlo_scenario(portfolio_data, scenario)
        elif scenario.scenario_type == ScenarioType.CUSTOM:
            return self._run_custom_scenario(portfolio_data, scenario)
        elif scenario.scenario_type == ScenarioType.FACTOR_SHOCK:
            return self._run_factor_shock_scenario(portfolio_data, scenario)
        else:
            raise ValidationError(f"Unsupported scenario type: {scenario.scenario_type}")
    
    def _run_historical_scenario(
        self,
        portfolio_data: Dict[str, Any],
        scenario: StressScenario,
        market_data: Optional[pd.DataFrame]
    ) -> StressResult:
        """Run historical scenario stress test."""
        if market_data is None:
            # Generate synthetic historical data for demonstration
            market_data = self._generate_synthetic_market_data(scenario)
        
        # Extract scenario parameters
        params = scenario.parameters
        market_shock = params.get('market_shock', -0.2)
        volatility_multiplier = params.get('volatility_multiplier', 1.5)
        correlation_increase = params.get('correlation_increase', 0.2)
        
        # Calculate base portfolio value
        base_value = portfolio_data.get('total_value', 1000000)
        
        # Apply market shock to positions
        position_impacts = {}
        total_impact = 0
        
        for symbol, position in portfolio_data.get('positions', {}).items():
            # Base market impact
            position_shock = market_shock
            
            # Add position-specific factors (beta, sector, etc.)
            beta = self._get_position_beta(symbol, portfolio_data)
            position_shock *= beta
            
            # Calculate position impact
            position_value = position.get('value', 0)
            position_impact = position_value * position_shock
            position_impacts[symbol] = position_impact
            total_impact += position_impact
        
        # Calculate sector impacts
        sector_impacts = self._calculate_sector_impacts(
            portfolio_data, market_shock, volatility_multiplier
        )
        
        # Calculate stressed portfolio value
        stressed_value = base_value + total_impact
        pnl = total_impact
        pnl_percent = (pnl / base_value) * 100 if base_value > 0 else 0
        
        # Calculate risk metrics for stressed portfolio
        stressed_returns = self._generate_stressed_returns(
            portfolio_data, market_shock, volatility_multiplier
        )
        
        var_95 = np.percentile(stressed_returns, 5) * base_value
        var_99 = np.percentile(stressed_returns, 1) * base_value
        max_drawdown = np.min(np.cumsum(stressed_returns))
        volatility = np.std(stressed_returns) * np.sqrt(252)
        
        # Factor contributions (simplified)
        factor_contributions = {
            'market': market_shock * 0.7,
            'volatility': volatility_multiplier * 0.2,
            'correlation': correlation_increase * 0.1
        }
        
        result = StressResult(
            scenario_name=scenario.name,
            portfolio_id=portfolio_data.get('portfolio_id', 'unknown'),
            base_value=base_value,
            stressed_value=stressed_value,
            pnl=pnl,
            pnl_percent=pnl_percent,
            var_95=var_95,
            var_99=var_99,
            max_drawdown=max_drawdown,
            volatility=volatility,
            position_impacts=position_impacts,
            sector_impacts=sector_impacts,
            factor_contributions=factor_contributions,
            timestamp=datetime.now(),
            metadata={'scenario_type': scenario.scenario_type.value}
        )
        
        self.results_history.append(result)
        return result
    
    def _run_monte_carlo_scenario(
        self,
        portfolio_data: Dict[str, Any],
        scenario: StressScenario
    ) -> StressResult:
        """Run Monte Carlo stress test."""
        params = scenario.parameters
        n_simulations = params.get('n_simulations', 10000)
        time_horizon = params.get('time_horizon', 30)
        volatility_multiplier = params.get('volatility_multiplier', 2.0)
        correlation_increase = params.get('correlation_increase', 0.2)
        
        base_value = portfolio_data.get('total_value', 1000000)
        positions = portfolio_data.get('positions', {})
        
        # Generate correlation matrix
        n_assets = len(positions)
        base_correlation = 0.3
        stressed_correlation = min(0.9, base_correlation + correlation_increase)
        
        # Create correlation matrix
        corr_matrix = np.full((n_assets, n_assets), stressed_correlation)
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Generate random returns for each simulation
        np.random.seed(42)
        simulation_results = []
        
        for _ in range(n_simulations):
            # Generate correlated random returns
            random_returns = np.random.multivariate_normal(
                mean=np.zeros(n_assets),
                cov=corr_matrix,
                size=time_horizon
            )
            
            # Apply volatility scaling
            portfolio_returns = []
            for i, (symbol, position) in enumerate(positions.items()):
                weight = position.get('weight', 0)
                volatility = self._get_position_volatility(symbol, portfolio_data)
                scaled_returns = random_returns[:, i] * volatility * volatility_multiplier
                portfolio_returns.append(scaled_returns * weight)
            
            # Calculate portfolio return for this simulation
            if portfolio_returns:
                total_return = np.sum(portfolio_returns, axis=0).sum()
                simulation_results.append(total_return)
        
        simulation_results = np.array(simulation_results)
        
        # Calculate statistics
        mean_return = np.mean(simulation_results)
        worst_case = np.min(simulation_results)
        var_95 = np.percentile(simulation_results, 5) * base_value
        var_99 = np.percentile(simulation_results, 1) * base_value
        
        # Position impacts (average across simulations)
        position_impacts = {}
        for symbol, position in positions.items():
            weight = position.get('weight', 0)
            position_impact = mean_return * weight * base_value
            position_impacts[symbol] = position_impact
        
        # Sector impacts
        sector_impacts = self._calculate_sector_impacts(
            portfolio_data, mean_return, volatility_multiplier
        )
        
        stressed_value = base_value + (mean_return * base_value)
        pnl = mean_return * base_value
        pnl_percent = mean_return * 100
        
        result = StressResult(
            scenario_name=scenario.name,
            portfolio_id=portfolio_data.get('portfolio_id', 'unknown'),
            base_value=base_value,
            stressed_value=stressed_value,
            pnl=pnl,
            pnl_percent=pnl_percent,
            var_95=var_95,
            var_99=var_99,
            max_drawdown=worst_case,
            volatility=np.std(simulation_results) * np.sqrt(252),
            position_impacts=position_impacts,
            sector_impacts=sector_impacts,
            factor_contributions={
                'volatility': volatility_multiplier * 0.6,
                'correlation': correlation_increase * 0.4
            },
            timestamp=datetime.now(),
            metadata={
                'scenario_type': scenario.scenario_type.value,
                'n_simulations': n_simulations,
                'time_horizon': time_horizon
            }
        )
        
        self.results_history.append(result)
        return result
    
    def _run_custom_scenario(
        self,
        portfolio_data: Dict[str, Any],
        scenario: StressScenario
    ) -> StressResult:
        """Run custom scenario stress test."""
        params = scenario.parameters
        market_shock = params.get('market_shock', -0.2)
        volatility_multiplier = params.get('volatility_multiplier', 1.5)
        correlation_increase = params.get('correlation_increase', 0.2)
        liquidity_impact = params.get('liquidity_impact', 0.01)
        
        base_value = portfolio_data.get('total_value', 1000000)
        
        # Apply shocks to each position
        position_impacts = {}
        total_impact = 0
        
        for symbol, position in portfolio_data.get('positions', {}).items():
            position_value = position.get('value', 0)
            weight = position.get('weight', 0)
            
            # Base market shock
            market_impact = position_value * market_shock
            
            # Liquidity impact (higher for larger positions)
            liquidity_cost = position_value * liquidity_impact * (weight * 10)  # Scale by weight
            
            # Volatility impact
            volatility = self._get_position_volatility(symbol, portfolio_data)
            volatility_impact = position_value * volatility * (volatility_multiplier - 1)
            
            total_position_impact = market_impact - liquidity_cost - volatility_impact
            position_impacts[symbol] = total_position_impact
            total_impact += total_position_impact
        
        # Calculate sector impacts
        sector_impacts = self._calculate_sector_impacts(
            portfolio_data, market_shock, volatility_multiplier
        )
        
        # Calculate stressed metrics
        stressed_value = base_value + total_impact
        pnl = total_impact
        pnl_percent = (pnl / base_value) * 100 if base_value > 0 else 0
        
        # Generate stressed returns for risk metrics
        stressed_returns = self._generate_stressed_returns(
            portfolio_data, market_shock, volatility_multiplier
        )
        
        var_95 = np.percentile(stressed_returns, 5) * base_value
        var_99 = np.percentile(stressed_returns, 1) * base_value
        max_drawdown = np.min(np.cumsum(stressed_returns))
        volatility = np.std(stressed_returns) * np.sqrt(252)
        
        result = StressResult(
            scenario_name=scenario.name,
            portfolio_id=portfolio_data.get('portfolio_id', 'unknown'),
            base_value=base_value,
            stressed_value=stressed_value,
            pnl=pnl,
            pnl_percent=pnl_percent,
            var_95=var_95,
            var_99=var_99,
            max_drawdown=max_drawdown,
            volatility=volatility,
            position_impacts=position_impacts,
            sector_impacts=sector_impacts,
            factor_contributions={
                'market': market_shock * 0.6,
                'liquidity': -liquidity_impact * 0.2,
                'volatility': volatility_multiplier * 0.2
            },
            timestamp=datetime.now(),
            metadata={'scenario_type': scenario.scenario_type.value}
        )
        
        self.results_history.append(result)
        return result
    
    def _run_factor_shock_scenario(
        self,
        portfolio_data: Dict[str, Any],
        scenario: StressScenario
    ) -> StressResult:
        """Run factor shock stress test."""
        params = scenario.parameters
        base_value = portfolio_data.get('total_value', 1000000)
        
        # Apply factor-specific shocks
        position_impacts = {}
        sector_impacts = {}
        total_impact = 0
        
        if 'rate_change' in params:
            # Interest rate shock
            rate_change = params['rate_change']
            duration_impact = params.get('duration_impact', -0.05)
            
            for symbol, position in portfolio_data.get('positions', {}).items():
                position_value = position.get('value', 0)
                # Simplified duration impact
                duration = self._get_position_duration(symbol, portfolio_data)
                rate_impact = position_value * duration * rate_change * duration_impact
                position_impacts[symbol] = rate_impact
                total_impact += rate_impact
        
        # Sector-specific impacts
        if 'sector_impacts' in params:
            sector_shocks = params['sector_impacts']
            sector_exposures = portfolio_data.get('sector_exposures', {})
            
            for sector, exposure in sector_exposures.items():
                if sector in sector_shocks:
                    sector_shock = sector_shocks[sector]
                    sector_impact = base_value * exposure * sector_shock
                    sector_impacts[sector] = sector_impact
                    total_impact += sector_impact
        
        # Calculate final results
        stressed_value = base_value + total_impact
        pnl = total_impact
        pnl_percent = (pnl / base_value) * 100 if base_value > 0 else 0
        
        # Simplified risk metrics for factor shocks
        var_95 = total_impact * 1.2  # Approximate
        var_99 = total_impact * 1.5
        max_drawdown = min(0, pnl_percent / 100)
        volatility = abs(pnl_percent) / 100 * 2  # Rough estimate
        
        result = StressResult(
            scenario_name=scenario.name,
            portfolio_id=portfolio_data.get('portfolio_id', 'unknown'),
            base_value=base_value,
            stressed_value=stressed_value,
            pnl=pnl,
            pnl_percent=pnl_percent,
            var_95=var_95,
            var_99=var_99,
            max_drawdown=max_drawdown,
            volatility=volatility,
            position_impacts=position_impacts,
            sector_impacts=sector_impacts,
            factor_contributions=params.copy(),
            timestamp=datetime.now(),
            metadata={'scenario_type': scenario.scenario_type.value}
        )
        
        self.results_history.append(result)
        return result
    
    def run_all_scenarios(
        self,
        portfolio_data: Dict[str, Any],
        market_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, StressResult]:
        """Run all enabled scenarios."""
        results = {}
        
        for scenario_name, scenario in self.scenarios.items():
            if scenario.enabled:
                try:
                    result = self.run_stress_test(portfolio_data, scenario_name, market_data)
                    results[scenario_name] = result
                except Exception as e:
                    print(f"Error running scenario {scenario_name}: {e}")
        
        return results
    
    def _generate_synthetic_market_data(self, scenario: StressScenario) -> pd.DataFrame:
        """Generate synthetic market data for historical scenarios."""
        params = scenario.parameters
        start_date = pd.to_datetime(params.get('start_date', '2020-01-01'))
        end_date = pd.to_datetime(params.get('end_date', '2020-12-31'))
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        n_days = len(date_range)
        
        # Generate synthetic returns
        np.random.seed(42)
        returns = np.random.normal(-0.001, 0.02, n_days)  # Slightly negative mean
        
        # Apply scenario-specific adjustments
        market_shock = params.get('market_shock', -0.2)
        volatility_multiplier = params.get('volatility_multiplier', 1.5)
        
        # Add shock events
        shock_days = np.random.choice(n_days, size=max(1, n_days // 20), replace=False)
        returns[shock_days] *= volatility_multiplier
        returns[shock_days] += market_shock / len(shock_days)
        
        return pd.DataFrame({
            'date': date_range,
            'returns': returns,
            'cumulative_returns': np.cumsum(returns)
        })
    
    def _generate_stressed_returns(
        self,
        portfolio_data: Dict[str, Any],
        market_shock: float,
        volatility_multiplier: float,
        n_days: int = 252
    ) -> np.ndarray:
        """Generate stressed return series."""
        base_volatility = portfolio_data.get('volatility', 0.15)
        stressed_volatility = base_volatility * volatility_multiplier
        
        np.random.seed(42)
        returns = np.random.normal(
            market_shock / n_days,  # Spread shock over time
            stressed_volatility / np.sqrt(252),
            n_days
        )
        
        return returns
    
    def _calculate_sector_impacts(
        self,
        portfolio_data: Dict[str, Any],
        market_shock: float,
        volatility_multiplier: float
    ) -> Dict[str, float]:
        """Calculate sector-specific impacts."""
        sector_exposures = portfolio_data.get('sector_exposures', {})
        base_value = portfolio_data.get('total_value', 1000000)
        
        # Sector-specific shock multipliers
        sector_multipliers = {
            'Technology': 1.2,
            'Financials': 1.5,
            'Healthcare': 0.8,
            'Consumer': 1.0,
            'Energy': 1.8,
            'Utilities': 0.6,
            'Real Estate': 1.4
        }
        
        sector_impacts = {}
        for sector, exposure in sector_exposures.items():
            multiplier = sector_multipliers.get(sector, 1.0)
            sector_shock = market_shock * multiplier * volatility_multiplier
            sector_impact = base_value * exposure * sector_shock
            sector_impacts[sector] = sector_impact
        
        return sector_impacts
    
    def _get_position_beta(self, symbol: str, portfolio_data: Dict[str, Any]) -> float:
        """Get position beta (simplified)."""
        # In practice, this would look up actual beta values
        betas = {
            'AAPL': 1.2,
            'GOOGL': 1.1,
            'MSFT': 0.9,
            'TSLA': 2.0,
            'AMZN': 1.3,
            'JPM': 1.5,
            'JNJ': 0.7
        }
        return betas.get(symbol, 1.0)
    
    def _get_position_volatility(self, symbol: str, portfolio_data: Dict[str, Any]) -> float:
        """Get position volatility (simplified)."""
        # In practice, this would look up actual volatility values
        volatilities = {
            'AAPL': 0.25,
            'GOOGL': 0.28,
            'MSFT': 0.22,
            'TSLA': 0.45,
            'AMZN': 0.30,
            'JPM': 0.35,
            'JNJ': 0.18
        }
        return volatilities.get(symbol, 0.25)
    
    def _get_position_duration(self, symbol: str, portfolio_data: Dict[str, Any]) -> float:
        """Get position duration for interest rate sensitivity (simplified)."""
        # In practice, this would look up actual duration values
        durations = {
            'AAPL': 2.0,
            'GOOGL': 3.0,
            'MSFT': 2.5,
            'TSLA': 1.5,
            'AMZN': 4.0,
            'JPM': 0.5,  # Banks benefit from rising rates
            'JNJ': 3.5
        }
        return durations.get(symbol, 2.0)
    
    def get_scenario_summary(self) -> pd.DataFrame:
        """Get summary of all scenarios."""
        if not self.scenarios:
            return pd.DataFrame()
        
        data = []
        for scenario in self.scenarios.values():
            data.append({
                'name': scenario.name,
                'type': scenario.scenario_type.value,
                'description': scenario.description,
                'enabled': scenario.enabled,
                'created_date': scenario.created_date
            })
        
        return pd.DataFrame(data)
    
    def get_results_summary(
        self,
        portfolio_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get summary of stress test results."""
        results = self.results_history
        
        # Filter results
        if portfolio_id:
            results = [r for r in results if r.portfolio_id == portfolio_id]
        if start_date:
            results = [r for r in results if r.timestamp >= start_date]
        if end_date:
            results = [r for r in results if r.timestamp <= end_date]
        
        if not results:
            return pd.DataFrame()
        
        data = []
        for result in results:
            data.append({
                'scenario_name': result.scenario_name,
                'portfolio_id': result.portfolio_id,
                'pnl': result.pnl,
                'pnl_percent': result.pnl_percent,
                'var_95': result.var_95,
                'var_99': result.var_99,
                'max_drawdown': result.max_drawdown,
                'volatility': result.volatility,
                'timestamp': result.timestamp
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('timestamp', ascending=False)
    
    def export_results(
        self,
        filename: str,
        portfolio_id: Optional[str] = None,
        format: str = 'csv'
    ) -> None:
        """Export stress test results to file."""
        df = self.get_results_summary(portfolio_id=portfolio_id)
        
        if format.lower() == 'csv':
            df.to_csv(filename, index=False)
        elif format.lower() == 'excel':
            df.to_excel(filename, index=False)
        else:
            raise ValidationError(f"Unsupported export format: {format}")
    
    def clear_results_history(self, older_than_days: Optional[int] = None) -> int:
        """Clear results history."""
        if older_than_days is None:
            count = len(self.results_history)
            self.results_history.clear()
            return count
        
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        original_count = len(self.results_history)
        self.results_history = [
            r for r in self.results_history if r.timestamp >= cutoff_date
        ]
        return original_count - len(self.results_history)