"""
Liquidity risk assessment and monitoring system.

This module provides comprehensive liquidity risk analysis including:
- Liquidity metrics calculation
- Market impact estimation
- Liquidity stress testing
- Time-to-liquidation analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass, field
import warnings

from ...domain.entities import Portfolio, Position
from ...domain.exceptions import ValidationError


class LiquidityTier(Enum):
    """Liquidity tiers for assets."""
    TIER_1 = "tier_1"  # Highly liquid (large cap stocks, major ETFs)
    TIER_2 = "tier_2"  # Moderately liquid (mid cap stocks)
    TIER_3 = "tier_3"  # Less liquid (small cap stocks, corporate bonds)
    TIER_4 = "tier_4"  # Illiquid (private equity, real estate)


class LiquidityRisk(Enum):
    """Liquidity risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for an asset or portfolio."""
    asset_id: str
    bid_ask_spread: float
    average_daily_volume: float
    volume_weighted_spread: float
    market_impact_cost: float
    time_to_liquidate_days: float
    liquidity_tier: LiquidityTier
    liquidity_risk: LiquidityRisk
    amihud_illiquidity: float
    roll_measure: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'asset_id': self.asset_id,
            'bid_ask_spread': self.bid_ask_spread,
            'average_daily_volume': self.average_daily_volume,
            'volume_weighted_spread': self.volume_weighted_spread,
            'market_impact_cost': self.market_impact_cost,
            'time_to_liquidate_days': self.time_to_liquidate_days,
            'liquidity_tier': self.liquidity_tier.value,
            'liquidity_risk': self.liquidity_risk.value,
            'amihud_illiquidity': self.amihud_illiquidity,
            'roll_measure': self.roll_measure,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PortfolioLiquidityProfile:
    """Portfolio-level liquidity profile."""
    portfolio_id: str
    total_value: float
    liquid_assets_percent: float
    weighted_avg_spread: float
    weighted_time_to_liquidate: float
    liquidity_concentration: float
    stress_liquidation_cost: float
    tier_distribution: Dict[str, float]
    largest_position_liquidity_days: float
    timestamp: datetime
    
    def get_liquidity_score(self) -> float:
        """Calculate overall liquidity score (0-100, higher is more liquid)."""
        # Weighted combination of liquidity factors
        spread_score = max(0, 100 - (self.weighted_avg_spread * 10000))  # Convert to bps
        time_score = max(0, 100 - (self.weighted_time_to_liquidate * 10))
        concentration_score = max(0, 100 - (self.liquidity_concentration * 100))
        
        return (spread_score * 0.4 + time_score * 0.4 + concentration_score * 0.2)


class LiquidityAnalyzer:
    """
    Comprehensive liquidity risk analyzer.
    
    This class provides various liquidity risk assessment methods including
    market impact estimation, time-to-liquidation analysis, and stress testing.
    """
    
    def __init__(
        self,
        lookback_days: int = 60,
        confidence_level: float = 0.95,
        max_participation_rate: float = 0.20
    ):
        """
        Initialize liquidity analyzer.
        
        Args:
            lookback_days: Days of historical data for analysis
            confidence_level: Confidence level for risk calculations
            max_participation_rate: Maximum participation rate in daily volume
        """
        self.lookback_days = lookback_days
        self.confidence_level = confidence_level
        self.max_participation_rate = max_participation_rate
        
        # Liquidity data cache
        self.liquidity_cache: Dict[str, LiquidityMetrics] = {}
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        
        # Liquidity tier thresholds
        self.tier_thresholds = {
            'daily_volume_usd': {
                LiquidityTier.TIER_1: 100_000_000,  # $100M+
                LiquidityTier.TIER_2: 10_000_000,   # $10M+
                LiquidityTier.TIER_3: 1_000_000,    # $1M+
                LiquidityTier.TIER_4: 0             # Below $1M
            },
            'bid_ask_spread': {
                LiquidityTier.TIER_1: 0.001,  # 10 bps or less
                LiquidityTier.TIER_2: 0.005,  # 50 bps or less
                LiquidityTier.TIER_3: 0.020,  # 200 bps or less
                LiquidityTier.TIER_4: 1.000   # Above 200 bps
            }
        }
    
    def calculate_asset_liquidity(
        self,
        asset_id: str,
        market_data: Optional[pd.DataFrame] = None,
        position_size: Optional[float] = None
    ) -> LiquidityMetrics:
        """
        Calculate comprehensive liquidity metrics for an asset.
        
        Args:
            asset_id: Asset identifier
            market_data: Historical market data (OHLCV)
            position_size: Position size for market impact calculation
            
        Returns:
            LiquidityMetrics object with all liquidity measures
        """
        # Check cache first
        if asset_id in self.liquidity_cache:
            cached_metrics = self.liquidity_cache[asset_id]
            if (datetime.now() - cached_metrics.timestamp).seconds < 3600:  # 1 hour cache
                return cached_metrics
        
        # Get or generate market data
        if market_data is None:
            market_data = self._get_market_data(asset_id)
        
        if market_data.empty:
            return self._create_default_liquidity_metrics(asset_id)
        
        # Calculate individual liquidity measures
        bid_ask_spread = self._calculate_bid_ask_spread(market_data)
        avg_daily_volume = self._calculate_average_daily_volume(market_data)
        volume_weighted_spread = self._calculate_volume_weighted_spread(market_data)
        amihud_illiquidity = self._calculate_amihud_illiquidity(market_data)
        roll_measure = self._calculate_roll_measure(market_data)
        
        # Calculate market impact cost
        market_impact_cost = self._calculate_market_impact_cost(
            market_data, position_size or avg_daily_volume * 0.1
        )
        
        # Calculate time to liquidate
        time_to_liquidate = self._calculate_time_to_liquidate(
            position_size or avg_daily_volume * 0.1, avg_daily_volume
        )
        
        # Determine liquidity tier and risk level
        liquidity_tier = self._determine_liquidity_tier(avg_daily_volume, bid_ask_spread)
        liquidity_risk = self._assess_liquidity_risk(
            bid_ask_spread, time_to_liquidate, market_impact_cost
        )
        
        metrics = LiquidityMetrics(
            asset_id=asset_id,
            bid_ask_spread=bid_ask_spread,
            average_daily_volume=avg_daily_volume,
            volume_weighted_spread=volume_weighted_spread,
            market_impact_cost=market_impact_cost,
            time_to_liquidate_days=time_to_liquidate,
            liquidity_tier=liquidity_tier,
            liquidity_risk=liquidity_risk,
            amihud_illiquidity=amihud_illiquidity,
            roll_measure=roll_measure,
            timestamp=datetime.now()
        )
        
        # Cache the results
        self.liquidity_cache[asset_id] = metrics
        
        return metrics
    
    def _get_market_data(self, asset_id: str) -> pd.DataFrame:
        """Get market data for asset (placeholder implementation)."""
        # In practice, this would fetch real market data
        # For demonstration, generate synthetic data
        
        if asset_id in self.market_data_cache:
            return self.market_data_cache[asset_id]
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=self.lookback_days),
            end=datetime.now(),
            freq='D'
        )
        
        np.random.seed(hash(asset_id) % 2**32)
        
        # Generate synthetic market data
        n_days = len(dates)
        base_price = 100
        
        # Price data
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Volume data (varies by asset type)
        volume_multipliers = {
            'AAPL': 50_000_000,
            'GOOGL': 20_000_000,
            'MSFT': 30_000_000,
            'TSLA': 25_000_000,
            'AMZN': 15_000_000
        }
        
        base_volume = volume_multipliers.get(asset_id, 1_000_000)
        volumes = np.random.lognormal(
            np.log(base_volume), 0.5, n_days
        )
        
        # Bid-ask spread (varies by liquidity)
        spread_multipliers = {
            'AAPL': 0.0005,
            'GOOGL': 0.0008,
            'MSFT': 0.0006,
            'TSLA': 0.0015,
            'AMZN': 0.0010
        }
        
        base_spread = spread_multipliers.get(asset_id, 0.002)
        spreads = np.random.exponential(base_spread, n_days)
        
        market_data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, n_days)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
            'close': prices,
            'volume': volumes,
            'bid_ask_spread': spreads,
            'dollar_volume': prices * volumes
        })
        
        self.market_data_cache[asset_id] = market_data
        return market_data
    
    def _create_default_liquidity_metrics(self, asset_id: str) -> LiquidityMetrics:
        """Create default liquidity metrics when no data is available."""
        return LiquidityMetrics(
            asset_id=asset_id,
            bid_ask_spread=0.01,  # 1% default spread
            average_daily_volume=1_000_000,
            volume_weighted_spread=0.01,
            market_impact_cost=0.02,
            time_to_liquidate_days=5.0,
            liquidity_tier=LiquidityTier.TIER_3,
            liquidity_risk=LiquidityRisk.MEDIUM,
            amihud_illiquidity=0.001,
            roll_measure=0.005,
            timestamp=datetime.now()
        )
    
    def _calculate_bid_ask_spread(self, market_data: pd.DataFrame) -> float:
        """Calculate average bid-ask spread."""
        if 'bid_ask_spread' in market_data.columns:
            return market_data['bid_ask_spread'].mean()
        else:
            # Estimate from high-low spread
            hl_spread = (market_data['high'] - market_data['low']) / market_data['close']
            return hl_spread.mean() * 0.5  # Rough approximation
    
    def _calculate_average_daily_volume(self, market_data: pd.DataFrame) -> float:
        """Calculate average daily dollar volume."""
        if 'dollar_volume' in market_data.columns:
            return market_data['dollar_volume'].mean()
        else:
            return (market_data['volume'] * market_data['close']).mean()
    
    def _calculate_volume_weighted_spread(self, market_data: pd.DataFrame) -> float:
        """Calculate volume-weighted average spread."""
        if 'bid_ask_spread' in market_data.columns and 'volume' in market_data.columns:
            total_volume = market_data['volume'].sum()
            if total_volume > 0:
                weighted_spread = (
                    market_data['bid_ask_spread'] * market_data['volume']
                ).sum() / total_volume
                return weighted_spread
        
        return self._calculate_bid_ask_spread(market_data)
    
    def _calculate_amihud_illiquidity(self, market_data: pd.DataFrame) -> float:
        """Calculate Amihud illiquidity measure."""
        # Amihud = |Return| / Dollar Volume
        returns = market_data['close'].pct_change().abs()
        dollar_volume = market_data.get('dollar_volume', 
                                       market_data['volume'] * market_data['close'])
        
        # Avoid division by zero
        valid_data = (dollar_volume > 0) & (returns.notna())
        if valid_data.sum() == 0:
            return 0.001
        
        amihud = (returns[valid_data] / dollar_volume[valid_data]).mean()
        return amihud * 1_000_000  # Scale for readability
    
    def _calculate_roll_measure(self, market_data: pd.DataFrame) -> float:
        """Calculate Roll's measure of effective spread."""
        returns = market_data['close'].pct_change()
        
        if len(returns) < 2:
            return 0.005
        
        # Roll's measure: 2 * sqrt(-Cov(r_t, r_{t-1}))
        covariance = returns.autocorr(lag=1)
        if covariance < 0:
            roll_measure = 2 * np.sqrt(-covariance)
        else:
            roll_measure = 0.005  # Default if positive autocorrelation
        
        return min(roll_measure, 0.05)  # Cap at 5%
    
    def _calculate_market_impact_cost(
        self,
        market_data: pd.DataFrame,
        position_size: float
    ) -> float:
        """Calculate estimated market impact cost."""
        avg_daily_volume = self._calculate_average_daily_volume(market_data)
        
        if avg_daily_volume <= 0:
            return 0.05  # 5% default impact
        
        # Participation rate
        participation_rate = min(position_size / avg_daily_volume, 1.0)
        
        # Square root market impact model
        # Impact = α * (participation_rate)^β * volatility
        alpha = 0.1  # Market impact coefficient
        beta = 0.5   # Square root law
        
        volatility = market_data['close'].pct_change().std()
        market_impact = alpha * (participation_rate ** beta) * volatility
        
        return min(market_impact, 0.10)  # Cap at 10%
    
    def _calculate_time_to_liquidate(
        self,
        position_size: float,
        avg_daily_volume: float
    ) -> float:
        """Calculate estimated time to liquidate position."""
        if avg_daily_volume <= 0:
            return 30.0  # 30 days default
        
        # Assume we can trade at most max_participation_rate of daily volume
        daily_liquidation = avg_daily_volume * self.max_participation_rate
        
        if daily_liquidation <= 0:
            return 30.0
        
        days_to_liquidate = position_size / daily_liquidation
        return min(days_to_liquidate, 90.0)  # Cap at 90 days
    
    def _determine_liquidity_tier(
        self,
        avg_daily_volume: float,
        bid_ask_spread: float
    ) -> LiquidityTier:
        """Determine liquidity tier based on volume and spread."""
        volume_thresholds = self.tier_thresholds['daily_volume_usd']
        spread_thresholds = self.tier_thresholds['bid_ask_spread']
        
        # Determine tier based on volume
        if avg_daily_volume >= volume_thresholds[LiquidityTier.TIER_1]:
            volume_tier = LiquidityTier.TIER_1
        elif avg_daily_volume >= volume_thresholds[LiquidityTier.TIER_2]:
            volume_tier = LiquidityTier.TIER_2
        elif avg_daily_volume >= volume_thresholds[LiquidityTier.TIER_3]:
            volume_tier = LiquidityTier.TIER_3
        else:
            volume_tier = LiquidityTier.TIER_4
        
        # Determine tier based on spread
        if bid_ask_spread <= spread_thresholds[LiquidityTier.TIER_1]:
            spread_tier = LiquidityTier.TIER_1
        elif bid_ask_spread <= spread_thresholds[LiquidityTier.TIER_2]:
            spread_tier = LiquidityTier.TIER_2
        elif bid_ask_spread <= spread_thresholds[LiquidityTier.TIER_3]:
            spread_tier = LiquidityTier.TIER_3
        else:
            spread_tier = LiquidityTier.TIER_4
        
        # Take the worse (higher number) tier
        tier_values = {
            LiquidityTier.TIER_1: 1,
            LiquidityTier.TIER_2: 2,
            LiquidityTier.TIER_3: 3,
            LiquidityTier.TIER_4: 4
        }
        
        worse_tier_value = max(tier_values[volume_tier], tier_values[spread_tier])
        
        for tier, value in tier_values.items():
            if value == worse_tier_value:
                return tier
        
        return LiquidityTier.TIER_3  # Default
    
    def _assess_liquidity_risk(
        self,
        bid_ask_spread: float,
        time_to_liquidate: float,
        market_impact_cost: float
    ) -> LiquidityRisk:
        """Assess overall liquidity risk level."""
        # Risk scoring based on multiple factors
        spread_score = min(bid_ask_spread * 1000, 10)  # Convert to bps, cap at 10
        time_score = min(time_to_liquidate / 10, 10)   # Normalize, cap at 10
        impact_score = min(market_impact_cost * 100, 10)  # Convert to %, cap at 10
        
        total_score = spread_score + time_score + impact_score
        
        if total_score <= 3:
            return LiquidityRisk.LOW
        elif total_score <= 6:
            return LiquidityRisk.MEDIUM
        elif total_score <= 15:
            return LiquidityRisk.HIGH
        else:
            return LiquidityRisk.CRITICAL
    
    def calculate_portfolio_liquidity(
        self,
        portfolio_data: Dict[str, Any]
    ) -> PortfolioLiquidityProfile:
        """Calculate portfolio-level liquidity profile."""
        positions = portfolio_data.get('positions', {})
        total_value = portfolio_data.get('total_value', 1_000_000)
        
        if not positions:
            return self._create_default_portfolio_profile(
                portfolio_data.get('portfolio_id', 'unknown'), total_value
            )
        
        # Calculate liquidity metrics for each position
        position_liquidity = {}
        for symbol, position in positions.items():
            position_size = position.get('value', 0)
            liquidity_metrics = self.calculate_asset_liquidity(symbol, position_size=position_size)
            position_liquidity[symbol] = liquidity_metrics
        
        # Calculate portfolio-level metrics
        total_weight = sum(pos.get('weight', 0) for pos in positions.values())
        
        # Weighted average spread
        weighted_spread = sum(
            position_liquidity[symbol].bid_ask_spread * positions[symbol].get('weight', 0)
            for symbol in positions.keys()
        ) / max(total_weight, 0.01)
        
        # Weighted time to liquidate
        weighted_time = sum(
            position_liquidity[symbol].time_to_liquidate_days * positions[symbol].get('weight', 0)
            for symbol in positions.keys()
        ) / max(total_weight, 0.01)
        
        # Liquidity concentration (Herfindahl index of liquidity-adjusted weights)
        liquidity_weights = {}
        for symbol in positions.keys():
            weight = positions[symbol].get('weight', 0)
            liquidity_adjustment = 1 / (1 + position_liquidity[symbol].time_to_liquidate_days)
            liquidity_weights[symbol] = weight * liquidity_adjustment
        
        total_liquidity_weight = sum(liquidity_weights.values())
        if total_liquidity_weight > 0:
            normalized_weights = {k: v/total_liquidity_weight for k, v in liquidity_weights.items()}
            concentration = sum(w**2 for w in normalized_weights.values())
        else:
            concentration = 1.0
        
        # Tier distribution
        tier_distribution = {tier.value: 0.0 for tier in LiquidityTier}
        for symbol in positions.keys():
            weight = positions[symbol].get('weight', 0)
            tier = position_liquidity[symbol].liquidity_tier
            tier_distribution[tier.value] += weight
        
        # Liquid assets percentage (Tier 1 and 2)
        liquid_percent = tier_distribution[LiquidityTier.TIER_1.value] + \
                        tier_distribution[LiquidityTier.TIER_2.value]
        
        # Stress liquidation cost (worst-case scenario)
        stress_cost = sum(
            position_liquidity[symbol].market_impact_cost * positions[symbol].get('weight', 0) * 2
            for symbol in positions.keys()
        )  # Double the normal market impact for stress scenario
        
        # Largest position liquidity
        largest_position_symbol = max(positions.keys(), 
                                    key=lambda x: positions[x].get('weight', 0))
        largest_position_liquidity = position_liquidity[largest_position_symbol].time_to_liquidate_days
        
        return PortfolioLiquidityProfile(
            portfolio_id=portfolio_data.get('portfolio_id', 'unknown'),
            total_value=total_value,
            liquid_assets_percent=liquid_percent,
            weighted_avg_spread=weighted_spread,
            weighted_time_to_liquidate=weighted_time,
            liquidity_concentration=concentration,
            stress_liquidation_cost=stress_cost,
            tier_distribution=tier_distribution,
            largest_position_liquidity_days=largest_position_liquidity,
            timestamp=datetime.now()
        )
    
    def _create_default_portfolio_profile(
        self,
        portfolio_id: str,
        total_value: float
    ) -> PortfolioLiquidityProfile:
        """Create default portfolio liquidity profile."""
        return PortfolioLiquidityProfile(
            portfolio_id=portfolio_id,
            total_value=total_value,
            liquid_assets_percent=0.5,
            weighted_avg_spread=0.01,
            weighted_time_to_liquidate=5.0,
            liquidity_concentration=0.3,
            stress_liquidation_cost=0.05,
            tier_distribution={
                LiquidityTier.TIER_1.value: 0.3,
                LiquidityTier.TIER_2.value: 0.2,
                LiquidityTier.TIER_3.value: 0.3,
                LiquidityTier.TIER_4.value: 0.2
            },
            largest_position_liquidity_days=3.0,
            timestamp=datetime.now()
        )
    
    def stress_test_liquidity(
        self,
        portfolio_data: Dict[str, Any],
        stress_scenarios: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Perform liquidity stress testing."""
        if stress_scenarios is None:
            stress_scenarios = {
                'market_stress': 2.0,      # 2x normal market impact
                'volume_reduction': 0.5,   # 50% volume reduction
                'spread_widening': 3.0     # 3x normal spreads
            }
        
        base_profile = self.calculate_portfolio_liquidity(portfolio_data)
        stress_results = {}
        
        for scenario_name, stress_factor in stress_scenarios.items():
            if scenario_name == 'market_stress':
                # Increase market impact costs
                stressed_cost = base_profile.stress_liquidation_cost * stress_factor
                stress_results[scenario_name] = {
                    'liquidation_cost': stressed_cost,
                    'cost_increase': stressed_cost - base_profile.stress_liquidation_cost
                }
            
            elif scenario_name == 'volume_reduction':
                # Increase time to liquidate
                stressed_time = base_profile.weighted_time_to_liquidate / stress_factor
                stress_results[scenario_name] = {
                    'time_to_liquidate': stressed_time,
                    'time_increase': stressed_time - base_profile.weighted_time_to_liquidate
                }
            
            elif scenario_name == 'spread_widening':
                # Increase transaction costs
                stressed_spread = base_profile.weighted_avg_spread * stress_factor
                stress_results[scenario_name] = {
                    'spread_cost': stressed_spread,
                    'spread_increase': stressed_spread - base_profile.weighted_avg_spread
                }
        
        return {
            'base_profile': base_profile,
            'stress_scenarios': stress_results,
            'overall_stress_impact': sum(
                result.get('cost_increase', 0) + 
                result.get('time_increase', 0) * 0.01 +  # Convert time to cost
                result.get('spread_increase', 0)
                for result in stress_results.values()
            )
        }
    
    def get_liquidity_report(
        self,
        portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive liquidity report."""
        portfolio_profile = self.calculate_portfolio_liquidity(portfolio_data)
        stress_results = self.stress_test_liquidity(portfolio_data)
        
        # Position-level analysis
        positions = portfolio_data.get('positions', {})
        position_analysis = {}
        
        for symbol, position in positions.items():
            liquidity_metrics = self.calculate_asset_liquidity(symbol)
            position_analysis[symbol] = {
                'weight': position.get('weight', 0),
                'value': position.get('value', 0),
                'liquidity_tier': liquidity_metrics.liquidity_tier.value,
                'liquidity_risk': liquidity_metrics.liquidity_risk.value,
                'time_to_liquidate': liquidity_metrics.time_to_liquidate_days,
                'market_impact_cost': liquidity_metrics.market_impact_cost,
                'bid_ask_spread': liquidity_metrics.bid_ask_spread
            }
        
        # Risk alerts
        alerts = []
        if portfolio_profile.liquid_assets_percent < 0.3:
            alerts.append("Low liquid assets percentage (<30%)")
        if portfolio_profile.weighted_time_to_liquidate > 10:
            alerts.append("High time to liquidate (>10 days)")
        if portfolio_profile.liquidity_concentration > 0.5:
            alerts.append("High liquidity concentration risk")
        if portfolio_profile.stress_liquidation_cost > 0.10:
            alerts.append("High stress liquidation cost (>10%)")
        
        return {
            'portfolio_profile': portfolio_profile,
            'position_analysis': position_analysis,
            'stress_testing': stress_results,
            'liquidity_score': portfolio_profile.get_liquidity_score(),
            'risk_alerts': alerts,
            'recommendations': self._generate_liquidity_recommendations(portfolio_profile),
            'timestamp': datetime.now()
        }
    
    def _generate_liquidity_recommendations(
        self,
        profile: PortfolioLiquidityProfile
    ) -> List[str]:
        """Generate liquidity improvement recommendations."""
        recommendations = []
        
        if profile.liquid_assets_percent < 0.4:
            recommendations.append(
                "Consider increasing allocation to highly liquid assets (Tier 1 & 2)"
            )
        
        if profile.weighted_time_to_liquidate > 7:
            recommendations.append(
                "Reduce positions in illiquid assets to improve portfolio liquidity"
            )
        
        if profile.liquidity_concentration > 0.4:
            recommendations.append(
                "Diversify across assets with different liquidity profiles"
            )
        
        if profile.stress_liquidation_cost > 0.08:
            recommendations.append(
                "Consider liquidity buffers for stress scenarios"
            )
        
        if profile.largest_position_liquidity_days > 10:
            recommendations.append(
                "Reduce size of largest illiquid position"
            )
        
        return recommendations
    
    def export_liquidity_data(
        self,
        portfolio_data: Dict[str, Any],
        filename: str,
        format: str = 'csv'
    ) -> None:
        """Export liquidity analysis to file."""
        report = self.get_liquidity_report(portfolio_data)
        
        # Create DataFrame for export
        position_data = []
        for symbol, analysis in report['position_analysis'].items():
            position_data.append({
                'symbol': symbol,
                **analysis
            })
        
        df = pd.DataFrame(position_data)
        
        if format.lower() == 'csv':
            df.to_csv(filename, index=False)
        elif format.lower() == 'excel':
            with pd.ExcelWriter(filename) as writer:
                df.to_excel(writer, sheet_name='Position_Analysis', index=False)
                
                # Add portfolio summary
                summary_data = {
                    'Metric': [
                        'Liquidity Score',
                        'Liquid Assets %',
                        'Weighted Avg Spread',
                        'Weighted Time to Liquidate',
                        'Stress Liquidation Cost'
                    ],
                    'Value': [
                        report['liquidity_score'],
                        report['portfolio_profile'].liquid_assets_percent,
                        report['portfolio_profile'].weighted_avg_spread,
                        report['portfolio_profile'].weighted_time_to_liquidate,
                        report['portfolio_profile'].stress_liquidation_cost
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Portfolio_Summary', index=False)
        else:
            raise ValidationError(f"Unsupported export format: {format}")
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.liquidity_cache.clear()
        self.market_data_cache.clear()