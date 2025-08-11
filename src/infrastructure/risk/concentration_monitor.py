"""
Concentration risk monitoring and management system.

This module provides comprehensive concentration risk analysis including:
- Position concentration monitoring
- Sector and geographic concentration analysis
- Issuer and counterparty concentration tracking
- Concentration limit enforcement
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


class ConcentrationType(Enum):
    """Types of concentration risk."""
    POSITION = "position"
    SECTOR = "sector"
    GEOGRAPHY = "geography"
    ISSUER = "issuer"
    COUNTERPARTY = "counterparty"
    CURRENCY = "currency"
    ASSET_CLASS = "asset_class"
    MARKET_CAP = "market_cap"


class ConcentrationSeverity(Enum):
    """Concentration risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConcentrationLimits:
    """Concentration limits configuration."""
    max_single_position: float = 0.05  # 5%
    max_sector_exposure: float = 0.20   # 20%
    max_geography_exposure: float = 0.30  # 30%
    max_issuer_exposure: float = 0.10   # 10%
    max_currency_exposure: float = 0.25  # 25%
    max_top_5_positions: float = 0.40   # 40%
    max_top_10_positions: float = 0.60  # 60%
    herfindahl_threshold: float = 0.15  # HHI threshold
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'max_single_position': self.max_single_position,
            'max_sector_exposure': self.max_sector_exposure,
            'max_geography_exposure': self.max_geography_exposure,
            'max_issuer_exposure': self.max_issuer_exposure,
            'max_currency_exposure': self.max_currency_exposure,
            'max_top_5_positions': self.max_top_5_positions,
            'max_top_10_positions': self.max_top_10_positions,
            'herfindahl_threshold': self.herfindahl_threshold
        }


@dataclass
class ConcentrationAlert:
    """Concentration risk alert."""
    alert_id: str
    portfolio_id: str
    concentration_type: ConcentrationType
    severity: ConcentrationSeverity
    entity: str  # Position, sector, etc.
    current_exposure: float
    limit: float
    excess_exposure: float
    message: str
    timestamp: datetime
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'portfolio_id': self.portfolio_id,
            'concentration_type': self.concentration_type.value,
            'severity': self.severity.value,
            'entity': self.entity,
            'current_exposure': self.current_exposure,
            'limit': self.limit,
            'excess_exposure': self.excess_exposure,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }


@dataclass
class ConcentrationMetrics:
    """Concentration risk metrics."""
    portfolio_id: str
    herfindahl_index: float
    effective_positions: float
    top_5_concentration: float
    top_10_concentration: float
    max_position_weight: float
    position_concentration: Dict[str, float]
    sector_concentration: Dict[str, float]
    geography_concentration: Dict[str, float]
    currency_concentration: Dict[str, float]
    concentration_score: float
    timestamp: datetime
    
    def get_diversification_ratio(self) -> float:
        """Calculate diversification ratio (1/HHI)."""
        return 1.0 / max(self.herfindahl_index, 0.001)


class ConcentrationMonitor:
    """
    Comprehensive concentration risk monitoring system.
    
    This class provides monitoring and analysis of various types of
    concentration risk in portfolios.
    """
    
    def __init__(
        self,
        limits: Optional[ConcentrationLimits] = None,
        monitoring_frequency: int = 3600  # seconds
    ):
        """
        Initialize concentration monitor.
        
        Args:
            limits: Concentration limits configuration
            monitoring_frequency: Monitoring frequency in seconds
        """
        self.limits = limits or ConcentrationLimits()
        self.monitoring_frequency = monitoring_frequency
        
        # Alert tracking
        self.active_alerts: Dict[str, List[ConcentrationAlert]] = {}
        self.alert_history: List[ConcentrationAlert] = []
        
        # Asset metadata cache
        self.asset_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Initialize asset metadata
        self._initialize_asset_metadata()
    
    def _initialize_asset_metadata(self) -> None:
        """Initialize asset metadata for concentration analysis."""
        # In practice, this would come from external data sources
        # For demonstration, create sample metadata
        
        self.asset_metadata = {
            'AAPL': {
                'sector': 'Technology',
                'geography': 'United States',
                'currency': 'USD',
                'market_cap': 'Large',
                'issuer': 'Apple Inc.',
                'asset_class': 'Equity'
            },
            'GOOGL': {
                'sector': 'Technology',
                'geography': 'United States',
                'currency': 'USD',
                'market_cap': 'Large',
                'issuer': 'Alphabet Inc.',
                'asset_class': 'Equity'
            },
            'MSFT': {
                'sector': 'Technology',
                'geography': 'United States',
                'currency': 'USD',
                'market_cap': 'Large',
                'issuer': 'Microsoft Corp.',
                'asset_class': 'Equity'
            },
            'TSLA': {
                'sector': 'Consumer Discretionary',
                'geography': 'United States',
                'currency': 'USD',
                'market_cap': 'Large',
                'issuer': 'Tesla Inc.',
                'asset_class': 'Equity'
            },
            'AMZN': {
                'sector': 'Consumer Discretionary',
                'geography': 'United States',
                'currency': 'USD',
                'market_cap': 'Large',
                'issuer': 'Amazon.com Inc.',
                'asset_class': 'Equity'
            },
            'JPM': {
                'sector': 'Financials',
                'geography': 'United States',
                'currency': 'USD',
                'market_cap': 'Large',
                'issuer': 'JPMorgan Chase & Co.',
                'asset_class': 'Equity'
            },
            'JNJ': {
                'sector': 'Healthcare',
                'geography': 'United States',
                'currency': 'USD',
                'market_cap': 'Large',
                'issuer': 'Johnson & Johnson',
                'asset_class': 'Equity'
            }
        }
    
    def add_asset_metadata(
        self,
        symbol: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Add or update asset metadata."""
        required_fields = ['sector', 'geography', 'currency', 'issuer']
        for field in required_fields:
            if field not in metadata:
                warnings.warn(f"Missing required field '{field}' for asset {symbol}")
        
        self.asset_metadata[symbol] = metadata
    
    def calculate_concentration_metrics(
        self,
        portfolio_data: Dict[str, Any]
    ) -> ConcentrationMetrics:
        """Calculate comprehensive concentration metrics."""
        positions = portfolio_data.get('positions', {})
        portfolio_id = portfolio_data.get('portfolio_id', 'unknown')
        
        if not positions:
            return self._create_default_concentration_metrics(portfolio_id)
        
        # Extract position weights
        weights = {symbol: pos.get('weight', 0) for symbol, pos in positions.items()}
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            return self._create_default_concentration_metrics(portfolio_id)
        
        # Normalize weights
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate basic concentration metrics
        herfindahl_index = sum(w**2 for w in normalized_weights.values())
        effective_positions = 1.0 / herfindahl_index if herfindahl_index > 0 else 0
        max_position_weight = max(normalized_weights.values()) if normalized_weights else 0
        
        # Top N concentration
        sorted_weights = sorted(normalized_weights.values(), reverse=True)
        top_5_concentration = sum(sorted_weights[:5])
        top_10_concentration = sum(sorted_weights[:10])
        
        # Sector concentration
        sector_concentration = self._calculate_sector_concentration(normalized_weights)
        
        # Geography concentration
        geography_concentration = self._calculate_geography_concentration(normalized_weights)
        
        # Currency concentration
        currency_concentration = self._calculate_currency_concentration(normalized_weights)
        
        # Overall concentration score (0-100, lower is better)
        concentration_score = self._calculate_concentration_score(
            herfindahl_index, max_position_weight, sector_concentration
        )
        
        return ConcentrationMetrics(
            portfolio_id=portfolio_id,
            herfindahl_index=herfindahl_index,
            effective_positions=effective_positions,
            top_5_concentration=top_5_concentration,
            top_10_concentration=top_10_concentration,
            max_position_weight=max_position_weight,
            position_concentration=normalized_weights,
            sector_concentration=sector_concentration,
            geography_concentration=geography_concentration,
            currency_concentration=currency_concentration,
            concentration_score=concentration_score,
            timestamp=datetime.now()
        )
    
    def _create_default_concentration_metrics(self, portfolio_id: str) -> ConcentrationMetrics:
        """Create default concentration metrics for empty portfolio."""
        return ConcentrationMetrics(
            portfolio_id=portfolio_id,
            herfindahl_index=0.0,
            effective_positions=0.0,
            top_5_concentration=0.0,
            top_10_concentration=0.0,
            max_position_weight=0.0,
            position_concentration={},
            sector_concentration={},
            geography_concentration={},
            currency_concentration={},
            concentration_score=0.0,
            timestamp=datetime.now()
        )
    
    def _calculate_sector_concentration(
        self,
        position_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate sector concentration."""
        sector_weights = {}
        
        for symbol, weight in position_weights.items():
            metadata = self.asset_metadata.get(symbol, {})
            sector = metadata.get('sector', 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        return sector_weights
    
    def _calculate_geography_concentration(
        self,
        position_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate geographic concentration."""
        geography_weights = {}
        
        for symbol, weight in position_weights.items():
            metadata = self.asset_metadata.get(symbol, {})
            geography = metadata.get('geography', 'Unknown')
            geography_weights[geography] = geography_weights.get(geography, 0) + weight
        
        return geography_weights
    
    def _calculate_currency_concentration(
        self,
        position_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate currency concentration."""
        currency_weights = {}
        
        for symbol, weight in position_weights.items():
            metadata = self.asset_metadata.get(symbol, {})
            currency = metadata.get('currency', 'Unknown')
            currency_weights[currency] = currency_weights.get(currency, 0) + weight
        
        return currency_weights
    
    def _calculate_concentration_score(
        self,
        herfindahl_index: float,
        max_position_weight: float,
        sector_concentration: Dict[str, float]
    ) -> float:
        """Calculate overall concentration score."""
        # HHI component (0-40 points)
        hhi_score = min(herfindahl_index * 200, 40)
        
        # Max position component (0-30 points)
        position_score = min(max_position_weight * 300, 30)
        
        # Sector concentration component (0-30 points)
        max_sector_weight = max(sector_concentration.values()) if sector_concentration else 0
        sector_score = min(max_sector_weight * 150, 30)
        
        return hhi_score + position_score + sector_score
    
    def check_concentration_limits(
        self,
        portfolio_data: Dict[str, Any]
    ) -> List[ConcentrationAlert]:
        """Check portfolio against concentration limits."""
        metrics = self.calculate_concentration_metrics(portfolio_data)
        alerts = []
        portfolio_id = portfolio_data.get('portfolio_id', 'unknown')
        
        # Check position concentration
        for symbol, weight in metrics.position_concentration.items():
            if weight > self.limits.max_single_position:
                alert = ConcentrationAlert(
                    alert_id=f"{portfolio_id}_position_{symbol}_{int(datetime.now().timestamp())}",
                    portfolio_id=portfolio_id,
                    concentration_type=ConcentrationType.POSITION,
                    severity=self._determine_severity(weight, self.limits.max_single_position),
                    entity=symbol,
                    current_exposure=weight,
                    limit=self.limits.max_single_position,
                    excess_exposure=weight - self.limits.max_single_position,
                    message=f"Position {symbol} exceeds limit: {weight:.2%} > {self.limits.max_single_position:.2%}",
                    timestamp=datetime.now()
                )
                alerts.append(alert)
        
        # Check sector concentration
        for sector, weight in metrics.sector_concentration.items():
            if weight > self.limits.max_sector_exposure:
                alert = ConcentrationAlert(
                    alert_id=f"{portfolio_id}_sector_{sector}_{int(datetime.now().timestamp())}",
                    portfolio_id=portfolio_id,
                    concentration_type=ConcentrationType.SECTOR,
                    severity=self._determine_severity(weight, self.limits.max_sector_exposure),
                    entity=sector,
                    current_exposure=weight,
                    limit=self.limits.max_sector_exposure,
                    excess_exposure=weight - self.limits.max_sector_exposure,
                    message=f"Sector {sector} exceeds limit: {weight:.2%} > {self.limits.max_sector_exposure:.2%}",
                    timestamp=datetime.now()
                )
                alerts.append(alert)
        
        # Check geography concentration
        for geography, weight in metrics.geography_concentration.items():
            if weight > self.limits.max_geography_exposure:
                alert = ConcentrationAlert(
                    alert_id=f"{portfolio_id}_geography_{geography}_{int(datetime.now().timestamp())}",
                    portfolio_id=portfolio_id,
                    concentration_type=ConcentrationType.GEOGRAPHY,
                    severity=self._determine_severity(weight, self.limits.max_geography_exposure),
                    entity=geography,
                    current_exposure=weight,
                    limit=self.limits.max_geography_exposure,
                    excess_exposure=weight - self.limits.max_geography_exposure,
                    message=f"Geography {geography} exceeds limit: {weight:.2%} > {self.limits.max_geography_exposure:.2%}",
                    timestamp=datetime.now()
                )
                alerts.append(alert)
        
        # Check currency concentration
        for currency, weight in metrics.currency_concentration.items():
            if weight > self.limits.max_currency_exposure:
                alert = ConcentrationAlert(
                    alert_id=f"{portfolio_id}_currency_{currency}_{int(datetime.now().timestamp())}",
                    portfolio_id=portfolio_id,
                    concentration_type=ConcentrationType.CURRENCY,
                    severity=self._determine_severity(weight, self.limits.max_currency_exposure),
                    entity=currency,
                    current_exposure=weight,
                    limit=self.limits.max_currency_exposure,
                    excess_exposure=weight - self.limits.max_currency_exposure,
                    message=f"Currency {currency} exceeds limit: {weight:.2%} > {self.limits.max_currency_exposure:.2%}",
                    timestamp=datetime.now()
                )
                alerts.append(alert)
        
        # Check top N concentration
        if metrics.top_5_concentration > self.limits.max_top_5_positions:
            alert = ConcentrationAlert(
                alert_id=f"{portfolio_id}_top5_{int(datetime.now().timestamp())}",
                portfolio_id=portfolio_id,
                concentration_type=ConcentrationType.POSITION,
                severity=self._determine_severity(metrics.top_5_concentration, self.limits.max_top_5_positions),
                entity="Top 5 Positions",
                current_exposure=metrics.top_5_concentration,
                limit=self.limits.max_top_5_positions,
                excess_exposure=metrics.top_5_concentration - self.limits.max_top_5_positions,
                message=f"Top 5 positions exceed limit: {metrics.top_5_concentration:.2%} > {self.limits.max_top_5_positions:.2%}",
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        if metrics.top_10_concentration > self.limits.max_top_10_positions:
            alert = ConcentrationAlert(
                alert_id=f"{portfolio_id}_top10_{int(datetime.now().timestamp())}",
                portfolio_id=portfolio_id,
                concentration_type=ConcentrationType.POSITION,
                severity=self._determine_severity(metrics.top_10_concentration, self.limits.max_top_10_positions),
                entity="Top 10 Positions",
                current_exposure=metrics.top_10_concentration,
                limit=self.limits.max_top_10_positions,
                excess_exposure=metrics.top_10_concentration - self.limits.max_top_10_positions,
                message=f"Top 10 positions exceed limit: {metrics.top_10_concentration:.2%} > {self.limits.max_top_10_positions:.2%}",
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        # Check Herfindahl Index
        if metrics.herfindahl_index > self.limits.herfindahl_threshold:
            alert = ConcentrationAlert(
                alert_id=f"{portfolio_id}_hhi_{int(datetime.now().timestamp())}",
                portfolio_id=portfolio_id,
                concentration_type=ConcentrationType.POSITION,
                severity=self._determine_severity(metrics.herfindahl_index, self.limits.herfindahl_threshold),
                entity="Portfolio HHI",
                current_exposure=metrics.herfindahl_index,
                limit=self.limits.herfindahl_threshold,
                excess_exposure=metrics.herfindahl_index - self.limits.herfindahl_threshold,
                message=f"Herfindahl Index exceeds threshold: {metrics.herfindahl_index:.3f} > {self.limits.herfindahl_threshold:.3f}",
                timestamp=datetime.now()
            )
            alerts.append(alert)
        
        # Store alerts
        if alerts:
            if portfolio_id not in self.active_alerts:
                self.active_alerts[portfolio_id] = []
            self.active_alerts[portfolio_id].extend(alerts)
            self.alert_history.extend(alerts)
        
        return alerts
    
    def _determine_severity(self, current_value: float, limit: float) -> ConcentrationSeverity:
        """Determine alert severity based on limit breach."""
        excess_ratio = (current_value - limit) / limit if limit > 0 else 0
        
        if excess_ratio <= 0.1:  # Up to 10% over limit
            return ConcentrationSeverity.LOW
        elif excess_ratio <= 0.25:  # Up to 25% over limit
            return ConcentrationSeverity.MEDIUM
        elif excess_ratio <= 0.5:   # Up to 50% over limit
            return ConcentrationSeverity.HIGH
        else:
            return ConcentrationSeverity.CRITICAL
    
    def suggest_rebalancing(
        self,
        portfolio_data: Dict[str, Any],
        target_concentration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Suggest rebalancing to reduce concentration risk."""
        metrics = self.calculate_concentration_metrics(portfolio_data)
        alerts = self.check_concentration_limits(portfolio_data)
        
        if not alerts:
            return {'status': 'no_rebalancing_needed', 'current_metrics': metrics}
        
        suggestions = []
        
        # Position-level suggestions
        for alert in alerts:
            if alert.concentration_type == ConcentrationType.POSITION:
                if alert.entity not in ['Top 5 Positions', 'Top 10 Positions', 'Portfolio HHI']:
                    target_weight = alert.limit * 0.9  # 10% buffer below limit
                    reduction_needed = alert.current_exposure - target_weight
                    
                    suggestions.append({
                        'type': 'reduce_position',
                        'symbol': alert.entity,
                        'current_weight': alert.current_exposure,
                        'target_weight': target_weight,
                        'reduction_needed': reduction_needed,
                        'priority': alert.severity.value
                    })
        
        # Sector-level suggestions
        sector_alerts = [a for a in alerts if a.concentration_type == ConcentrationType.SECTOR]
        for alert in sector_alerts:
            target_weight = alert.limit * 0.9
            reduction_needed = alert.current_exposure - target_weight
            
            # Find positions in this sector
            sector_positions = []
            for symbol, weight in metrics.position_concentration.items():
                metadata = self.asset_metadata.get(symbol, {})
                if metadata.get('sector') == alert.entity:
                    sector_positions.append((symbol, weight))
            
            # Sort by weight (largest first)
            sector_positions.sort(key=lambda x: x[1], reverse=True)
            
            suggestions.append({
                'type': 'reduce_sector',
                'sector': alert.entity,
                'current_weight': alert.current_exposure,
                'target_weight': target_weight,
                'reduction_needed': reduction_needed,
                'affected_positions': sector_positions,
                'priority': alert.severity.value
            })
        
        # Calculate diversification opportunities
        diversification_suggestions = self._suggest_diversification(metrics)
        
        return {
            'status': 'rebalancing_recommended',
            'current_metrics': metrics,
            'alerts': [alert.to_dict() for alert in alerts],
            'rebalancing_suggestions': suggestions,
            'diversification_opportunities': diversification_suggestions,
            'estimated_improvement': self._estimate_rebalancing_impact(suggestions, metrics)
        }
    
    def _suggest_diversification(
        self,
        metrics: ConcentrationMetrics
    ) -> List[Dict[str, Any]]:
        """Suggest diversification opportunities."""
        suggestions = []
        
        # Sector diversification
        sector_weights = list(metrics.sector_concentration.values())
        if sector_weights:
            sector_hhi = sum(w**2 for w in sector_weights)
            if sector_hhi > 0.25:  # High sector concentration
                suggestions.append({
                    'type': 'sector_diversification',
                    'message': 'Consider diversifying across more sectors',
                    'current_sector_hhi': sector_hhi,
                    'underrepresented_sectors': self._identify_underrepresented_sectors(metrics)
                })
        
        # Geographic diversification
        geo_weights = list(metrics.geography_concentration.values())
        if geo_weights:
            geo_hhi = sum(w**2 for w in geo_weights)
            if geo_hhi > 0.5:  # High geographic concentration
                suggestions.append({
                    'type': 'geographic_diversification',
                    'message': 'Consider international diversification',
                    'current_geographic_hhi': geo_hhi
                })
        
        # Position count suggestion
        if metrics.effective_positions < 10:
            suggestions.append({
                'type': 'increase_positions',
                'message': f'Consider increasing number of positions (current effective: {metrics.effective_positions:.1f})',
                'target_positions': 15
            })
        
        return suggestions
    
    def _identify_underrepresented_sectors(
        self,
        metrics: ConcentrationMetrics
    ) -> List[str]:
        """Identify sectors that are underrepresented in the portfolio."""
        # Common sectors that might be underrepresented
        common_sectors = [
            'Technology', 'Healthcare', 'Financials', 'Consumer Discretionary',
            'Consumer Staples', 'Industrials', 'Energy', 'Utilities',
            'Real Estate', 'Materials', 'Communication Services'
        ]
        
        current_sectors = set(metrics.sector_concentration.keys())
        underrepresented = []
        
        for sector in common_sectors:
            if sector not in current_sectors:
                underrepresented.append(sector)
            elif metrics.sector_concentration.get(sector, 0) < 0.05:  # Less than 5%
                underrepresented.append(sector)
        
        return underrepresented[:5]  # Return top 5 suggestions
    
    def _estimate_rebalancing_impact(
        self,
        suggestions: List[Dict[str, Any]],
        current_metrics: ConcentrationMetrics
    ) -> Dict[str, float]:
        """Estimate the impact of rebalancing suggestions."""
        # Simplified impact estimation
        total_reduction = sum(
            s.get('reduction_needed', 0) for s in suggestions
            if s['type'] == 'reduce_position'
        )
        
        # Estimate new HHI after rebalancing
        estimated_hhi_reduction = total_reduction * 0.5  # Rough estimate
        new_hhi = max(0.02, current_metrics.herfindahl_index - estimated_hhi_reduction)
        
        # Estimate new concentration score
        new_score = current_metrics.concentration_score * (new_hhi / current_metrics.herfindahl_index)
        
        return {
            'estimated_hhi_improvement': current_metrics.herfindahl_index - new_hhi,
            'estimated_score_improvement': current_metrics.concentration_score - new_score,
            'estimated_effective_positions': 1.0 / new_hhi if new_hhi > 0 else 0
        }
    
    def get_concentration_dashboard(
        self,
        portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get concentration risk dashboard data."""
        metrics = self.calculate_concentration_metrics(portfolio_data)
        alerts = self.check_concentration_limits(portfolio_data)
        
        # Risk level assessment
        risk_level = "Low"
        if metrics.concentration_score > 60:
            risk_level = "Critical"
        elif metrics.concentration_score > 40:
            risk_level = "High"
        elif metrics.concentration_score > 20:
            risk_level = "Medium"
        
        # Top concentrations
        top_positions = sorted(
            metrics.position_concentration.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        top_sectors = sorted(
            metrics.sector_concentration.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'portfolio_id': metrics.portfolio_id,
            'timestamp': metrics.timestamp,
            'risk_level': risk_level,
            'concentration_score': metrics.concentration_score,
            'herfindahl_index': metrics.herfindahl_index,
            'effective_positions': metrics.effective_positions,
            'diversification_ratio': metrics.get_diversification_ratio(),
            'active_alerts': len(alerts),
            'critical_alerts': len([a for a in alerts if a.severity == ConcentrationSeverity.CRITICAL]),
            'top_positions': top_positions,
            'top_sectors': top_sectors,
            'limit_utilization': self._calculate_limit_utilization(metrics),
            'recommendations': len(self.suggest_rebalancing(portfolio_data)['rebalancing_suggestions'])
        }
    
    def _calculate_limit_utilization(
        self,
        metrics: ConcentrationMetrics
    ) -> Dict[str, float]:
        """Calculate utilization of concentration limits."""
        return {
            'max_position': (metrics.max_position_weight / self.limits.max_single_position) * 100,
            'top_5_positions': (metrics.top_5_concentration / self.limits.max_top_5_positions) * 100,
            'top_10_positions': (metrics.top_10_concentration / self.limits.max_top_10_positions) * 100,
            'herfindahl_index': (metrics.herfindahl_index / self.limits.herfindahl_threshold) * 100,
            'max_sector': max(
                (weight / self.limits.max_sector_exposure) * 100
                for weight in metrics.sector_concentration.values()
            ) if metrics.sector_concentration else 0
        }
    
    def export_concentration_report(
        self,
        portfolio_data: Dict[str, Any],
        filename: str,
        format: str = 'excel'
    ) -> None:
        """Export concentration analysis report."""
        metrics = self.calculate_concentration_metrics(portfolio_data)
        dashboard = self.get_concentration_dashboard(portfolio_data)
        rebalancing = self.suggest_rebalancing(portfolio_data)
        
        if format.lower() == 'excel':
            with pd.ExcelWriter(filename) as writer:
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Concentration Score',
                        'Herfindahl Index',
                        'Effective Positions',
                        'Max Position Weight',
                        'Top 5 Concentration',
                        'Top 10 Concentration'
                    ],
                    'Value': [
                        metrics.concentration_score,
                        metrics.herfindahl_index,
                        metrics.effective_positions,
                        metrics.max_position_weight,
                        metrics.top_5_concentration,
                        metrics.top_10_concentration
                    ],
                    'Status': [
                        dashboard['risk_level'],
                        'High' if metrics.herfindahl_index > self.limits.herfindahl_threshold else 'OK',
                        'Low' if metrics.effective_positions < 10 else 'OK',
                        'High' if metrics.max_position_weight > self.limits.max_single_position else 'OK',
                        'High' if metrics.top_5_concentration > self.limits.max_top_5_positions else 'OK',
                        'High' if metrics.top_10_concentration > self.limits.max_top_10_positions else 'OK'
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Position concentration
                position_data = [
                    {'Symbol': symbol, 'Weight': weight}
                    for symbol, weight in metrics.position_concentration.items()
                ]
                pd.DataFrame(position_data).to_excel(writer, sheet_name='Positions', index=False)
                
                # Sector concentration
                sector_data = [
                    {'Sector': sector, 'Weight': weight}
                    for sector, weight in metrics.sector_concentration.items()
                ]
                pd.DataFrame(sector_data).to_excel(writer, sheet_name='Sectors', index=False)
                
                # Alerts
                if 'alerts' in rebalancing:
                    alerts_data = rebalancing['alerts']
                    pd.DataFrame(alerts_data).to_excel(writer, sheet_name='Alerts', index=False)
        
        elif format.lower() == 'csv':
            # Export position concentration as CSV
            position_data = [
                {'Symbol': symbol, 'Weight': weight, 'Sector': self.asset_metadata.get(symbol, {}).get('sector', 'Unknown')}
                for symbol, weight in metrics.position_concentration.items()
            ]
            pd.DataFrame(position_data).to_csv(filename, index=False)
        
        else:
            raise ValidationError(f"Unsupported export format: {format}")
    
    def get_historical_concentration(
        self,
        portfolio_history: List[Dict[str, Any]],
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """Calculate historical concentration metrics."""
        historical_data = []
        
        for portfolio_snapshot in portfolio_history[-lookback_days:]:
            metrics = self.calculate_concentration_metrics(portfolio_snapshot)
            historical_data.append({
                'date': portfolio_snapshot.get('date', datetime.now()),
                'herfindahl_index': metrics.herfindahl_index,
                'effective_positions': metrics.effective_positions,
                'concentration_score': metrics.concentration_score,
                'max_position_weight': metrics.max_position_weight,
                'top_5_concentration': metrics.top_5_concentration
            })
        
        return pd.DataFrame(historical_data)
    
    def clear_alerts(self, portfolio_id: Optional[str] = None) -> int:
        """Clear concentration alerts."""
        if portfolio_id:
            if portfolio_id in self.active_alerts:
                count = len(self.active_alerts[portfolio_id])
                del self.active_alerts[portfolio_id]
                return count
            return 0
        else:
            total_count = sum(len(alerts) for alerts in self.active_alerts.values())
            self.active_alerts.clear()
            return total_count