#!/usr/bin/env python3
"""
Comprehensive Constraint Management System Demo

This demo showcases the enhanced constraint management system including:
- Position limits with dynamic adjustment
- Sector constraints with correlation adjustment
- Turnover limits with transaction cost integration
- ESG constraints with sustainable investing filters
- Dynamic constraint adjustment based on market conditions
- Constraint conflict resolution and reporting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from src.infrastructure.optimization import (
    ConstraintManager, PositionLimitConstraint, SectorConstraint,
    TurnoverConstraint, ESGConstraint, ConstraintType, ConstraintSeverity,
    MarketCondition, PortfolioOptimizer
)


def create_sample_data():
    """Create sample data for demonstration."""
    # Asset universe
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'BAC', 'XOM', 'JNJ', 'PG']
    
    # Expected returns
    np.random.seed(42)
    expected_returns = pd.Series(
        np.random.normal(0.10, 0.05, len(assets)),
        index=assets
    )
    
    # Covariance matrix
    correlation = np.random.uniform(0.1, 0.7, (len(assets), len(assets)))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)
    
    volatilities = np.random.uniform(0.15, 0.35, len(assets))
    covariance_matrix = pd.DataFrame(
        np.outer(volatilities, volatilities) * correlation,
        index=assets,
        columns=assets
    )
    
    # Sector mapping
    sector_mapping = {
        'AAPL': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
        'AMZN': 'Technology', 'TSLA': 'Technology',
        'JPM': 'Financial', 'BAC': 'Financial',
        'XOM': 'Energy',
        'JNJ': 'Healthcare', 'PG': 'Consumer'
    }
    
    # ESG scores (0-100)
    esg_scores = {
        'AAPL': 85, 'GOOGL': 80, 'MSFT': 90, 'AMZN': 75, 'TSLA': 70,
        'JPM': 65, 'BAC': 60, 'XOM': 30, 'JNJ': 85, 'PG': 80
    }
    
    # Environmental scores
    environmental_scores = {
        'AAPL': 80, 'GOOGL': 85, 'MSFT': 95, 'AMZN': 70, 'TSLA': 95,
        'JPM': 60, 'BAC': 55, 'XOM': 20, 'JNJ': 80, 'PG': 85
    }
    
    # Carbon intensities (tons CO2/million revenue)
    carbon_intensities = {
        'AAPL': 10, 'GOOGL': 8, 'MSFT': 5, 'AMZN': 15, 'TSLA': 12,
        'JPM': 25, 'BAC': 30, 'XOM': 150, 'JNJ': 20, 'PG': 35
    }
    
    # Current portfolio weights (for turnover constraint)
    current_weights = pd.Series(
        [0.15, 0.12, 0.13, 0.10, 0.08, 0.12, 0.08, 0.05, 0.10, 0.07],
        index=assets
    )
    
    return {
        'assets': assets,
        'expected_returns': expected_returns,
        'covariance_matrix': covariance_matrix,
        'sector_mapping': sector_mapping,
        'esg_scores': esg_scores,
        'environmental_scores': environmental_scores,
        'carbon_intensities': carbon_intensities,
        'current_weights': current_weights
    }


def demo_basic_constraints():
    """Demonstrate basic constraint functionality."""
    print("=" * 60)
    print("BASIC CONSTRAINT FUNCTIONALITY DEMO")
    print("=" * 60)
    
    data = create_sample_data()
    
    # Create constraint manager
    manager = ConstraintManager()
    
    # 1. Position Limit Constraint
    print("\n1. Position Limit Constraint")
    print("-" * 30)
    
    position_constraint = PositionLimitConstraint(
        name="Basic Position Limits",
        description="Limit individual position sizes to 5-25%",
        constraint_type=ConstraintType.POSITION_LIMIT,
        min_weight=0.05,
        max_weight=0.25,
        severity=ConstraintSeverity.HARD
    )
    
    manager.add_constraint(position_constraint)
    
    # Test with sample weights
    test_weights = np.array([0.30, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.02, 0.02, 0.01])
    
    validation = manager.validate_all_constraints(test_weights, asset_names=data['assets'])
    print(f"Validation result: {validation['is_valid']}")
    if validation['violations']:
        for violation in validation['violations']:
            print(f"  Violation: {violation.message}")
    
    # 2. Sector Constraint
    print("\n2. Sector Constraint")
    print("-" * 30)
    
    sector_constraint = SectorConstraint(
        name="Sector Exposure Limits",
        description="Limit sector exposures",
        constraint_type=ConstraintType.SECTOR_LIMIT,
        sector_mapping=data['sector_mapping'],
        sector_limits={
            'Technology': (0.20, 0.50),
            'Financial': (0.10, 0.30),
            'Energy': (0.00, 0.10),
            'Healthcare': (0.05, 0.20),
            'Consumer': (0.05, 0.15)
        },
        severity=ConstraintSeverity.HARD
    )
    
    manager.add_constraint(sector_constraint)
    
    # Check sector exposures
    exposures = sector_constraint.get_sector_exposures(test_weights, data['assets'])
    print("Current sector exposures:")
    for sector, exposure in exposures.items():
        print(f"  {sector}: {exposure:.1%}")
    
    # 3. ESG Constraint
    print("\n3. ESG Constraint")
    print("-" * 30)
    
    esg_constraint = ESGConstraint(
        name="ESG and Sustainability",
        description="ESG scoring and carbon footprint limits",
        constraint_type=ConstraintType.ESG_CONSTRAINT,
        esg_scores=data['esg_scores'],
        min_portfolio_score=70,
        environmental_scores=data['environmental_scores'],
        min_environmental_score=75,
        carbon_intensities=data['carbon_intensities'],
        max_portfolio_carbon_intensity=30,
        exclude_assets=['XOM'],  # Exclude oil company
        severity=ConstraintSeverity.HARD
    )
    
    manager.add_constraint(esg_constraint)
    
    # Calculate ESG metrics
    esg_metrics = esg_constraint.calculate_esg_metrics(test_weights, data['assets'])
    print("ESG metrics:")
    for metric, value in esg_metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    # Final validation with all constraints
    print("\n4. Combined Constraint Validation")
    print("-" * 30)
    
    final_validation = manager.validate_all_constraints(test_weights, asset_names=data['assets'])
    print(f"All constraints satisfied: {final_validation['is_valid']}")
    print(f"Total violations: {final_validation['num_violations']}")
    
    if final_validation['violations']:
        print("Violations:")
        for violation in final_validation['violations']:
            print(f"  - {violation.constraint_name}: {violation.message}")


def demo_dynamic_constraints():
    """Demonstrate dynamic constraint adjustment based on market conditions."""
    print("\n" + "=" * 60)
    print("DYNAMIC CONSTRAINT ADJUSTMENT DEMO")
    print("=" * 60)
    
    data = create_sample_data()
    
    # Create constraint manager
    manager = ConstraintManager()
    
    # Create dynamic constraints
    dynamic_position_constraint = PositionLimitConstraint(
        name="Dynamic Position Limits",
        description="Market-sensitive position limits",
        constraint_type=ConstraintType.POSITION_LIMIT,
        min_weight=0.05,
        max_weight=0.30,
        is_dynamic=True,
        market_condition_sensitivity=0.7,
        severity=ConstraintSeverity.HARD
    )
    
    dynamic_sector_constraint = SectorConstraint(
        name="Dynamic Sector Limits",
        description="Market-sensitive sector limits",
        constraint_type=ConstraintType.SECTOR_LIMIT,
        sector_mapping=data['sector_mapping'],
        sector_limits={
            'Technology': (0.20, 0.60),
            'Financial': (0.10, 0.40),
            'Energy': (0.00, 0.15),
            'Healthcare': (0.05, 0.25),
            'Consumer': (0.05, 0.20)
        },
        is_dynamic=True,
        market_condition_sensitivity=0.5,
        severity=ConstraintSeverity.HARD
    )
    
    manager.add_constraint(dynamic_position_constraint)
    manager.add_constraint(dynamic_sector_constraint)
    
    # Test different market conditions
    market_conditions = [
        MarketCondition(
            volatility=0.15,
            liquidity=0.9,
            market_stress=0.2,
            correlation_regime='low'
        ),
        MarketCondition(
            volatility=0.35,
            liquidity=0.6,
            market_stress=0.7,
            correlation_regime='medium'
        ),
        MarketCondition(
            volatility=0.50,
            liquidity=0.3,
            market_stress=0.9,
            correlation_regime='high'
        )
    ]
    
    condition_names = ['Normal Market', 'Stressed Market', 'Crisis Market']
    
    for i, (condition, name) in enumerate(zip(market_conditions, condition_names)):
        print(f"\n{i+1}. {name}")
        print("-" * 30)
        print(f"Volatility: {condition.volatility:.1%}")
        print(f"Liquidity: {condition.liquidity:.1%}")
        print(f"Market Stress: {condition.market_stress:.1%}")
        print(f"Correlation Regime: {condition.correlation_regime}")
        
        # Update market conditions
        manager.update_market_condition(condition)
        
        # Get adjusted constraints
        active_constraints = manager.get_active_constraints()
        
        print("\nAdjusted constraint limits:")
        for constraint in active_constraints:
            if constraint.constraint_type == ConstraintType.POSITION_LIMIT:
                print(f"  Max position size: {constraint.max_weight:.1%}")
            elif constraint.constraint_type == ConstraintType.SECTOR_LIMIT:
                print(f"  Technology sector max: {constraint.sector_limits['Technology'][1]:.1%}")


def demo_turnover_and_transaction_costs():
    """Demonstrate turnover constraints with transaction cost integration."""
    print("\n" + "=" * 60)
    print("TURNOVER AND TRANSACTION COST DEMO")
    print("=" * 60)
    
    data = create_sample_data()
    
    # Simple transaction cost model
    def transaction_cost_model(trades):
        """Linear + quadratic transaction cost model."""
        linear_cost = 0.001 * np.sum(np.abs(trades))  # 10 bps linear cost
        market_impact = 0.0005 * np.sum(trades ** 2)  # Quadratic market impact
        return linear_cost + market_impact
    
    # Create turnover constraint with transaction costs
    turnover_constraint = TurnoverConstraint(
        name="Turnover and Cost Control",
        description="Limit turnover and transaction costs",
        constraint_type=ConstraintType.TURNOVER_LIMIT,
        max_turnover=0.30,
        current_weights=data['current_weights'].values,
        transaction_cost_model=transaction_cost_model,
        max_transaction_cost=0.015,  # 1.5% max transaction cost
        asset_specific_turnover_limits={
            'AAPL': 0.10,  # Limit AAPL turnover to 10%
            'TSLA': 0.08   # Limit TSLA turnover to 8% (more volatile)
        },
        severity=ConstraintSeverity.HARD
    )
    
    # Test different target portfolios
    target_portfolios = [
        np.array([0.16, 0.13, 0.14, 0.11, 0.09, 0.11, 0.07, 0.04, 0.09, 0.06]),  # Low turnover
        np.array([0.20, 0.20, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),  # Medium turnover
        np.array([0.30, 0.25, 0.20, 0.10, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01])   # High turnover
    ]
    
    portfolio_names = ['Conservative Rebalance', 'Moderate Rebalance', 'Aggressive Rebalance']
    
    for i, (target_weights, name) in enumerate(zip(target_portfolios, portfolio_names)):
        print(f"\n{i+1}. {name}")
        print("-" * 30)
        
        # Calculate turnover metrics
        metrics = turnover_constraint.calculate_turnover_metrics(target_weights)
        
        print(f"Total turnover: {metrics['total_turnover']:.1%}")
        print(f"Buy turnover: {metrics['buy_turnover']:.1%}")
        print(f"Sell turnover: {metrics['sell_turnover']:.1%}")
        print(f"Number of trades: {metrics['num_trades']}")
        print(f"Max individual trade: {metrics['max_individual_trade']:.1%}")
        
        # Calculate transaction costs
        trades = target_weights - data['current_weights'].values
        transaction_cost = transaction_cost_model(trades)
        print(f"Estimated transaction cost: {transaction_cost:.2%}")
        
        # Validate constraint
        is_valid = turnover_constraint.validate(target_weights, asset_names=data['assets'])
        print(f"Constraint satisfied: {is_valid}")
        
        if not is_valid:
            violation = turnover_constraint.check_violation(target_weights, asset_names=data['assets'])
            if violation:
                print(f"Violation: {violation.message}")


def demo_constraint_optimization():
    """Demonstrate portfolio optimization with comprehensive constraints."""
    print("\n" + "=" * 60)
    print("CONSTRAINT-AWARE PORTFOLIO OPTIMIZATION DEMO")
    print("=" * 60)
    
    data = create_sample_data()
    
    # Create comprehensive constraint set
    manager = ConstraintManager()
    
    # Position limits
    manager.add_constraint(PositionLimitConstraint(
        name="Position Limits",
        description="Individual position size limits",
        constraint_type=ConstraintType.POSITION_LIMIT,
        min_weight=0.02,
        max_weight=0.25,
        severity=ConstraintSeverity.HARD
    ))
    
    # Sector limits
    manager.add_constraint(SectorConstraint(
        name="Sector Limits",
        description="Sector exposure limits",
        constraint_type=ConstraintType.SECTOR_LIMIT,
        sector_mapping=data['sector_mapping'],
        sector_limits={
            'Technology': (0.25, 0.55),
            'Financial': (0.10, 0.30),
            'Energy': (0.00, 0.08),
            'Healthcare': (0.08, 0.20),
            'Consumer': (0.05, 0.15)
        },
        severity=ConstraintSeverity.HARD
    ))
    
    # ESG constraints
    manager.add_constraint(ESGConstraint(
        name="ESG Requirements",
        description="ESG and sustainability requirements",
        constraint_type=ConstraintType.ESG_CONSTRAINT,
        esg_scores=data['esg_scores'],
        min_portfolio_score=75,
        environmental_scores=data['environmental_scores'],
        min_environmental_score=80,
        carbon_intensities=data['carbon_intensities'],
        max_portfolio_carbon_intensity=25,
        exclude_assets=['XOM'],
        severity=ConstraintSeverity.HARD
    ))
    
    # Turnover limits
    manager.add_constraint(TurnoverConstraint(
        name="Turnover Control",
        description="Portfolio turnover limits",
        constraint_type=ConstraintType.TURNOVER_LIMIT,
        max_turnover=0.40,
        current_weights=data['current_weights'].values,
        severity=ConstraintSeverity.SOFT
    ))
    
    # Create optimizer
    optimizer = PortfolioOptimizer()
    
    # Prepare constraints for optimization
    test_weights = np.ones(len(data['assets'])) / len(data['assets'])  # Equal weights as starting point
    constraint_params = manager.apply_all_constraints(
        test_weights, 
        asset_names=data['assets']
    )
    
    print("Constraint Summary:")
    summary = manager.get_constraint_summary()
    print(f"  Total constraints: {summary['total_constraints']}")
    print(f"  Active constraints: {summary['active_constraints']}")
    print(f"  Constraint types: {list(summary['constraint_types'].keys())}")
    
    # Generate constraint report
    print("\nConstraint Report:")
    report = manager.create_constraint_report(test_weights, asset_names=data['assets'])
    
    print(f"  Validation status: {report['validation']['is_valid']}")
    print(f"  Number of violations: {report['validation']['num_violations']}")
    
    if report['validation']['violations']:
        print("  Violations:")
        for violation in report['validation']['violations'][:3]:  # Show first 3
            print(f"    - {violation.constraint_name}: {violation.message}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    # Demonstrate conflict detection
    conflicts = manager.resolve_constraint_conflicts()
    if conflicts:
        print("\nConstraint Conflicts:")
        for conflict in conflicts:
            print(f"  - {conflict}")
    else:
        print("\nNo constraint conflicts detected.")


def create_visualization():
    """Create visualizations of constraint effects."""
    print("\n" + "=" * 60)
    print("CONSTRAINT VISUALIZATION")
    print("=" * 60)
    
    data = create_sample_data()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Sector exposure limits visualization
    sectors = list(set(data['sector_mapping'].values()))
    sector_limits_min = [0.20, 0.10, 0.00, 0.05, 0.05]  # Tech, Financial, Energy, Healthcare, Consumer
    sector_limits_max = [0.50, 0.30, 0.10, 0.20, 0.15]
    
    x_pos = np.arange(len(sectors))
    ax1.bar(x_pos, sector_limits_max, alpha=0.7, label='Max Limit', color='lightcoral')
    ax1.bar(x_pos, sector_limits_min, alpha=0.7, label='Min Limit', color='lightblue')
    ax1.set_xlabel('Sectors')
    ax1.set_ylabel('Weight Limit')
    ax1.set_title('Sector Exposure Limits')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(sectors, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. ESG scores visualization
    assets_subset = data['assets'][:6]  # First 6 assets for clarity
    esg_values = [data['esg_scores'][asset] for asset in assets_subset]
    colors = ['green' if score >= 70 else 'orange' if score >= 50 else 'red' for score in esg_values]
    
    ax2.bar(assets_subset, esg_values, color=colors, alpha=0.7)
    ax2.axhline(y=70, color='red', linestyle='--', label='Min ESG Score (70)')
    ax2.set_xlabel('Assets')
    ax2.set_ylabel('ESG Score')
    ax2.set_title('ESG Scores by Asset')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Position limits with dynamic adjustment
    market_stress_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    base_max_weight = 0.25
    adjusted_weights = [base_max_weight * (1 - stress * 0.5) for stress in market_stress_levels]
    
    ax3.plot(market_stress_levels, adjusted_weights, 'bo-', linewidth=2, markersize=8)
    ax3.axhline(y=base_max_weight, color='red', linestyle='--', alpha=0.7, label='Base Limit')
    ax3.set_xlabel('Market Stress Level')
    ax3.set_ylabel('Max Position Weight')
    ax3.set_title('Dynamic Position Limit Adjustment')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Transaction cost vs turnover
    turnover_levels = np.linspace(0, 0.5, 20)
    linear_costs = 0.001 * turnover_levels
    quadratic_costs = 0.0005 * turnover_levels ** 2
    total_costs = linear_costs + quadratic_costs
    
    ax4.plot(turnover_levels, linear_costs * 100, label='Linear Cost', linestyle='--')
    ax4.plot(turnover_levels, quadratic_costs * 100, label='Market Impact', linestyle=':')
    ax4.plot(turnover_levels, total_costs * 100, label='Total Cost', linewidth=2)
    ax4.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Cost Limit (1.5%)')
    ax4.set_xlabel('Portfolio Turnover')
    ax4.set_ylabel('Transaction Cost (%)')
    ax4.set_title('Transaction Costs vs Turnover')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('constraint_management_demo.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'constraint_management_demo.png'")
    
    return fig


def main():
    """Run the comprehensive constraint management demo."""
    print("COMPREHENSIVE CONSTRAINT MANAGEMENT SYSTEM DEMO")
    print("=" * 60)
    print("This demo showcases the enhanced constraint management system")
    print("with dynamic adjustment, transaction cost integration, and")
    print("comprehensive ESG/sustainability constraints.")
    print()
    
    try:
        # Run all demos
        demo_basic_constraints()
        demo_dynamic_constraints()
        demo_turnover_and_transaction_costs()
        demo_constraint_optimization()
        
        # Create visualizations
        fig = create_visualization()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("Key features demonstrated:")
        print("✓ Enhanced position limits with asset-specific controls")
        print("✓ Sector constraints with correlation adjustment")
        print("✓ Comprehensive ESG and sustainability constraints")
        print("✓ Turnover limits with transaction cost integration")
        print("✓ Dynamic constraint adjustment for market conditions")
        print("✓ Constraint conflict detection and resolution")
        print("✓ Comprehensive constraint reporting and validation")
        
        # Show plot if running interactively
        try:
            plt.show()
        except:
            pass
            
    except Exception as e:
        print(f"\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()