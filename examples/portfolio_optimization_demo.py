"""
Portfolio Optimization Demo

This demo showcases the various portfolio optimization methods implemented
in the quantitative framework, including mean-variance, Black-Litterman,
risk parity, factor-based, and transaction cost optimization.
"""

import numpy as np
import pandas as pd
from decimal import Decimal
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.optimization import (
    PortfolioOptimizer, BlackLittermanOptimizer, RiskParityOptimizer,
    FactorBasedOptimizer, TransactionCostOptimizer
)
from src.domain.entities import Portfolio, Position


def create_sample_data():
    """Create sample market data for demonstration."""
    np.random.seed(42)
    
    # Sample assets
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'KO', 'WMT']
    
    # Expected returns (annualized)
    expected_returns = pd.Series([
        0.12, 0.15, 0.10, 0.14, 0.18, 0.08, 0.07, 0.06, 0.05, 0.04
    ], index=assets)
    
    # Create realistic covariance matrix
    n_assets = len(assets)
    
    # Base correlation matrix with sector clustering
    correlation = np.eye(n_assets)
    
    # Technology stocks (AAPL, GOOGL, MSFT, AMZN, TSLA) - higher correlation
    tech_indices = [0, 1, 2, 3, 4]
    for i in tech_indices:
        for j in tech_indices:
            if i != j:
                correlation[i, j] = np.random.uniform(0.4, 0.7)
    
    # Other stocks - lower correlation
    other_indices = [5, 6, 7, 8, 9]
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j and (i not in tech_indices or j not in tech_indices):
                correlation[i, j] = np.random.uniform(0.1, 0.3)
    
    # Ensure positive semi-definite
    correlation = (correlation + correlation.T) / 2
    eigenvals, eigenvecs = np.linalg.eigh(correlation)
    eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
    correlation = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Volatilities (annualized)
    volatilities = np.array([0.25, 0.30, 0.22, 0.28, 0.45, 0.20, 0.15, 0.12, 0.10, 0.08])
    
    # Create covariance matrix
    covariance_matrix = pd.DataFrame(
        np.outer(volatilities, volatilities) * correlation,
        index=assets,
        columns=assets
    )
    
    return expected_returns, covariance_matrix


def demo_mean_variance_optimization():
    """Demonstrate mean-variance optimization."""
    print("=" * 60)
    print("MEAN-VARIANCE OPTIMIZATION DEMO")
    print("=" * 60)
    
    expected_returns, covariance_matrix = create_sample_data()
    optimizer = PortfolioOptimizer()
    
    # Standard mean-variance optimization
    constraints = [{'type': 'method', 'method': 'mean_variance'}]
    result = optimizer.optimize(expected_returns, covariance_matrix, constraints)
    
    if result['success']:
        weights = result['weights']
        print(f"Expected Return: {result['expected_return']:.4f}")
        print(f"Expected Risk: {result['expected_risk']:.4f}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.4f}")
        print("\nOptimal Weights:")
        for asset, weight in weights.items():
            if weight > 0.01:  # Only show significant weights
                print(f"  {asset}: {weight:.4f} ({weight*100:.1f}%)")
    else:
        print(f"Optimization failed: {result['message']}")
    
    # Minimum variance optimization
    print("\n" + "-" * 40)
    print("MINIMUM VARIANCE OPTIMIZATION")
    print("-" * 40)
    
    constraints = [{'type': 'method', 'method': 'min_variance'}]
    result = optimizer.optimize(expected_returns, covariance_matrix, constraints)
    
    if result['success']:
        weights = result['weights']
        print(f"Expected Risk: {result['expected_risk']:.4f}")
        print("\nMinimum Variance Weights:")
        for asset, weight in weights.items():
            if weight > 0.01:
                print(f"  {asset}: {weight:.4f} ({weight*100:.1f}%)")


def demo_risk_parity_optimization():
    """Demonstrate risk parity optimization."""
    print("\n" + "=" * 60)
    print("RISK PARITY OPTIMIZATION DEMO")
    print("=" * 60)
    
    expected_returns, covariance_matrix = create_sample_data()
    rp_optimizer = RiskParityOptimizer()
    
    # Equal Risk Contribution
    print("EQUAL RISK CONTRIBUTION")
    print("-" * 30)
    
    result = rp_optimizer.optimize_equal_risk_contribution(covariance_matrix, [])
    
    if result.success:
        weights = result.weights
        print(f"Portfolio Risk: {result.expected_risk:.4f}")
        print("\nERC Weights:")
        for asset, weight in weights.items():
            if weight > 0.01:
                print(f"  {asset}: {weight:.4f} ({weight*100:.1f}%)")
        
        # Show risk contributions
        risk_contrib = rp_optimizer.calculate_risk_contributions(weights, covariance_matrix)
        print("\nRisk Contributions:")
        for asset, contrib in risk_contrib.items():
            if contrib > 0.01:
                print(f"  {asset}: {contrib:.4f} ({contrib*100:.1f}%)")
    
    # Inverse Volatility
    print("\n" + "-" * 30)
    print("INVERSE VOLATILITY")
    print("-" * 30)
    
    result = rp_optimizer.optimize_inverse_volatility(covariance_matrix, [])
    
    if result.success:
        weights = result.weights
        print(f"Portfolio Risk: {result.expected_risk:.4f}")
        print("\nInverse Volatility Weights:")
        for asset, weight in weights.items():
            if weight > 0.01:
                print(f"  {asset}: {weight:.4f} ({weight*100:.1f}%)")


def demo_black_litterman_optimization():
    """Demonstrate Black-Litterman optimization."""
    print("\n" + "=" * 60)
    print("BLACK-LITTERMAN OPTIMIZATION DEMO")
    print("=" * 60)
    
    expected_returns, covariance_matrix = create_sample_data()
    bl_optimizer = BlackLittermanOptimizer()
    
    # Market capitalization weights (proxy)
    market_caps = pd.Series([
        2000, 1500, 1800, 1600, 800, 400, 450, 350, 280, 320
    ], index=expected_returns.index)
    
    # Create some views
    views = [
        bl_optimizer.create_absolute_view('TSLA', 0.25, confidence=0.02),  # TSLA will return 25%
        bl_optimizer.create_relative_view('AAPL', 'MSFT', 0.05, confidence=0.015),  # AAPL outperforms MSFT by 5%
    ]
    
    constraints = [{
        'type': 'black_litterman',
        'market_caps': market_caps,
        'views': views
    }]
    
    result = bl_optimizer.optimize(expected_returns, covariance_matrix, constraints)
    
    if result.success:
        weights = result.weights
        print(f"Expected Return: {result.expected_return:.4f}")
        print(f"Expected Risk: {result.expected_risk:.4f}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
        print("\nBlack-Litterman Weights:")
        for asset, weight in weights.items():
            if weight > 0.01:
                print(f"  {asset}: {weight:.4f} ({weight*100:.1f}%)")


def demo_transaction_cost_optimization():
    """Demonstrate transaction cost optimization."""
    print("\n" + "=" * 60)
    print("TRANSACTION COST OPTIMIZATION DEMO")
    print("=" * 60)
    
    # Create current and target portfolios
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    current_weights = pd.Series([0.30, 0.20, 0.25, 0.15, 0.10], index=assets)
    target_weights = pd.Series([0.20, 0.20, 0.20, 0.20, 0.20], index=assets)  # Equal weight target
    portfolio_value = Decimal('1000000')  # $1M portfolio
    
    tc_optimizer = TransactionCostOptimizer()
    
    # Linear cost optimization
    print("LINEAR TRANSACTION COST OPTIMIZATION")
    print("-" * 40)
    
    cost_params = {
        'method': 'linear_cost',
        'linear_cost_rate': 0.001,  # 10 bps
        'tracking_error_weight': 1.0,
        'transaction_cost_weight': 1.0
    }
    
    result = tc_optimizer.optimize_rebalancing(
        current_weights, target_weights, portfolio_value, cost_params
    )
    
    if result.success:
        print(f"Total Turnover: {result.total_turnover:.4f} ({result.total_turnover*100:.1f}%)")
        print(f"Transaction Costs: ${result.transaction_costs:.2f}")
        print(f"Number of Trades: {result.num_trades}")
        
        print("\nTrades:")
        for _, trade in result.trades.iterrows():
            if abs(trade['trade_amount']) > 0.001:
                print(f"  {trade['symbol']}: {trade['current_weight']:.3f} -> "
                      f"{trade['current_weight'] + trade['trade_amount']:.3f} "
                      f"(trade: {trade['trade_amount']:+.3f})")
    
    # Quadratic cost optimization
    print("\n" + "-" * 40)
    print("QUADRATIC TRANSACTION COST OPTIMIZATION")
    print("-" * 40)
    
    cost_params = {
        'method': 'quadratic_cost',
        'linear_cost_rate': 0.001,
        'market_impact_rate': 0.0001,
        'tracking_error_weight': 1.0,
        'transaction_cost_weight': 1.0
    }
    
    result = tc_optimizer.optimize_rebalancing(
        current_weights, target_weights, portfolio_value, cost_params
    )
    
    if result.success:
        print(f"Total Turnover: {result.total_turnover:.4f} ({result.total_turnover*100:.1f}%)")
        print(f"Transaction Costs: ${result.transaction_costs:.2f}")
        print(f"Number of Trades: {result.num_trades}")


def demo_factor_based_optimization():
    """Demonstrate factor-based optimization."""
    print("\n" + "=" * 60)
    print("FACTOR-BASED OPTIMIZATION DEMO")
    print("=" * 60)
    
    expected_returns, covariance_matrix = create_sample_data()
    factor_optimizer = FactorBasedOptimizer()
    
    # Create sample factor loadings
    assets = expected_returns.index
    factors = ['Market', 'Size', 'Value', 'Momentum']
    
    # Sample factor loadings (in practice, these would come from factor models)
    np.random.seed(42)
    factor_loadings = pd.DataFrame(
        np.random.normal(0, 0.5, (len(assets), len(factors))),
        index=assets,
        columns=factors
    )
    
    # Market factor should be close to 1 for most stocks
    factor_loadings['Market'] = np.random.normal(1.0, 0.3, len(assets))
    
    print("FACTOR EXPOSURE OPTIMIZATION")
    print("-" * 35)
    
    constraints = [{
        'type': 'factor_based',
        'method': 'factor_exposure',
        'factor_loadings': factor_loadings,
        'target_exposures': {
            'Market': 1.0,    # Market neutral
            'Size': -0.2,     # Small cap tilt
            'Value': 0.3,     # Value tilt
            'Momentum': 0.1   # Slight momentum tilt
        },
        'exposure_tolerance': 0.2,
        'risk_aversion': 2.0
    }]
    
    result = factor_optimizer.optimize(expected_returns, covariance_matrix, constraints)
    
    if result.success:
        weights = result.weights
        print(f"Expected Return: {result.expected_return:.4f}")
        print(f"Expected Risk: {result.expected_risk:.4f}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.4f}")
        
        print("\nFactor-Based Weights:")
        for asset, weight in weights.items():
            if weight > 0.01:
                print(f"  {asset}: {weight:.4f} ({weight*100:.1f}%)")
        
        if 'factor_exposures' in result.metadata:
            print("\nFactor Exposures:")
            for factor, exposure in result.metadata['factor_exposures'].items():
                print(f"  {factor}: {exposure:.3f}")


def main():
    """Run all optimization demos."""
    print("PORTFOLIO OPTIMIZATION METHODS DEMONSTRATION")
    print("=" * 80)
    
    try:
        demo_mean_variance_optimization()
        demo_risk_parity_optimization()
        demo_black_litterman_optimization()
        demo_transaction_cost_optimization()
        demo_factor_based_optimization()
        
        print("\n" + "=" * 80)
        print("ALL OPTIMIZATION DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()