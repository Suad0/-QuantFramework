"""
Tests for portfolio optimization infrastructure.

This module tests the various portfolio optimization methods including
mean-variance, Black-Litterman, risk parity, factor-based, and transaction cost optimization.
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime

from src.infrastructure.optimization import (
    PortfolioOptimizer, BlackLittermanOptimizer, RiskParityOptimizer,
    FactorBasedOptimizer, TransactionCostOptimizer,
    PositionLimitConstraint, SectorConstraint, TurnoverConstraint, ESGConstraint,
    ConstraintManager, ConstraintType, ConstraintSeverity, MarketCondition
)
from src.domain.entities import Portfolio, Position
from src.domain.exceptions import OptimizationError, ValidationError


class TestPortfolioOptimizer:
    """Test the main portfolio optimizer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        
        # Create sample expected returns
        assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        expected_returns = pd.Series(
            [0.12, 0.15, 0.10, 0.14, 0.18],
            index=assets
        )
        
        # Create sample covariance matrix
        n_assets = len(assets)
        correlation = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1.0)
        
        volatilities = np.array([0.20, 0.25, 0.18, 0.22, 0.35])
        covariance_matrix = pd.DataFrame(
            np.outer(volatilities, volatilities) * correlation,
            index=assets,
            columns=assets
        )
        
        return expected_returns, covariance_matrix
    
    @pytest.fixture
    def optimizer(self):
        """Create portfolio optimizer instance."""
        return PortfolioOptimizer()
    
    def test_mean_variance_optimization(self, optimizer, sample_data):
        """Test mean-variance optimization."""
        expected_returns, covariance_matrix = sample_data
        
        constraints = [{'type': 'method', 'method': 'mean_variance'}]
        
        result = optimizer.optimize(expected_returns, covariance_matrix, constraints)
        
        assert result['success'] is True
        assert 'weights' in result
        assert len(result['weights']) == len(expected_returns)
        assert abs(sum(result['weights'].values()) - 1.0) < 1e-6  # Weights sum to 1
        assert all(w >= -1e-6 for w in result['weights'].values())  # Long-only
        assert result['expected_return'] > 0
        assert result['expected_risk'] > 0
    
    def test_min_variance_optimization(self, optimizer, sample_data):
        """Test minimum variance optimization."""
        expected_returns, covariance_matrix = sample_data
        
        constraints = [{'type': 'method', 'method': 'min_variance'}]
        
        result = optimizer.optimize(expected_returns, covariance_matrix, constraints)
        
        assert result['success'] is True
        assert result['method'] == 'min_variance'
        assert result['expected_risk'] > 0
        assert abs(sum(result['weights'].values()) - 1.0) < 1e-6
    
    def test_max_diversification_optimization(self, optimizer, sample_data):
        """Test maximum diversification optimization."""
        expected_returns, covariance_matrix = sample_data
        
        constraints = [{'type': 'method', 'method': 'max_diversification'}]
        
        result = optimizer.optimize(expected_returns, covariance_matrix, constraints)
        
        assert result['success'] is True
        assert result['method'] == 'max_diversification'
        assert result['expected_return'] > 0
        assert result['expected_risk'] > 0
    
    def test_invalid_inputs(self, optimizer):
        """Test optimization with invalid inputs."""
        # Empty expected returns
        with pytest.raises(Exception):
            optimizer.optimize(pd.Series(dtype=float), pd.DataFrame(), [])
        
        # Mismatched dimensions
        expected_returns = pd.Series([0.1, 0.2], index=['A', 'B'])
        covariance_matrix = pd.DataFrame([[0.04]], index=['A'], columns=['A'])
        
        with pytest.raises(Exception):
            optimizer.optimize(expected_returns, covariance_matrix, [])


class TestBlackLittermanOptimizer:
    """Test Black-Litterman optimization."""
    
    @pytest.fixture
    def bl_optimizer(self):
        """Create Black-Litterman optimizer instance."""
        return BlackLittermanOptimizer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for Black-Litterman testing."""
        assets = ['AAPL', 'GOOGL', 'MSFT']
        
        # Market equilibrium returns (implied from market caps)
        expected_returns = pd.Series([0.10, 0.12, 0.08], index=assets)
        
        # Covariance matrix
        covariance_matrix = pd.DataFrame([
            [0.04, 0.02, 0.01],
            [0.02, 0.06, 0.015],
            [0.01, 0.015, 0.03]
        ], index=assets, columns=assets)
        
        # Market capitalization weights
        market_caps = pd.Series([2000, 1500, 1800], index=assets)
        
        return expected_returns, covariance_matrix, market_caps
    
    def test_black_litterman_no_views(self, bl_optimizer, sample_data):
        """Test Black-Litterman with no views (should return market portfolio)."""
        expected_returns, covariance_matrix, market_caps = sample_data
        
        constraints = [{
            'type': 'black_litterman',
            'market_caps': market_caps
        }]
        
        result = bl_optimizer.optimize(expected_returns, covariance_matrix, constraints)
        
        assert result.success is True
        assert result.method == 'black_litterman_no_views'
        assert abs(result.weights.sum() - 1.0) < 1e-6
    
    def test_black_litterman_with_absolute_view(self, bl_optimizer, sample_data):
        """Test Black-Litterman with absolute view."""
        expected_returns, covariance_matrix, market_caps = sample_data
        
        # Create absolute view: AAPL will return 15%
        view = bl_optimizer.create_absolute_view('AAPL', 0.15, confidence=0.01)
        
        constraints = [{
            'type': 'black_litterman',
            'market_caps': market_caps,
            'views': [view]
        }]
        
        result = bl_optimizer.optimize(expected_returns, covariance_matrix, constraints)
        
        assert result.success is True
        assert result.method == 'black_litterman'
        assert abs(result.weights.sum() - 1.0) < 1e-6
        # AAPL should have higher weight due to positive view
        assert result.weights['AAPL'] > 0
    
    def test_black_litterman_with_relative_view(self, bl_optimizer, sample_data):
        """Test Black-Litterman with relative view."""
        expected_returns, covariance_matrix, market_caps = sample_data
        
        # Create relative view: AAPL will outperform MSFT by 5%
        view = bl_optimizer.create_relative_view('AAPL', 'MSFT', 0.05, confidence=0.01)
        
        constraints = [{
            'type': 'black_litterman',
            'market_caps': market_caps,
            'views': [view]
        }]
        
        result = bl_optimizer.optimize(expected_returns, covariance_matrix, constraints)
        
        assert result.success is True
        assert result.method == 'black_litterman'
        assert abs(result.weights.sum() - 1.0) < 1e-6


class TestRiskParityOptimizer:
    """Test risk parity optimization."""
    
    @pytest.fixture
    def rp_optimizer(self):
        """Create risk parity optimizer instance."""
        return RiskParityOptimizer()
    
    @pytest.fixture
    def sample_covariance(self):
        """Create sample covariance matrix."""
        assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        covariance_matrix = pd.DataFrame([
            [0.04, 0.02, 0.01, 0.015],
            [0.02, 0.06, 0.015, 0.02],
            [0.01, 0.015, 0.03, 0.012],
            [0.015, 0.02, 0.012, 0.05]
        ], index=assets, columns=assets)
        
        return covariance_matrix
    
    def test_equal_risk_contribution(self, rp_optimizer, sample_covariance):
        """Test equal risk contribution optimization."""
        result = rp_optimizer.optimize_equal_risk_contribution(sample_covariance, [])
        
        assert result.success is True
        assert result.method == 'equal_risk_contribution'
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert all(w >= -1e-6 for w in result.weights.values)  # Long-only
        
        # Check risk contributions are approximately equal
        risk_contrib = rp_optimizer.calculate_risk_contributions(result.weights, sample_covariance)
        target_contrib = 1.0 / len(sample_covariance)
        
        # Risk contributions should be close to equal (within tolerance)
        for contrib in risk_contrib.values:
            assert abs(contrib - target_contrib) < 0.1  # 10% tolerance
    
    def test_inverse_volatility(self, rp_optimizer, sample_covariance):
        """Test inverse volatility optimization."""
        result = rp_optimizer.optimize_inverse_volatility(sample_covariance, [])
        
        assert result.success is True
        assert result.method == 'inverse_volatility'
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert all(w >= -1e-6 for w in result.weights.values)
        
        # Assets with lower volatility should have higher weights
        volatilities = np.sqrt(np.diag(sample_covariance.values))
        expected_weights = (1.0 / volatilities) / np.sum(1.0 / volatilities)
        
        for i, asset in enumerate(sample_covariance.index):
            assert abs(result.weights[asset] - expected_weights[i]) < 1e-6
    
    def test_risk_contributions_calculation(self, rp_optimizer, sample_covariance):
        """Test risk contributions calculation."""
        # Equal weights portfolio
        equal_weights = pd.Series(0.25, index=sample_covariance.index)
        
        risk_contrib = rp_optimizer.calculate_risk_contributions(equal_weights, sample_covariance)
        
        assert len(risk_contrib) == len(sample_covariance)
        assert abs(risk_contrib.sum() - 1.0) < 1e-6  # Risk contributions sum to 1
        assert all(contrib >= 0 for contrib in risk_contrib.values)


class TestFactorBasedOptimizer:
    """Test factor-based optimization."""
    
    @pytest.fixture
    def factor_optimizer(self):
        """Create factor-based optimizer instance."""
        return FactorBasedOptimizer()
    
    @pytest.fixture
    def sample_factor_data(self):
        """Create sample factor data."""
        assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        factors = ['Market', 'Size', 'Value']
        
        expected_returns = pd.Series([0.12, 0.15, 0.10, 0.14], index=assets)
        
        covariance_matrix = pd.DataFrame([
            [0.04, 0.02, 0.01, 0.015],
            [0.02, 0.06, 0.015, 0.02],
            [0.01, 0.015, 0.03, 0.012],
            [0.015, 0.02, 0.012, 0.05]
        ], index=assets, columns=assets)
        
        # Factor loadings (assets x factors)
        factor_loadings = pd.DataFrame([
            [1.2, -0.5, 0.3],   # AAPL: high market beta, small cap, growth
            [1.1, -0.8, 0.1],   # GOOGL: high market beta, small cap, growth
            [0.9, 0.2, -0.2],   # MSFT: market beta, large cap, growth
            [1.3, -0.3, 0.5]    # AMZN: high market beta, small cap, growth
        ], index=assets, columns=factors)
        
        return expected_returns, covariance_matrix, factor_loadings
    
    def test_factor_exposure_optimization(self, factor_optimizer, sample_factor_data):
        """Test factor exposure optimization."""
        expected_returns, covariance_matrix, factor_loadings = sample_factor_data
        
        constraints = [{
            'type': 'factor_based',
            'method': 'factor_exposure',
            'factor_loadings': factor_loadings,
            'target_exposures': {
                'Market': 1.0,  # Target market beta of 1.0
                'Size': -0.2,   # Slight small cap tilt
                'Value': 0.1    # Slight value tilt
            },
            'exposure_tolerance': 0.2
        }]
        
        result = factor_optimizer.optimize(expected_returns, covariance_matrix, constraints)
        
        assert result.success is True
        assert result.method == 'factor_exposure'
        assert abs(result.weights.sum() - 1.0) < 1e-6
        assert 'factor_exposures' in result.metadata
        
        # Check that factor exposures are close to targets
        exposures = result.metadata['factor_exposures']
        assert abs(exposures['Market'] - 1.0) < 0.3  # Within tolerance
        assert abs(exposures['Size'] - (-0.2)) < 0.3
        assert abs(exposures['Value'] - 0.1) < 0.3


class TestTransactionCostOptimizer:
    """Test transaction cost optimization."""
    
    @pytest.fixture
    def tc_optimizer(self):
        """Create transaction cost optimizer instance."""
        return TransactionCostOptimizer()
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Create sample portfolio data for rebalancing."""
        assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        
        current_weights = pd.Series([0.3, 0.2, 0.3, 0.2], index=assets)
        target_weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=assets)
        portfolio_value = Decimal('1000000')  # $1M portfolio
        
        return current_weights, target_weights, portfolio_value
    
    def test_linear_cost_optimization(self, tc_optimizer, sample_portfolio_data):
        """Test linear transaction cost optimization."""
        current_weights, target_weights, portfolio_value = sample_portfolio_data
        
        cost_params = {
            'method': 'linear_cost',
            'linear_cost_rate': 0.001,  # 10 bps
            'tracking_error_weight': 1.0,
            'transaction_cost_weight': 1.0
        }
        
        result = tc_optimizer.optimize_rebalancing(
            current_weights, target_weights, portfolio_value, cost_params
        )
        
        assert result.success is True
        assert result.method == 'linear_cost'
        assert len(result.trades) == len(current_weights)
        assert 'symbol' in result.trades.columns
        assert 'current_weight' in result.trades.columns
        assert 'target_weight' in result.trades.columns
        assert 'trade_amount' in result.trades.columns
        
        # Check that trades move towards target (but may not reach exactly due to costs)
        for _, trade in result.trades.iterrows():
            symbol = trade['symbol']
            current_w = current_weights[symbol]
            target_w = target_weights[symbol]
            new_w = trade['current_weight'] + trade['trade_amount']
            
            # New weight should be between current and target
            if target_w > current_w:
                assert new_w >= current_w  # Should increase
            elif target_w < current_w:
                assert new_w <= current_w  # Should decrease
    
    def test_quadratic_cost_optimization(self, tc_optimizer, sample_portfolio_data):
        """Test quadratic transaction cost optimization."""
        current_weights, target_weights, portfolio_value = sample_portfolio_data
        
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
        
        assert result.success is True
        assert result.method == 'quadratic_cost'
        assert result.total_turnover >= 0
        assert result.transaction_costs >= 0
    
    def test_implementation_shortfall_estimation(self, tc_optimizer, sample_portfolio_data):
        """Test implementation shortfall estimation."""
        current_weights, target_weights, portfolio_value = sample_portfolio_data
        
        trades = target_weights - current_weights
        
        shortfall = tc_optimizer.estimate_implementation_shortfall(trades)
        
        assert 'total_implementation_shortfall' in shortfall
        assert 'temporary_market_impact' in shortfall
        assert 'permanent_market_impact' in shortfall
        assert 'timing_risk' in shortfall
        assert 'opportunity_cost' in shortfall
        
        assert shortfall['total_implementation_shortfall'] >= 0
        assert shortfall['total_turnover'] > 0


class TestConstraints:
    """Test portfolio optimization constraints."""
    
    def test_position_limit_constraint(self):
        """Test enhanced position limit constraint."""
        constraint = PositionLimitConstraint(
            name="Position Limits",
            description="Limit individual position sizes",
            constraint_type=ConstraintType.POSITION_LIMIT,
            min_weight=0.05,
            max_weight=0.30
        )
        
        # Test valid weights
        valid_weights = np.array([0.25, 0.25, 0.25, 0.25])
        assert constraint.validate(valid_weights)
        
        # Test invalid weights (too small)
        invalid_weights_small = np.array([0.02, 0.33, 0.33, 0.32])
        assert not constraint.validate(invalid_weights_small)
        
        # Test invalid weights (too large)
        invalid_weights_large = np.array([0.40, 0.20, 0.20, 0.20])
        assert not constraint.validate(invalid_weights_large)
        
        # Test violation checking
        violation = constraint.check_violation(invalid_weights_large)
        assert violation is not None
        assert violation.violation_type == "position_limit_max"
        assert violation.current_value == 0.40
        assert violation.limit_value == 0.30
    
    def test_position_limit_with_asset_specific_limits(self):
        """Test position limits with asset-specific constraints."""
        constraint = PositionLimitConstraint(
            name="Position Limits",
            description="Limit individual position sizes",
            constraint_type=ConstraintType.POSITION_LIMIT,
            min_weight=0.05,
            max_weight=0.30,
            asset_specific_limits={'AAPL': (0.10, 0.25), 'GOOGL': (0.05, 0.20)}
        )
        
        asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        weights = np.array([0.22, 0.18, 0.30, 0.30])  # AAPL and GOOGL within specific limits
        
        assert constraint.validate(weights, asset_names=asset_names)
        
        # Test violation of asset-specific limit
        weights_violation = np.array([0.27, 0.18, 0.30, 0.25])  # AAPL exceeds its specific limit
        assert not constraint.validate(weights_violation, asset_names=asset_names)
    
    def test_position_limit_dynamic_adjustment(self):
        """Test position limit dynamic adjustment for market conditions."""
        constraint = PositionLimitConstraint(
            name="Dynamic Position Limits",
            description="Position limits with market adjustment",
            constraint_type=ConstraintType.POSITION_LIMIT,
            min_weight=0.05,
            max_weight=0.30,
            is_dynamic=True,
            market_condition_sensitivity=0.5
        )
        
        # High stress market condition
        market_condition = MarketCondition(
            volatility=0.4,
            liquidity=0.6,
            market_stress=0.8,
            correlation_regime='high'
        )
        
        adjusted_constraint = constraint.adjust_for_market_conditions(market_condition)
        
        # Max weight should be reduced due to high stress
        assert adjusted_constraint.max_weight < constraint.max_weight
    
    def test_sector_constraint_enhanced(self):
        """Test enhanced sector constraint."""
        sector_mapping = {
            'AAPL': 'Technology',
            'GOOGL': 'Technology',
            'MSFT': 'Technology',
            'JPM': 'Financial',
            'BAC': 'Financial'
        }
        
        sector_limits = {
            'Technology': (0.2, 0.6),  # 20-60% in tech
            'Financial': (0.1, 0.4)    # 10-40% in financial
        }
        
        constraint = SectorConstraint(
            name="Sector Limits",
            description="Limit sector exposures",
            constraint_type=ConstraintType.SECTOR_LIMIT,
            sector_mapping=sector_mapping,
            sector_limits=sector_limits,
            correlation_adjustment=True
        )
        
        assets = ['AAPL', 'GOOGL', 'MSFT', 'JPM', 'BAC']
        
        # Test valid weights
        # Tech: 50% (within 20-60% limit), Financial: 35% (within 10-40% limit)
        valid_weights = np.array([0.10, 0.15, 0.25, 0.15, 0.20])  # Tech: 50%, Financial: 35%
        assert constraint.validate(valid_weights, asset_names=assets)
        
        # Test sector exposure calculation
        exposures = constraint.get_sector_exposures(valid_weights, assets)
        assert abs(exposures['Technology'] - 0.5) < 1e-6
        assert abs(exposures['Financial'] - 0.35) < 1e-6
        
        # Test invalid weights (too much tech)
        invalid_weights = np.array([0.30, 0.30, 0.30, 0.05, 0.05])  # 90% tech, 10% financial
        assert not constraint.validate(invalid_weights, asset_names=assets)
        
        violation = constraint.check_violation(invalid_weights, asset_names=assets)
        assert violation is not None
        assert "Technology" in violation.constraint_name
    
    def test_turnover_constraint_enhanced(self):
        """Test enhanced turnover constraint."""
        current_weights = np.array([0.3, 0.2, 0.3, 0.2])
        
        constraint = TurnoverConstraint(
            name="Turnover Limit",
            description="Limit portfolio turnover",
            constraint_type=ConstraintType.TURNOVER_LIMIT,
            max_turnover=0.2,
            current_weights=current_weights,
            one_way_turnover=False
        )
        
        # Test valid weights (low turnover)
        valid_weights = np.array([0.28, 0.22, 0.28, 0.22])  # 8% turnover
        assert constraint.validate(valid_weights)
        
        # Test turnover metrics calculation
        metrics = constraint.calculate_turnover_metrics(valid_weights)
        assert 'total_turnover' in metrics
        assert 'one_way_turnover' in metrics
        assert 'buy_turnover' in metrics
        assert 'sell_turnover' in metrics
        assert abs(metrics['total_turnover'] - 0.08) < 1e-6
        
        # Test invalid weights (high turnover)
        invalid_weights = np.array([0.1, 0.4, 0.1, 0.4])  # 80% turnover
        assert not constraint.validate(invalid_weights)
        
        violation = constraint.check_violation(invalid_weights)
        assert violation is not None
        assert violation.violation_type == "total_turnover"
    
    def test_turnover_constraint_with_transaction_costs(self):
        """Test turnover constraint with transaction cost integration."""
        current_weights = np.array([0.3, 0.2, 0.3, 0.2])
        
        # Simple linear transaction cost model
        def transaction_cost_model(trades):
            return 0.001 * np.sum(np.abs(trades))  # 10 bps per trade
        
        constraint = TurnoverConstraint(
            name="Turnover with Transaction Costs",
            description="Limit turnover and transaction costs",
            constraint_type=ConstraintType.TURNOVER_LIMIT,
            max_turnover=0.5,
            current_weights=current_weights,
            transaction_cost_model=transaction_cost_model,
            max_transaction_cost=0.01  # 1% max transaction cost
        )
        
        # Test weights that violate transaction cost limit
        high_cost_weights = np.array([0.0, 0.0, 0.5, 0.5])  # High turnover, high cost
        assert not constraint.validate(high_cost_weights)
        
        violation = constraint.check_violation(high_cost_weights)
        assert violation is not None
        # Should have either turnover or transaction cost violation
        assert violation.violation_type in ["total_turnover", "transaction_cost"]
    
    def test_esg_constraint_comprehensive(self):
        """Test comprehensive ESG constraint."""
        esg_scores = {'AAPL': 85, 'GOOGL': 80, 'MSFT': 90, 'XOM': 40, 'TSLA': 75}
        environmental_scores = {'AAPL': 80, 'GOOGL': 85, 'MSFT': 95, 'XOM': 20, 'TSLA': 90}
        carbon_intensities = {'AAPL': 10, 'GOOGL': 8, 'MSFT': 5, 'XOM': 100, 'TSLA': 15}
        
        constraint = ESGConstraint(
            name="Comprehensive ESG",
            description="ESG constraints with carbon limits",
            constraint_type=ConstraintType.ESG_CONSTRAINT,
            esg_scores=esg_scores,
            min_portfolio_score=70,
            environmental_scores=environmental_scores,
            min_environmental_score=75,
            carbon_intensities=carbon_intensities,
            max_portfolio_carbon_intensity=25,
            exclude_assets=['XOM']  # Exclude oil company
        )
        
        assets = ['AAPL', 'GOOGL', 'MSFT', 'XOM', 'TSLA']
        
        # Test valid weights (excluding XOM, meeting ESG and carbon criteria)
        valid_weights = np.array([0.3, 0.3, 0.3, 0.0, 0.1])
        assert constraint.validate(valid_weights, asset_names=assets)
        
        # Test ESG metrics calculation
        metrics = constraint.calculate_esg_metrics(valid_weights, assets)
        assert 'esg_score' in metrics
        assert 'environmental_score' in metrics
        assert 'carbon_intensity' in metrics
        assert metrics['esg_score'] >= 70
        assert metrics['environmental_score'] >= 75
        assert metrics['carbon_intensity'] <= 25
        
        # Test invalid weights (includes excluded asset)
        invalid_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Includes XOM
        assert not constraint.validate(invalid_weights, asset_names=assets)
    
    def test_constraint_manager(self):
        """Test the comprehensive constraint manager."""
        manager = ConstraintManager()
        
        # Add multiple constraints
        position_constraint = PositionLimitConstraint(
            name="Position Limits",
            description="Basic position limits",
            constraint_type=ConstraintType.POSITION_LIMIT,
            min_weight=0.05,
            max_weight=0.30
        )
        
        sector_mapping = {'AAPL': 'Tech', 'GOOGL': 'Tech', 'MSFT': 'Tech', 'JPM': 'Finance'}
        sector_constraint = SectorConstraint(
            name="Sector Limits",
            description="Sector exposure limits",
            constraint_type=ConstraintType.SECTOR_LIMIT,
            sector_mapping=sector_mapping,
            sector_limits={'Tech': (0.2, 0.7), 'Finance': (0.1, 0.3)}
        )
        
        manager.add_constraint(position_constraint)
        manager.add_constraint(sector_constraint)
        
        # Test constraint summary
        summary = manager.get_constraint_summary()
        assert summary['total_constraints'] == 2
        assert summary['active_constraints'] == 2
        assert ConstraintType.POSITION_LIMIT.value in summary['constraint_types']
        assert ConstraintType.SECTOR_LIMIT.value in summary['constraint_types']
        
        # Test constraint application
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        asset_names = ['AAPL', 'GOOGL', 'MSFT', 'JPM']
        
        constraint_params = manager.apply_all_constraints(weights, asset_names=asset_names)
        assert 'constraints' in constraint_params
        assert 'bounds' in constraint_params
        assert 'constraint_metadata' in constraint_params
        
        # Test validation
        validation_results = manager.validate_all_constraints(weights, asset_names=asset_names)
        assert 'is_valid' in validation_results
        assert 'violations' in validation_results
        assert 'validation_results' in validation_results
        
        # Test constraint report
        report = manager.create_constraint_report(weights, asset_names=asset_names)
        assert 'timestamp' in report
        assert 'summary' in report
        assert 'validation' in report
        assert 'recommendations' in report
    
    def test_constraint_manager_with_market_conditions(self):
        """Test constraint manager with dynamic market adjustment."""
        manager = ConstraintManager()
        
        # Add dynamic constraint
        dynamic_constraint = PositionLimitConstraint(
            name="Dynamic Position Limits",
            description="Market-sensitive position limits",
            constraint_type=ConstraintType.POSITION_LIMIT,
            min_weight=0.05,
            max_weight=0.30,
            is_dynamic=True,
            market_condition_sensitivity=0.5
        )
        
        manager.add_constraint(dynamic_constraint)
        
        # Set market conditions
        market_condition = MarketCondition(
            volatility=0.35,
            liquidity=0.7,
            market_stress=0.6,
            correlation_regime='medium'
        )
        
        manager.update_market_condition(market_condition)
        
        # Get active constraints (should be adjusted)
        active_constraints = manager.get_active_constraints()
        assert len(active_constraints) == 1
        
        adjusted_constraint = active_constraints[0]
        # Max weight should be reduced due to market stress
        assert adjusted_constraint.max_weight < dynamic_constraint.max_weight
        
        # Test summary with market conditions
        summary = manager.get_constraint_summary()
        assert summary['market_condition_applied'] is True
        assert summary['dynamic_constraints'] == 1


if __name__ == '__main__':
    pytest.main([__file__])