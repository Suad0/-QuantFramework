"""
Advanced Feature Engineering Demo

This script demonstrates the advanced feature engineering capabilities including:
1. Configurable rolling window operations with multiple lookback periods
2. Multiple data imputation strategies
3. Look-ahead bias detection and prevention
4. Feature selection and dimensionality reduction tools
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings

# Add the src directory to the path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.application.services.feature_engine import FeatureEngine
from src.domain.exceptions import ValidationError


def create_sample_data(n_days=252, start_date='2020-01-01'):
    """Create sample financial data with realistic patterns"""
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data with trends and volatility
    returns = np.random.randn(n_days) * 0.02
    returns[50:100] += 0.001  # Add a trend period
    returns[150:200] *= 2     # Add a volatile period
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.random.rand(n_days) * 0.02),
        'low': prices * (1 - np.random.rand(n_days) * 0.02),
        'volume': np.random.randint(100000, 1000000, n_days),
        'sector': np.random.choice(['Tech', 'Finance', 'Healthcare'], n_days)
    }, index=dates)
    
    # Introduce some missing values to demonstrate imputation
    missing_indices = np.random.choice(n_days, size=int(n_days * 0.05), replace=False)
    data.loc[data.index[missing_indices[:len(missing_indices)//2]], 'close'] = np.nan
    data.loc[data.index[missing_indices[len(missing_indices)//2:]], 'volume'] = np.nan
    
    return data


def demonstrate_rolling_operations():
    """Demonstrate configurable rolling window operations"""
    print("=" * 60)
    print("ROLLING WINDOW OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(100)
    engine = FeatureEngine()
    
    # Configure rolling operations with multiple lookback periods
    rolling_config = {
        'rolling_operations': {
            'enabled': True,
            'windows': [5, 10, 20, 50],  # Multiple lookback periods
            'operations': ['mean', 'std', 'min', 'max', 'rsi', 'momentum', 'volatility'],
            'columns': ['close', 'volume'],
            'quantiles': [0.25, 0.75]  # For quantile operations
        }
    }
    
    # Apply rolling operations
    result = engine.compute_features(data, rolling_config)
    
    print(f"Original data shape: {data.shape}")
    print(f"Data with rolling features shape: {result.shape}")
    print(f"Added {result.shape[1] - data.shape[1]} rolling window features")
    
    # Show some examples
    rolling_features = [col for col in result.columns if 'rolling' in col]
    print(f"\nSample rolling features created:")
    for feature in rolling_features[:10]:
        print(f"  - {feature}")
    
    # Demonstrate different window sizes for same operation
    print(f"\nClose price rolling means with different windows:")
    print(f"5-day mean (last 5 values): {result['close_rolling_5_mean'].iloc[-1]:.2f}")
    print(f"20-day mean (last 5 values): {result['close_rolling_20_mean'].iloc[-1]:.2f}")
    print(f"50-day mean (last 5 values): {result['close_rolling_50_mean'].iloc[-1]:.2f}")
    
    return result


def demonstrate_data_imputation():
    """Demonstrate multiple data imputation strategies"""
    print("\n" + "=" * 60)
    print("DATA IMPUTATION STRATEGIES DEMONSTRATION")
    print("=" * 60)
    
    # Create data with missing values
    data = create_sample_data(100)
    engine = FeatureEngine()
    
    print(f"Original data missing values:")
    print(data.isnull().sum())
    
    # Test different imputation strategies
    strategies = [
        'forward_fill',
        'linear_interpolation',
        'mean_imputation',
        'knn_imputation',
        'seasonal_imputation'
    ]
    
    results = {}
    for strategy in strategies:
        try:
            config = {
                'imputation': {
                    'enabled': True,
                    'strategy': strategy,
                    'params': {'n_neighbors': 3} if strategy == 'knn_imputation' else {}
                }
            }
            
            imputed_data = engine.compute_features(data, config)
            results[strategy] = imputed_data.isnull().sum().sum()
            
            print(f"\n{strategy.replace('_', ' ').title()}:")
            print(f"  Remaining missing values: {results[strategy]}")
            
            if strategy == 'linear_interpolation':
                # Show example of interpolated values
                original_missing = data['close'].isnull()
                if original_missing.any():
                    first_missing_idx = original_missing.idxmax()
                    print(f"  Example: Original NaN at {first_missing_idx} -> {imputed_data.loc[first_missing_idx, 'close']:.2f}")
                    
        except Exception as e:
            print(f"  Error with {strategy}: {str(e)}")
    
    # Compare strategies
    print(f"\nImputation Strategy Comparison:")
    for strategy, remaining_missing in results.items():
        print(f"  {strategy}: {remaining_missing} missing values remaining")


def demonstrate_bias_detection():
    """Demonstrate look-ahead bias detection and prevention"""
    print("\n" + "=" * 60)
    print("LOOK-AHEAD BIAS DETECTION DEMONSTRATION")
    print("=" * 60)
    
    data = create_sample_data(100)
    engine = FeatureEngine()
    
    # Create features with potential bias
    biased_features = data.copy()
    
    # Add a feature that uses future information (look-ahead bias)
    biased_features['future_return'] = data['close'].pct_change(fill_method=None).shift(-5)  # Future data!
    biased_features['perfect_predictor'] = data['close'].shift(-1)  # Tomorrow's price
    biased_features['suspicious_stable'] = 42.0  # Unrealistically stable
    
    print("Testing bias detection on features with known issues...")
    
    # Detect bias
    bias_report = engine.detect_look_ahead_bias(data, biased_features)
    
    print(f"Look-ahead bias detected: {bias_report.look_ahead_bias_detected}")
    print(f"Number of bias warnings: {len(bias_report.bias_details)}")
    print(f"Number of general warnings: {len(bias_report.warnings)}")
    
    if bias_report.bias_details:
        print("\nBias detection details:")
        for detail in bias_report.bias_details[:3]:  # Show first 3
            print(f"  - {detail}")
    
    # Demonstrate bias prevention in configuration
    print("\nDemonstrating bias prevention in feature configuration...")
    
    potentially_biased_config = {
        'rolling_operations': {
            'enabled': True,
            'windows': [10],
            'operations': ['mean'],
            'columns': ['close']
        }
    }
    
    safe_config = engine.bias_detector.prevent_bias(data, potentially_biased_config)
    
    print("Added temporal constraints to prevent bias:")
    print(f"  Max future periods: {safe_config['temporal_constraints']['max_future_periods']}")
    print(f"  Min history periods: {safe_config['temporal_constraints']['min_history_periods']}")


def demonstrate_feature_selection():
    """Demonstrate feature selection and dimensionality reduction"""
    print("\n" + "=" * 60)
    print("FEATURE SELECTION AND DIMENSIONALITY REDUCTION")
    print("=" * 60)
    
    data = create_sample_data(200)
    engine = FeatureEngine()
    
    # Create many features first
    feature_config = {
        'rolling_operations': {
            'enabled': True,
            'windows': [5, 10, 20, 50],
            'operations': ['mean', 'std', 'min', 'max', 'rsi'],
            'columns': ['close', 'volume']
        }
    }
    
    features = engine.compute_features(data, feature_config)
    
    # Create a target variable (future returns)
    target = features['close'].pct_change(5, fill_method=None).shift(-5)  # 5-day future return
    
    print(f"Created {features.shape[1]} features from original data")
    
    # Test different feature selection methods
    selection_methods = ['univariate_f', 'mutual_info', 'rfe', 'correlation_filter']
    
    for method in selection_methods:
        try:
            selected_features, feature_names = engine.select_features(
                features.drop('close', axis=1),  # Don't include target in features
                target.dropna(),
                method=method,
                n_features=10
            )
            
            print(f"\n{method.replace('_', ' ').title()} Selection:")
            print(f"  Selected {len(feature_names)} features")
            print(f"  Top 5 features: {feature_names[:5]}")
            
        except Exception as e:
            print(f"  Error with {method}: {str(e)}")
    
    # Demonstrate dimensionality reduction
    print(f"\nDimensionality Reduction:")
    
    reduction_methods = ['pca', 'ica']
    
    for method in reduction_methods:
        try:
            reduced_features, model = engine.reduce_dimensions(
                features.select_dtypes(include=[np.number]),
                method=method,
                n_components=5
            )
            
            print(f"\n{method.upper()} Reduction:")
            print(f"  Reduced to {reduced_features.shape[1]} components")
            print(f"  Component names: {list(reduced_features.columns)}")
            
            if method == 'pca' and hasattr(model, 'explained_variance_ratio_'):
                print(f"  Explained variance ratio: {model.explained_variance_ratio_[:3]}")
                
        except Exception as e:
            print(f"  Error with {method}: {str(e)}")


def demonstrate_comprehensive_pipeline():
    """Demonstrate a comprehensive feature engineering pipeline"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    data = create_sample_data(300)
    engine = FeatureEngine()
    
    # Comprehensive configuration
    comprehensive_config = {
        'imputation': {
            'enabled': True,
            'strategy': 'linear_interpolation'
        },
        'rolling_operations': {
            'enabled': True,
            'windows': [10, 20, 50],
            'operations': ['mean', 'std', 'rsi', 'momentum'],
            'columns': ['close', 'volume']
        },
        'indicators': {
            'ADX': {'period': 14},
            'Aroon': {'period': 25}
        },
        'cross_sectional': {
            'relative_strength': True,
            'sector_column': 'sector'
        },
        'bias_prevention': {
            'enabled': True
        }
    }
    
    print("Applying comprehensive feature engineering pipeline...")
    
    # Apply the full pipeline
    result = engine.compute_features(data, comprehensive_config)
    
    print(f"Original data shape: {data.shape}")
    print(f"Final feature set shape: {result.shape}")
    print(f"Total features created: {result.shape[1] - data.shape[1]}")
    
    # Validate the results
    validation_report = engine.validate_features(result, data)
    
    print(f"\nFeature Validation Report:")
    print(f"  Valid features: {validation_report.is_valid}")
    print(f"  Total features: {validation_report.statistics['total_features']}")
    print(f"  Numeric features: {validation_report.statistics['numeric_features']}")
    print(f"  Missing data %: {validation_report.statistics['missing_data_percentage']:.2f}%")
    print(f"  Memory usage: {validation_report.statistics['memory_usage_mb']:.2f} MB")
    print(f"  Look-ahead bias detected: {validation_report.look_ahead_bias_detected}")
    
    if validation_report.errors:
        print(f"  Errors: {len(validation_report.errors)}")
        for error in validation_report.errors[:3]:
            print(f"    - {error}")
    
    if validation_report.warnings:
        print(f"  Warnings: {len(validation_report.warnings)}")
        for warning in validation_report.warnings[:3]:
            print(f"    - {warning}")
    
    # Create feature summary
    summary = engine.create_feature_summary(result)
    
    print(f"\nFeature Summary:")
    print(f"  Feature density: {summary['feature_density']:.3f}")
    print(f"  High correlation pairs: {summary['high_correlation_pairs']}")
    
    return result, validation_report


def demonstrate_feature_importance():
    """Demonstrate feature importance calculation"""
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    data = create_sample_data(200)
    engine = FeatureEngine()
    
    # Create features
    config = {
        'rolling_operations': {
            'enabled': True,
            'windows': [10, 20],
            'operations': ['mean', 'std', 'rsi'],
            'columns': ['close']
        }
    }
    
    features = engine.compute_features(data, config)
    
    # Create target (future returns)
    target = features['close'].pct_change(5, fill_method=None).shift(-5).dropna()
    
    # Calculate feature importance
    importance_rf = engine.get_feature_importance(
        features.drop('close', axis=1), 
        target, 
        method='random_forest'
    )
    
    importance_corr = engine.get_feature_importance(
        features.drop('close', axis=1), 
        target, 
        method='correlation'
    )
    
    print("Top 10 Most Important Features (Random Forest):")
    for i, (feature, importance) in enumerate(importance_rf.head(10).items()):
        print(f"  {i+1:2d}. {feature}: {importance:.4f}")
    
    print("\nTop 10 Most Important Features (Correlation):")
    for i, (feature, importance) in enumerate(importance_corr.head(10).items()):
        print(f"  {i+1:2d}. {feature}: {importance:.4f}")


def main():
    """Run all demonstrations"""
    print("Advanced Feature Engineering Capabilities Demo")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_rolling_operations()
        demonstrate_data_imputation()
        demonstrate_bias_detection()
        demonstrate_feature_selection()
        result, report = demonstrate_comprehensive_pipeline()
        demonstrate_feature_importance()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey capabilities demonstrated:")
        print("✓ Configurable rolling window operations with multiple lookback periods")
        print("✓ Multiple data imputation strategies (forward-fill, interpolation, model-based)")
        print("✓ Look-ahead bias detection and prevention system")
        print("✓ Feature selection tools (univariate, mutual info, RFE, correlation filter)")
        print("✓ Dimensionality reduction (PCA, ICA)")
        print("✓ Comprehensive feature validation and reporting")
        print("✓ Feature importance analysis")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()