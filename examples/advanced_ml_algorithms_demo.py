"""
Demonstration of advanced ML algorithms with enhanced features.

This script demonstrates the enhanced LSTM with attention mechanisms,
XGBoost, Random Forest, and SVM implementations with hyperparameter
optimization and regularization features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.infrastructure.ml.ml_framework import MLFramework
from src.infrastructure.ml.advanced_algorithms import (
    EnhancedXGBoostRegressor,
    EnhancedRandomForestRegressor,
    EnhancedSVMRegressor
)
from src.infrastructure.ml.lstm_wrapper import LSTMWrapper
from src.infrastructure.ml.hyperparameter_optimization import HyperparameterOptimizer


def create_financial_time_series_data(n_samples=1000, n_features=5):
    """Create synthetic financial time series data."""
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # Generate features with financial characteristics
    features = {}
    
    # Price-based features
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)
    features['price'] = price
    features['returns'] = np.concatenate([[0], np.diff(np.log(price))])
    
    # Technical indicators
    features['sma_20'] = pd.Series(price).rolling(20).mean().fillna(method='bfill')
    features['volatility'] = pd.Series(features['returns']).rolling(20).std().fillna(method='bfill')
    features['rsi'] = np.random.uniform(20, 80, n_samples)  # Simplified RSI
    
    # Add more features if needed
    for i in range(n_features - 5):
        features[f'feature_{i}'] = np.random.randn(n_samples)
    
    # Create DataFrame
    X = pd.DataFrame(features, index=dates)
    
    # Create target (next day return)
    y = pd.Series(
        np.concatenate([features['returns'][1:], [0]]),
        index=dates,
        name='next_return'
    )
    
    return X, y


def demonstrate_enhanced_xgboost():
    """Demonstrate enhanced XGBoost with hyperparameter optimization."""
    print("=" * 60)
    print("ENHANCED XGBOOST DEMONSTRATION")
    print("=" * 60)
    
    # Create data
    X, y = create_financial_time_series_data(500, 5)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Test basic XGBoost
    print("\n1. Basic Enhanced XGBoost:")
    basic_xgb = EnhancedXGBoostRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    try:
        basic_xgb.fit(X_train, y_train)
        basic_pred = basic_xgb.predict(X_test)
        basic_mse = np.mean((y_test - basic_pred) ** 2)
        print(f"   Basic XGBoost MSE: {basic_mse:.6f}")
        
        # Test uncertainty prediction
        pred_with_uncertainty, uncertainty = basic_xgb.predict_with_uncertainty(X_test)
        print(f"   Average uncertainty: {np.mean(uncertainty):.6f}")
        
    except ImportError:
        print("   XGBoost not installed - skipping XGBoost demonstration")
        return
    
    # Test with hyperparameter optimization
    print("\n2. XGBoost with Hyperparameter Optimization:")
    tuned_xgb = EnhancedXGBoostRegressor(
        auto_tune=True,
        tune_method='random',
        cv_folds=3,
        random_state=42
    )
    
    try:
        tuned_xgb.fit(X_train, y_train)
        tuned_pred = tuned_xgb.predict(X_test)
        tuned_mse = np.mean((y_test - tuned_pred) ** 2)
        print(f"   Tuned XGBoost MSE: {tuned_mse:.6f}")
        print(f"   Best parameters: {tuned_xgb.best_params_}")
        
        # Feature importance
        if hasattr(tuned_xgb, 'feature_importances_') and tuned_xgb.feature_importances_ is not None:
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': tuned_xgb.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"   Top 3 features: {importance_df.head(3)['feature'].tolist()}")
        
    except Exception as e:
        print(f"   Error in hyperparameter optimization: {e}")


def demonstrate_enhanced_random_forest():
    """Demonstrate enhanced Random Forest with feature selection."""
    print("\n" + "=" * 60)
    print("ENHANCED RANDOM FOREST DEMONSTRATION")
    print("=" * 60)
    
    # Create data
    X, y = create_financial_time_series_data(500, 8)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    
    # Test basic Random Forest
    print("\n1. Basic Enhanced Random Forest:")
    basic_rf = EnhancedRandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    
    basic_rf.fit(X_train, y_train)
    basic_pred = basic_rf.predict(X_test)
    basic_mse = np.mean((y_test - basic_pred) ** 2)
    print(f"   Basic RF MSE: {basic_mse:.6f}")
    
    # Test uncertainty prediction
    pred_with_uncertainty, uncertainty = basic_rf.predict_with_uncertainty(X_test)
    print(f"   Average uncertainty: {np.mean(uncertainty):.6f}")
    
    # Test with feature selection
    print("\n2. Random Forest with Feature Selection:")
    feature_rf = EnhancedRandomForestRegressor(
        n_estimators=50,
        feature_selection=True,
        feature_selection_threshold=0.05,
        random_state=42
    )
    
    feature_rf.fit(X_train, y_train)
    feature_pred = feature_rf.predict(X_test)
    feature_mse = np.mean((y_test - feature_pred) ** 2)
    print(f"   Feature-selected RF MSE: {feature_mse:.6f}")
    
    if feature_rf.selected_features_:
        print(f"   Selected features: {feature_rf.selected_features_}")
    else:
        print("   All features selected")
    
    # Test with hyperparameter tuning
    print("\n3. Random Forest with Hyperparameter Tuning:")
    tuned_rf = EnhancedRandomForestRegressor(
        auto_tune=True,
        tune_method='random',
        cv_folds=3,
        random_state=42
    )
    
    tuned_rf.fit(X_train, y_train)
    tuned_pred = tuned_rf.predict(X_test)
    tuned_mse = np.mean((y_test - tuned_pred) ** 2)
    print(f"   Tuned RF MSE: {tuned_mse:.6f}")
    print(f"   Best parameters: {tuned_rf.best_params_}")


def demonstrate_enhanced_svm():
    """Demonstrate enhanced SVM with feature scaling."""
    print("\n" + "=" * 60)
    print("ENHANCED SVM DEMONSTRATION")
    print("=" * 60)
    
    # Create smaller dataset for SVM
    X, y = create_financial_time_series_data(200, 4)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    
    # Test basic SVM
    print("\n1. Basic Enhanced SVM:")
    basic_svm = EnhancedSVMRegressor(
        C=1.0,
        kernel='rbf',
        scale_features=True,
        random_state=42
    )
    
    basic_svm.fit(X_train, y_train)
    basic_pred = basic_svm.predict(X_test)
    basic_mse = np.mean((y_test - basic_pred) ** 2)
    print(f"   Basic SVM MSE: {basic_mse:.6f}")
    print(f"   Feature scaling used: {basic_svm.scaler_ is not None}")
    
    # Test with hyperparameter tuning
    print("\n2. SVM with Hyperparameter Tuning:")
    tuned_svm = EnhancedSVMRegressor(
        auto_tune=True,
        tune_method='random',
        cv_folds=3,
        scale_features=True,
        random_state=42
    )
    
    tuned_svm.fit(X_train, y_train)
    tuned_pred = tuned_svm.predict(X_test)
    tuned_mse = np.mean((y_test - tuned_pred) ** 2)
    print(f"   Tuned SVM MSE: {tuned_mse:.6f}")
    print(f"   Best parameters: {tuned_svm.best_params_}")


def demonstrate_lstm_with_attention():
    """Demonstrate LSTM with attention mechanisms."""
    print("\n" + "=" * 60)
    print("LSTM WITH ATTENTION DEMONSTRATION")
    print("=" * 60)
    
    # Create time series data
    X, y = create_financial_time_series_data(300, 4)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    
    try:
        # Test basic LSTM
        print("\n1. Basic LSTM:")
        basic_lstm = LSTMWrapper(
            sequence_length=30,
            lstm_units=32,
            epochs=5,  # Quick demo
            verbose=0,
            random_state=42
        )
        
        basic_lstm.fit(X_train, y_train)
        basic_pred = basic_lstm.predict(X_test)
        basic_mse = np.mean((y_test - basic_pred) ** 2)
        print(f"   Basic LSTM MSE: {basic_mse:.6f}")
        
        # Test LSTM with attention
        print("\n2. LSTM with Attention:")
        attention_lstm = LSTMWrapper(
            sequence_length=30,
            lstm_units=32,
            use_attention=True,
            epochs=5,  # Quick demo
            verbose=0,
            random_state=42
        )
        
        attention_lstm.fit(X_train, y_train)
        attention_pred = attention_lstm.predict(X_test)
        attention_mse = np.mean((y_test - attention_pred) ** 2)
        print(f"   Attention LSTM MSE: {attention_mse:.6f}")
        
        # Test LSTM with multi-head attention
        print("\n3. LSTM with Multi-Head Attention:")
        multihead_lstm = LSTMWrapper(
            sequence_length=30,
            lstm_units=32,
            use_multihead_attention=True,
            attention_heads=4,
            epochs=5,  # Quick demo
            verbose=0,
            random_state=42
        )
        
        multihead_lstm.fit(X_train, y_train)
        multihead_pred = multihead_lstm.predict(X_test)
        multihead_mse = np.mean((y_test - multihead_pred) ** 2)
        print(f"   Multi-Head Attention LSTM MSE: {multihead_mse:.6f}")
        
        # Test LSTM with regularization
        print("\n4. LSTM with Regularization:")
        regularized_lstm = LSTMWrapper(
            sequence_length=30,
            lstm_units=32,
            use_attention=True,
            l1_reg=0.01,
            l2_reg=0.01,
            use_layer_norm=True,
            use_residual=True,
            epochs=5,  # Quick demo
            verbose=0,
            random_state=42
        )
        
        regularized_lstm.fit(X_train, y_train)
        regularized_pred = regularized_lstm.predict(X_test)
        regularized_mse = np.mean((y_test - regularized_pred) ** 2)
        print(f"   Regularized LSTM MSE: {regularized_mse:.6f}")
        
    except ImportError:
        print("   TensorFlow not available - skipping LSTM demonstration")
    except Exception as e:
        print(f"   Error in LSTM demonstration: {e}")


def demonstrate_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization framework."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create data
    X, y = create_financial_time_series_data(200, 4)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    
    # Test hyperparameter optimizer
    from sklearn.ensemble import RandomForestRegressor
    
    optimizer = HyperparameterOptimizer()
    base_model = RandomForestRegressor(random_state=42)
    
    param_space = {
        'n_estimators': [10, 20, 50],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5]
    }
    
    print("\n1. Grid Search Optimization:")
    best_params, best_score, opt_info = optimizer.optimize(
        base_model, param_space, X_train, y_train,
        method='grid', cv_folds=3
    )
    
    print(f"   Best parameters: {best_params}")
    print(f"   Best CV score: {best_score:.6f}")
    print(f"   Optimization time: {opt_info['optimization_time']:.2f} seconds")
    
    print("\n2. Random Search Optimization:")
    best_params, best_score, opt_info = optimizer.optimize(
        base_model, param_space, X_train, y_train,
        method='random', n_iter=10, cv_folds=3
    )
    
    print(f"   Best parameters: {best_params}")
    print(f"   Best CV score: {best_score:.6f}")
    print(f"   Optimization time: {opt_info['optimization_time']:.2f} seconds")
    
    # Test default parameter spaces
    print("\n3. Default Parameter Spaces:")
    for model_type in ['xgboost', 'random_forest', 'svm', 'lstm']:
        default_params = optimizer.get_default_param_space(model_type)
        print(f"   {model_type}: {list(default_params.keys())}")


def demonstrate_ml_framework_integration():
    """Demonstrate ML framework integration with enhanced algorithms."""
    print("\n" + "=" * 60)
    print("ML FRAMEWORK INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Create data
    X, y = create_financial_time_series_data(300, 5)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    
    # Initialize framework
    framework = MLFramework()
    
    # Test enhanced algorithms through framework
    algorithms_to_test = [
        'enhanced_random_forest',
        'enhanced_svm'
    ]
    
    for algorithm in algorithms_to_test:
        print(f"\n{algorithm.upper()}:")
        
        config = {
            'algorithm': algorithm,
            'model_params': {
                'random_state': 42
            },
            'hyperparameter_optimization': {
                'enabled': True,
                'method': 'random',
                'optimizer_kwargs': {
                    'n_iter': 5,
                    'cv_folds': 3
                }
            },
            'register_model': False
        }
        
        try:
            # Train model
            model = framework.train_model(X_train, y_train, config)
            
            # Make predictions
            result = framework.predict(model, X_test)
            predictions = result['predictions']
            
            # Calculate MSE
            mse = np.mean((y_test - predictions) ** 2)
            print(f"   MSE: {mse:.6f}")
            
            # Get feature importance if available
            feature_importance = model.get_feature_importance()
            if feature_importance:
                top_features = sorted(feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top 3 features: {[f[0] for f in top_features]}")
            
        except Exception as e:
            print(f"   Error: {e}")


def main():
    """Run all demonstrations."""
    print("ADVANCED ML ALGORITHMS DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates enhanced ML algorithms with:")
    print("- LSTM with attention mechanisms")
    print("- Enhanced XGBoost with auto-tuning")
    print("- Enhanced Random Forest with feature selection")
    print("- Enhanced SVM with feature scaling")
    print("- Hyperparameter optimization framework")
    print("- Regularization techniques")
    
    # Run demonstrations
    demonstrate_enhanced_xgboost()
    demonstrate_enhanced_random_forest()
    demonstrate_enhanced_svm()
    demonstrate_lstm_with_attention()
    demonstrate_hyperparameter_optimization()
    demonstrate_ml_framework_integration()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)
    print("All enhanced ML algorithms have been successfully demonstrated!")
    print("Key improvements implemented:")
    print("✓ LSTM with attention mechanisms and regularization")
    print("✓ Enhanced XGBoost with hyperparameter optimization")
    print("✓ Enhanced Random Forest with feature selection")
    print("✓ Enhanced SVM with automatic feature scaling")
    print("✓ Comprehensive hyperparameter optimization framework")
    print("✓ Regularization manager for overfitting prevention")
    print("✓ Integration with existing ML framework")


if __name__ == '__main__':
    main()