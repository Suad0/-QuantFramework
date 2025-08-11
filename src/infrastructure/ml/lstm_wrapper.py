"""
LSTM wrapper for financial time series prediction.

This module provides a scikit-learn compatible wrapper for LSTM models
specifically designed for financial time series data.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
import warnings
from typing import Optional, Tuple, Any, Dict

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, MultiHeadAttention, LayerNormalization, Add
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class LSTMWrapper(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible LSTM wrapper for financial time series.
    
    This wrapper provides a simple interface for LSTM models that can be used
    with the ML framework while handling the unique aspects of time series data.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        lstm_units: int = 50,
        dropout_rate: float = 0.2,
        dense_units: int = 25,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        random_state: Optional[int] = None,
        use_attention: bool = False,
        attention_heads: int = 8,
        use_multihead_attention: bool = False,
        l1_reg: float = 0.0,
        l2_reg: float = 0.01,
        use_layer_norm: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize LSTM wrapper.
        
        Args:
            sequence_length: Length of input sequences
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            dense_units: Number of dense layer units
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            validation_split: Fraction of data for validation
            early_stopping_patience: Patience for early stopping
            random_state: Random seed for reproducibility
            use_attention: Whether to use attention mechanism
            attention_heads: Number of attention heads for multi-head attention
            use_multihead_attention: Whether to use multi-head attention
            l1_reg: L1 regularization strength
            l2_reg: L2 regularization strength
            use_layer_norm: Whether to use layer normalization
            use_residual: Whether to use residual connections
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        self.use_multihead_attention = use_multihead_attention
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        self.model_ = None
        self.scaler_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM. Install with: pip install tensorflow")
        
        # Set random seeds for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
            if TENSORFLOW_AVAILABLE:
                tf.random.set_seed(random_state)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LSTMWrapper':
        """
        Fit the LSTM model to training data.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        self.feature_names_ = X.columns.tolist()
        
        # Prepare data for LSTM
        X_sequences, y_sequences = self._prepare_sequences(X, y)
        
        if len(X_sequences) == 0:
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}")
        
        # Scale the data
        self.scaler_ = MinMaxScaler()
        n_samples, n_timesteps, n_features = X_sequences.shape
        X_scaled = self.scaler_.fit_transform(
            X_sequences.reshape(-1, n_features)
        ).reshape(n_samples, n_timesteps, n_features)
        
        # Build model
        self.model_ = self._build_model(n_features)
        
        # Set up callbacks
        callbacks = []
        if self.early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Add learning rate reduction
        lr_reducer = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.early_stopping_patience // 2,
            min_lr=1e-7,
            verbose=0
        )
        callbacks.append(lr_reducer)
        
        # Train model
        history = self.model_.fit(
            X_scaled,
            y_sequences,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted LSTM model.
        
        Args:
            X: Feature data
            
        Returns:
            Predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # For prediction, we need to handle the sequence creation differently
        # We'll use a sliding window approach
        predictions = []
        
        # Convert to numpy for easier manipulation
        X_values = X.values
        
        # Create sequences for prediction
        for i in range(len(X_values)):
            if i < self.sequence_length - 1:
                # For early samples, pad with the first available values
                sequence = np.vstack([
                    np.tile(X_values[0], (self.sequence_length - i - 1, 1)),
                    X_values[:i + 1]
                ])
            else:
                # Normal case: use the last sequence_length samples
                sequence = X_values[i - self.sequence_length + 1:i + 1]
            
            # Scale the sequence
            sequence_scaled = self.scaler_.transform(sequence)
            
            # Make prediction
            pred = self.model_.predict(
                sequence_scaled.reshape(1, self.sequence_length, -1),
                verbose=0
            )
            predictions.append(pred[0, 0])
        
        return np.array(predictions)
    
    def _prepare_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_values = X.values
        y_values = y.values
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_values)):
            X_sequences.append(X_values[i - self.sequence_length:i])
            y_sequences.append(y_values[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _build_model(self, n_features: int):
        """
        Build the LSTM model architecture with optional attention mechanisms.
        
        Args:
            n_features: Number of input features
            
        Returns:
            Compiled Keras model
        """
        if self.use_attention or self.use_multihead_attention:
            return self._build_attention_model(n_features)
        else:
            return self._build_standard_model(n_features)
    
    def _build_standard_model(self, n_features: int):
        """Build standard LSTM model without attention."""
        regularizer = l1_l2(l1=self.l1_reg, l2=self.l2_reg) if (self.l1_reg > 0 or self.l2_reg > 0) else None
        
        model = Sequential([
            LSTM(
                self.lstm_units,
                return_sequences=True,
                input_shape=(self.sequence_length, n_features),
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer
            ),
            Dropout(self.dropout_rate),
            
            LSTM(
                self.lstm_units, 
                return_sequences=False,
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer
            ),
            Dropout(self.dropout_rate),
            
            Dense(
                self.dense_units, 
                activation='relu',
                kernel_regularizer=regularizer
            ),
            Dropout(self.dropout_rate),
            
            Dense(1, kernel_regularizer=regularizer)
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _build_attention_model(self, n_features: int):
        """Build LSTM model with attention mechanisms."""
        regularizer = l1_l2(l1=self.l1_reg, l2=self.l2_reg) if (self.l1_reg > 0 or self.l2_reg > 0) else None
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, n_features))
        
        # First LSTM layer
        lstm1 = LSTM(
            self.lstm_units,
            return_sequences=True,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer
        )(inputs)
        
        if self.use_layer_norm:
            lstm1 = LayerNormalization()(lstm1)
        
        lstm1 = Dropout(self.dropout_rate)(lstm1)
        
        # Second LSTM layer
        lstm2 = LSTM(
            self.lstm_units,
            return_sequences=True,
            kernel_regularizer=regularizer,
            recurrent_regularizer=regularizer
        )(lstm1)
        
        if self.use_layer_norm:
            lstm2 = LayerNormalization()(lstm2)
        
        lstm2 = Dropout(self.dropout_rate)(lstm2)
        
        # Residual connection
        if self.use_residual and lstm1.shape[-1] == lstm2.shape[-1]:
            lstm2 = Add()([lstm1, lstm2])
        
        # Attention mechanism
        if self.use_multihead_attention:
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=self.lstm_units // self.attention_heads
            )(lstm2, lstm2)
        else:
            # Simple attention
            attention_output = Attention()([lstm2, lstm2])
        
        if self.use_layer_norm:
            attention_output = LayerNormalization()(attention_output)
        
        attention_output = Dropout(self.dropout_rate)(attention_output)
        
        # Residual connection for attention
        if self.use_residual:
            attention_output = Add()([lstm2, attention_output])
        
        # Global average pooling to reduce sequence dimension
        from tensorflow.keras.layers import GlobalAveragePooling1D
        pooled = GlobalAveragePooling1D()(attention_output)
        
        # Dense layers
        dense1 = Dense(
            self.dense_units,
            activation='relu',
            kernel_regularizer=regularizer
        )(pooled)
        
        if self.use_layer_norm:
            dense1 = LayerNormalization()(dense1)
        
        dense1 = Dropout(self.dropout_rate)(dense1)
        
        # Output layer
        outputs = Dense(1, kernel_regularizer=regularizer)(dense1)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'dense_units': self.dense_units,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'random_state': self.random_state,
            'use_attention': self.use_attention,
            'attention_heads': self.attention_heads,
            'use_multihead_attention': self.use_multihead_attention,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'use_layer_norm': self.use_layer_norm,
            'use_residual': self.use_residual
        }
    
    def set_params(self, **params) -> 'LSTMWrapper':
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self


class SimpleLSTMWrapper(BaseEstimator, RegressorMixin):
    """
    Simplified LSTM wrapper that falls back to neural network if TensorFlow is not available.
    """
    
    def __init__(self, **params):
        """Initialize with fallback to MLPRegressor if TensorFlow not available."""
        if TENSORFLOW_AVAILABLE:
            self.model = LSTMWrapper(**params)
        else:
            warnings.warn("TensorFlow not available, falling back to MLPRegressor")
            from sklearn.neural_network import MLPRegressor
            # Map some LSTM params to MLP params
            mlp_params = {
                'hidden_layer_sizes': (params.get('lstm_units', 50), params.get('dense_units', 25)),
                'max_iter': params.get('epochs', 100),
                'learning_rate_init': params.get('learning_rate', 0.001),
                'random_state': params.get('random_state')
            }
            # Remove None values
            mlp_params = {k: v for k, v in mlp_params.items() if v is not None}
            self.model = MLPRegressor(**mlp_params)
    
    def fit(self, X, y):
        """Fit the model."""
        return self.model.fit(X, y)
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters."""
        return self.model.get_params(deep)
    
    def set_params(self, **params):
        """Set parameters."""
        return self.model.set_params(**params)