import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMPredictor:
    def __init__(self, sequence_length=60, lstm_units=128, dropout_rate=0.3, learning_rate=0.001,
                 device='mps' if torch.mps.is_available() else 'cpu'):
        """
        Initialize LSTM predictor for financial time series using PyTorch.

        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units in each layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.device = device
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_target = MinMaxScaler()
        self.feature_columns = None
        self.is_trained = False

    def prepare_data(self, df, target_col, feature_cols=None, test_size=0.2, batch_size=32):
        """
        Prepare data for LSTM training with proper scaling and sequencing.

        Args:
            df: DataFrame with time series data
            target_col: Column name for prediction target
            feature_cols: List of feature column names (if None, uses all numeric columns)
            test_size: Proportion of data for testing
            batch_size: Batch size for DataLoader

        Returns:
            Tuple of (train_loader, test_loader, y_train, y_test)
        """
        if target_col not in df.columns:
            available_cols = [col for col in df.columns if 'Adj Close' in col or col == 'Adj Close']
            if available_cols:
                target_col = available_cols[0]
                print(f"Target column not found, using: {target_col}")
            else:
                raise ValueError(f"Target column '{target_col}' not found and no suitable alternatives available.")

        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != target_col]

        feature_cols = [col for col in feature_cols if col in df.columns and df[col].notna().sum() / len(df) > 0.7]

        if not feature_cols:
            raise ValueError("No suitable feature columns found after filtering.")

        self.feature_columns = feature_cols
        print(f"Using {len(feature_cols)} features: {feature_cols[:5]}..." if len(
            feature_cols) > 5 else f"Using features: {feature_cols}")

        df_clean = df[feature_cols + [target_col]].dropna()

        if len(df_clean) < self.sequence_length * 2:
            raise ValueError(f"Not enough data points. Need at least {self.sequence_length * 2}, got {len(df_clean)}")

        split_index = int(len(df_clean) * (1 - test_size))

        train_data = df_clean.iloc[:split_index]
        test_data = df_clean.iloc[split_index:]

        X_train_scaled = self.scaler_features.fit_transform(train_data[feature_cols])
        X_test_scaled = self.scaler_features.transform(test_data[feature_cols])

        y_train_scaled = self.scaler_target.fit_transform(train_data[[target_col]])
        y_test_scaled = self.scaler_target.transform(test_data[[target_col]])

        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train_scaled.flatten())
        X_test_seq, y_test_seq = self._create_sequences(X_test_scaled, y_test_scaled.flatten())

        train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
        test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Training sequences: {X_train_seq.shape}, Testing sequences: {X_test_seq.shape}")

        return train_loader, test_loader, y_train_seq, y_test_seq

    def _create_sequences(self, X, y):
        """Create sequences for LSTM input."""
        X_seq, y_seq = [], []

        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)

    def build_model(self, input_shape):
        """
        Build advanced LSTM model with attention mechanism using PyTorch.

        Args:
            input_shape: Shape of input data (sequence_length, n_features)
        """

        class LSTMModel(nn.Module):
            def __init__(self, input_size, lstm_units, dropout_rate):
                super(LSTMModel, self).__init__()
                self.lstm1 = nn.LSTM(input_size, lstm_units, batch_first=True)
                self.bn1 = nn.BatchNorm1d(lstm_units)
                self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)
                self.bn2 = nn.BatchNorm1d(lstm_units // 2)

                # Simplified attention mechanism
                self.attention = nn.Linear(lstm_units // 2, 1)
                self.lstm3 = nn.LSTM(lstm_units // 2, lstm_units // 4, batch_first=True)
                self.fc1 = nn.Linear(lstm_units // 4 + lstm_units // 2, 64)
                self.dropout1 = nn.Dropout(dropout_rate)
                self.bn3 = nn.BatchNorm1d(64)
                self.fc2 = nn.Linear(64, 32)
                self.dropout2 = nn.Dropout(dropout_rate / 2)
                self.fc3 = nn.Linear(32, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm1(x)
                lstm_out = self.bn1(lstm_out.permute(0, 2, 1)).permute(0, 2, 1)

                lstm_out, _ = self.lstm2(lstm_out)
                lstm_out = self.bn2(lstm_out.permute(0, 2, 1)).permute(0, 2, 1)

                # Attention mechanism
                attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context_vector = torch.sum(lstm_out * attention_weights, dim=1)

                lstm_out, _ = self.lstm3(lstm_out)
                lstm_out = lstm_out[:, -1, :]  # Take the last time step

                combined = torch.cat([context_vector, lstm_out], dim=1)

                x = torch.relu(self.fc1(combined))
                x = self.dropout1(x)
                x = self.bn3(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout2(x)
                x = self.fc3(x)
                return x

        self.model = LSTMModel(input_shape[-1], self.lstm_units, self.dropout_rate).to(self.device)

        return self.model

    def train(self, train_loader, test_loader, epochs=100, verbose=1):
        """
        Train the LSTM model with early stopping and learning rate reduction.

        Args:
            train_loader, test_loader: DataLoader for training and testing data
            epochs: Maximum number of epochs
            verbose: Verbosity level

        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model must be built before training.")

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-7)
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            self.model.eval()
            val_loss = 0
            val_mae = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    output = self.model(X_batch)
                    val_loss += criterion(output.squeeze(), y_batch).item() * X_batch.size(0)
                    val_mae += torch.mean(torch.abs(output.squeeze() - y_batch)).item() * X_batch.size(0)

            val_loss /= len(test_loader.dataset)
            val_mae /= len(test_loader.dataset)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model (simplified for example)
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    self.model.load_state_dict(best_model_state)
                    break

        self.is_trained = True
        return history

    def predict(self, X):
        """Make predictions and inverse transform to original scale."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions.")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        predictions = self.scaler_target.inverse_transform(predictions_scaled)
        return predictions.flatten()

    def evaluate(self, X_test, y_test_original):
        """
        Evaluate model performance with various metrics.

        Args:
            X_test: Test features
            y_test_original: Original scale test targets

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)

        mse = mean_squared_error(y_test_original, predictions)
        mae = mean_absolute_error(y_test_original, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test_original - predictions) / y_test_original)) * 100

        actual_direction = np.diff(y_test_original) > 0
        predicted_direction = np.diff(predictions) > 0
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100

        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy
        }

        print("LSTM Model Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        return metrics

    def generate_trading_signals(self, df, target_col, lookback_days=5, threshold=0.02):
        """
        Generate trading signals based on LSTM predictions.

        Args:
            df: DataFrame with features
            target_col: Target column for prediction
            lookback_days: Days to look ahead for signal generation
            threshold: Minimum percentage change to trigger signal

        Returns:
            DataFrame with trading signals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating signals.")

        feature_cols = self.feature_columns
        df_clean = df[feature_cols + [target_col]].dropna()

        if len(df_clean) < self.sequence_length:
            raise ValueError(f"Not enough data for prediction. Need at least {self.sequence_length} points.")

        X_scaled = self.scaler_features.transform(df_clean[feature_cols])

        X_sequences = []
        valid_indices = []

        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i - self.sequence_length:i])
            valid_indices.append(df_clean.index[i])

        X_sequences = np.array(X_sequences)
        predictions = self.predict(X_sequences)

        current_prices = df_clean[target_col].iloc[self.sequence_length:].values
        expected_returns = (predictions - current_prices) / current_prices

        signals = pd.Series(0, index=valid_indices)
        signals[expected_returns > threshold] = 1
        signals[expected_returns < -threshold] = -1

        result_df = pd.DataFrame({
            'Predicted_Price': predictions,
            'Current_Price': current_prices,
            'Expected_Return': expected_returns,
            'Signal': signals
        }, index=valid_indices)

        return result_df

    def plot_predictions(self, X_test, y_test_original, title="LSTM Predictions vs Actual"):
        """Plot predictions against actual values."""
        predictions = self.predict(X_test)

        plt.figure(figsize=(12, 6))
        plt.plot(y_test_original, label='Actual', alpha=0.7)
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()


class xLSTMPredictor(LSTMPredictor):
    """
    Extended LSTM with additional improvements (xLSTM-inspired features).
    """

    def __init__(self, sequence_length=60, lstm_units=128, dropout_rate=0.3, learning_rate=0.001,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(sequence_length, lstm_units, dropout_rate, learning_rate, device)
        self.model_type = "xLSTM"

    def build_model(self, input_shape):
        """
        Build enhanced xLSTM model with memory mixing and gating improvements.
        """

        class xLSTMModel(nn.Module):
            def __init__(self, input_size, lstm_units, dropout_rate):
                super(xLSTMModel, self).__init__()
                self.short_lstm = nn.LSTM(input_size, lstm_units, batch_first=True)
                self.short_bn = nn.BatchNorm1d(lstm_units)
                self.long_lstm = nn.LSTM(input_size, lstm_units, batch_first=True)
                self.long_bn = nn.BatchNorm1d(lstm_units)

                self.memory_gate = nn.Linear(lstm_units, lstm_units)
                self.attention_query = nn.Linear(lstm_units, lstm_units // 2)
                self.attention_key = nn.Linear(lstm_units, lstm_units // 2)
                self.attention_value = nn.Linear(lstm_units, lstm_units // 2)

                self.lstm_final = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True)

                self.fc1 = nn.Linear(lstm_units + lstm_units // 2, 128)
                self.dropout1 = nn.Dropout(dropout_rate)
                self.bn1 = nn.BatchNorm1d(128)
                self.fc_residual = nn.Linear(lstm_units + lstm_units // 2, 128)

                self.fc2 = nn.Linear(128, 64)
                self.dropout2 = nn.Dropout(dropout_rate / 2)
                self.bn2 = nn.BatchNorm1d(64)

                self.fc_output1 = nn.Linear(64, 32)
                self.fc_output2 = nn.Linear(64, 32)
                self.fc_final = nn.Linear(64, 1)

            def forward(self, x):
                short_out, _ = self.short_lstm(x)
                short_out = self.short_bn(short_out.permute(0, 2, 1)).permute(0, 2, 1)

                long_out, _ = self.long_lstm(x)
                long_out = self.long_bn(long_out.permute(0, 2, 1)).permute(0, 2, 1)

                memory_gate = torch.sigmoid(self.memory_gate(short_out))
                mixed_memory = memory_gate * short_out + (1 - memory_gate) * long_out

                query = self.attention_query(mixed_memory)
                key = self.attention_key(mixed_memory)
                value = self.attention_value(mixed_memory)

                attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.lstm_units // 2)
                attention_weights = torch.softmax(attention_scores, dim=-1)
                attention_output = torch.matmul(attention_weights, value)

                global_context = torch.mean(attention_output, dim=1)

                lstm_out, _ = self.lstm_final(mixed_memory)
                lstm_out = lstm_out[:, -1, :]

                combined = torch.cat([global_context, lstm_out], dim=1)

                dense1 = torch.relu(self.fc1(combined))
                dense1 = self.dropout1(dense1)
                dense1 = self.bn1(dense1)

                residual = self.fc_residual(combined)
                dense1 = dense1 + residual

                dense2 = torch.relu(self.fc2(dense1))
                dense2 = self.dropout2(dense2)
                dense2 = self.bn2(dense2)

                output1 = torch.relu(self.fc_output1(dense2))
                output2 = torch.tanh(self.fc_output2(dense2))
                output_combined = torch.cat([output1, output2], dim=1)

                final_output = self.fc_final(output_combined)
                return final_output

        self.model = xLSTMModel(input_shape[-1], self.lstm_units, self.dropout_rate).to(self.device)
        return self.model


class NeuralNetworkStrategy:
    """
    Integration class to connect LSTM predictions with the existing strategy framework.
    """

    def __init__(self, model_type='lstm', sequence_length=60, device='cuda' if torch.cuda.is_available() else 'cpu',
                 **kwargs):
        """
        Initialize neural network strategy.

        Args:
            model_type: 'lstm' or 'xlstm'
            sequence_length: Sequence length for LSTM
            device: Device to run models on
            **kwargs: Additional arguments for model initialization
        """
        self.model_type = model_type
        self.models = {}
        self.sequence_length = sequence_length
        self.model_kwargs = kwargs
        self.device = device

    def train_models(self, df, tickers=None):
        """
        Train LSTM models for each ticker.

        Args:
            df: DataFrame with engineered features
            tickers: List of tickers to train models for
        """
        if tickers is None:
            tickers = self._get_available_tickers(df)

        print(f"Training {self.model_type.upper()} models for tickers: {tickers}")

        for ticker in tickers:
            try:
                print(f"\nTraining model for {ticker}...")

                if self.model_type.lower() == 'xlstm':
                    model = xLSTMPredictor(sequence_length=self.sequence_length, device=self.device,
                                           **self.model_kwargs)
                else:
                    model = LSTMPredictor(sequence_length=self.sequence_length, device=self.device, **self.model_kwargs)

                target_col = f'{ticker}_Adj Close' if f'{ticker}_Adj Close' in df.columns else 'Adj Close'
                feature_cols = [col for col in df.columns if ticker in col and col != target_col]

                if not feature_cols:
                    print(f"No features found for {ticker}, skipping...")
                    continue

                train_loader, test_loader, y_train, y_test = model.prepare_data(df, target_col, feature_cols)
                history = model.train(train_loader, test_loader, epochs=50, verbose=0)

                y_test_original = model.scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
                metrics = model.evaluate(test_loader.dataset.X, y_test_original)

                self.models[ticker] = model
                print(f"Model trained successfully for {ticker}")

            except Exception as e:
                print(f"Failed to train model for {ticker}: {e}")
                continue

    def generate_neural_signals(self, df):
        """
        Generate trading signals using trained LSTM models.

        Args:
            df: DataFrame with current data

        Returns:
            DataFrame with neural network signals
        """
        signals = pd.DataFrame(index=df.index)

        for ticker, model in self.models.items():
            try:
                target_col = f'{ticker}_Adj Close' if f'{ticker}_Adj Close' in df.columns else 'Adj Close'
                signal_df = model.generate_trading_signals(df, target_col)

                aligned_signals = pd.Series(0, index=df.index)
                aligned_signals.loc[signal_df.index] = signal_df['Signal']
                signals[ticker] = aligned_signals

            except Exception as e:
                print(f"Failed to generate signals for {ticker}: {e}")
                signals[ticker] = 0

        return signals.fillna(0)

    def _get_available_tickers(self, df):
        """Extract available tickers from dataframe columns."""
        tickers = set()
        for col in df.columns:
            if '_Adj Close' in col:
                ticker = col.replace('_Adj Close', '')
                tickers.add(ticker)
            elif col == 'Adj Close':
                tickers.add('Stock')
        return list(tickers)
