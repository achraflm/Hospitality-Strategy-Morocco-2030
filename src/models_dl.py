import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# Helper function to dynamically import optuna and tensorflow
def check_dependencies():
    try:
        import optuna
        import tensorflow as tf
    except ImportError:
        print("Warning: optuna or tensorflow is not installed. Installing...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "tensorflow", "scikeras", "-q"])
        except Exception as e:
            print(f"Error: Failed to install deep learning dependencies: {e}")

check_dependencies()

# Import after dependencies check
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

def prepare_dl_sequences(df_in, window_size=12, split_ratio=0.8):
    """
    Interpolates, scales, and prepares sliding window sequences for deep learning models.
    Fits scaling parameters on train split only to prevent data leakage.
    """
    df_p = df_in.copy()
    
    # Feature columns as used in original DL notebook
    features = [
        c for c in ['Arrivals', 'Nights', 'is_covid', 'InterTourismeReceipts', 'REER', 'Oil_price', 'FDI', 'Poverty_rate'] 
        if c in df_p.columns and not df_p[c].isnull().all()
    ]
    
    # Clean data (interpolate and fill)
    df_p[features] = df_p[features].interpolate().bfill().ffill()
    data = df_p[features].values
    
    # Train / Test split on raw data
    train_size = int(len(data) * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Scale features (fitting scaler only on training data)
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)
    
    # Helper to construct sequences
    def create_sequences(scaled_arr):
        X, y = [], []
        for i in range(len(scaled_arr) - window_size):
            X.append(scaled_arr[i : i + window_size, :])
            # Target is the first feature (Arrivals)
            y.append(scaled_arr[i + window_size, 0])
        return np.array(X), np.array(y)
        
    X_train, y_train = create_sequences(scaled_train)
    X_test, y_test = create_sequences(scaled_test)
    
    return X_train, X_test, y_train, y_test, scaler

def build_dl_model(model_type, units, dropout, lr, input_shape):
    """
    Builds a deep learning model of type RNN, GRU, LSTM, Stacked_LSTM, or CNN-LSTM.
    """
    model = Sequential()
    
    if model_type == 'RNN':
        model.add(SimpleRNN(units, input_shape=input_shape))
    elif model_type == 'GRU':
        model.add(GRU(units, input_shape=input_shape))
    elif model_type == 'LSTM':
        model.add(LSTM(units, input_shape=input_shape))
    elif model_type == 'Stacked_LSTM':
        model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(LSTM(units // 2))
    elif model_type == 'CNN-LSTM':
        model.add(Conv1D(filters=32, kernel_size=3, activation='tanh', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(units))
        
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model

def optimize_dl_hyperparameters(X_train, y_train, X_test, y_test, n_trials=15, epochs=20):
    """
    Uses Optuna to optimize hyperparameters across different DL architectures.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['RNN', 'GRU', 'LSTM', 'CNN-LSTM', 'Stacked_LSTM'])
        units = trial.suggest_int('units', 32, 128)
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        
        model = build_dl_model(model_type, units, dropout, lr, input_shape)
        # Train for few epochs during search
        model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
        
        preds = model.predict(X_test, verbose=0)
        return mean_squared_error(y_test, preds)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study

# Transformer model encoder layer and builder
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Encoder block for the Transformer model."""
    # Multi-head attention
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=64, num_transformer_blocks=2, mlp_units=[64], dropout=0.1, mlp_dropout=0.1):
    """Builds a complete MultiHeadAttention Transformer model."""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)
    return tf.keras.Model(inputs, outputs)
