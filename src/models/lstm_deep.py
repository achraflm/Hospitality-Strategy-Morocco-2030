import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LstmDeepModel:
    def __init__(self, window_size=12, epochs=5, batch_size=16):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def fit(self, X_train, y_train):
        X_train_vals = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_vals = y_train.values.reshape(-1, 1) if hasattr(y_train, 'values') else y_train.reshape(-1, 1)
        
        X_scaled = self.scaler_x.fit_transform(X_train_vals)
        y_scaled = self.scaler_y.fit_transform(y_train_vals).flatten()
        
        X_seq, y_seq = [], []
        for i in range(self.window_size, len(X_scaled)):
            X_seq.append(X_scaled[i - self.window_size : i])
            y_seq.append(y_scaled[i])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        if len(X_seq) == 0:
            raise ValueError(f"Training dataset size {len(X_scaled)} is smaller than sequence window_size {self.window_size}.")
            
        self.model = Sequential([
            LSTM(32, input_shape=(self.window_size, X_train.shape[1]), activation='relu', return_sequences=True),
            LSTM(16, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        return self
        
    def predict(self, X_test, X_train_history):
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions.")
            
        X_test_vals = X_test.values if hasattr(X_test, 'values') else X_test
        X_train_vals = X_train_history.values if hasattr(X_train_history, 'values') else X_train_history
        
        X_full = np.concatenate([X_train_vals, X_test_vals], axis=0)
        X_scaled = self.scaler_x.transform(X_full)
        
        X_seq = []
        for i in range(len(X_train_vals), len(X_scaled)):
            X_seq.append(X_scaled[i - self.window_size : i])
            
        X_seq = np.array(X_seq)
        
        preds_scaled = self.model.predict(X_seq, verbose=0)
        preds_inv = self.scaler_y.inverse_transform(preds_scaled).flatten()
        return preds_inv
