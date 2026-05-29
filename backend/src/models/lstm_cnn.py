import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

class LstmCnnModel:
    def __init__(self, window_size=12, epochs=50, batch_size=16, lstm_units=64, filters=64, kernel_size=3, dropout=0.1, learning_rate=0.001):
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_units = lstm_units
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.learning_rate = learning_rate
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
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, activation='relu', input_shape=(self.window_size, X_train.shape[1]), padding='same'),
            MaxPooling1D(pool_size=2),
            LSTM(self.lstm_units, activation='tanh', return_sequences=False),
            Dropout(self.dropout),
            Dense(1)
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='mse')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
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
