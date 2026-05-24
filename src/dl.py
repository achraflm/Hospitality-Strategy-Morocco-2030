import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, SimpleRNN, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import optuna
import math

# Vérification sécurisée de gcForest
try:
    from deepforest import CascadeForestRegressor
    GCFOREST_AVAILABLE = True
except ImportError:
    GCFOREST_AVAILABLE = False
    print("⚠️ deep-forest non installé. Optuna ignorera gcForest.")

# ==========================================
# 0. PRÉPARATION DES DONNÉES
# ==========================================
def prepare_dl_sequences(df, window_size=12, split_ratio=0.8, target_col='Arrivals'):
    data = df.copy()
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        data = data.sort_index()

    y = data[[target_col]]
    X = data.drop(columns=[target_col])
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    data_scaled = np.hstack((X_scaled, y_scaled))
    target_idx = data_scaled.shape[1] - 1
    
    Xs, ys = [], []
    for i in range(len(data_scaled) - window_size):
        Xs.append(data_scaled[i : (i + window_size)])
        ys.append(data_scaled[i + window_size, target_idx])
        
    Xs = np.array(Xs)
    ys = np.array(ys)
    
    split_idx = int(len(Xs) * split_ratio)
    
    X_train, X_test = Xs[:split_idx], Xs[split_idx:]
    y_train, y_test = ys[:split_idx], ys[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler_y

# ==========================================
# 1. ARCHITECTURES DE DEEP LEARNING (KERAS)
# ==========================================
def build_keras_model(model_type, input_shape, units, dropout_rate, learning_rate):
    model = Sequential()
    
    # NOUVELLE SYNTAXE : Couche d'entrée explicite (Enlève le warning)
    model.add(Input(shape=input_shape))

    if model_type == "ANN":
        model.add(Flatten())
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units // 2, activation='relu'))
        
    elif model_type == "RNN":
        model.add(SimpleRNN(units, activation='relu'))
        model.add(Dropout(dropout_rate))
        
    elif model_type == "LSTM":
        model.add(LSTM(units, activation='relu'))
        model.add(Dropout(dropout_rate))
        
    elif model_type == "GRU":
        model.add(GRU(units, activation='relu'))
        model.add(Dropout(dropout_rate))
        
    elif model_type == "CNN_LSTM":
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(units, activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    
    # CORRECTION DES NaN : clipnorm=1.0 empêche les gradients d'exploser
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

# ==========================================
# 2. OPTIMISATION AVEC OPTUNA
# ==========================================
def optimize_dl_hyperparameters(X_train, y_train, X_test, y_test, n_trials=15, epochs=20):
    input_shape = (X_train.shape[1], X_train.shape[2])

    def objective(trial):
        # Définition dynamique des choix (si gcForest n'est pas là, on ne le propose pas)
        choices = ["ANN", "RNN", "LSTM", "GRU", "CNN_LSTM"]
        if GCFOREST_AVAILABLE:
            choices.append("gcForest")
            
        model_type = trial.suggest_categorical("model_type", choices)
        
        if model_type == "gcForest":
            X_train_2d = X_train.reshape(X_train.shape[0], -1)
            X_test_2d = X_test.reshape(X_test.shape[0], -1)
            
            n_estimators = trial.suggest_int("n_estimators", 2, 4)
            model = CascadeForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
            
            model.fit(X_train_2d, y_train.ravel())
            preds = model.predict(X_test_2d)
            
            mse = np.mean((y_test.ravel() - preds.ravel())**2)
            return mse

        else:
            units = trial.suggest_int("units", 32, 128, step=32)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32])
            
            model = build_keras_model(model_type, input_shape, units, dropout_rate, learning_rate)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            val_loss = history.history['val_loss'][-1]
            
            # GESTION DES NaN : Si l'erreur explose quand même, on dit à Optuna d'abandonner cet essai proprement
            if math.isnan(val_loss):
                raise optuna.TrialPruned("Le loss a retourné NaN (Exploding gradient). Essai ignoré.")
                
            return val_loss

    study = optuna.create_study(direction="minimize", study_name="Deep_Learning_Optimization")
    study.optimize(objective, n_trials=n_trials)
    
    return study
    from tensorflow.keras import backend as K

def r2_keras(y_true, y_pred):
    """Fonction personnalisée pour calculer le R² pendant l'entraînement Keras"""
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
# ... (début de la fonction build_keras_model avec les if/elif) ...

    model.add(Dense(1))
    
    # --- C'est ICI que va votre bloc compile ---
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer, 
        loss='mse', 
        metrics=[
            'mae', 
            tf.keras.metrics.RootMeanSquaredError(name='rmse'), 
            r2_keras
        ]
    )
    
    return model