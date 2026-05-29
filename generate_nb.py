import json

notebook = {
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]]
    })

add_markdown("# 1. Imports and Data Loading")
add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten

# Create directories for results
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Paths
data_dir = '../data'
df_x_train = pd.read_csv(os.path.join(data_dir, 'separted', 'X_train.csv'))
df_y_train = pd.read_csv(os.path.join(data_dir, 'separted', 'y_train.csv'))
df_x_test = pd.read_csv(os.path.join(data_dir, 'separted', 'X_test.csv'))
df_y_test = pd.read_csv(os.path.join(data_dir, 'separted', 'y_test.csv'))

# Clean
columns_to_drop = ['GDP_Construction_lag1']
df_x_train = df_x_train.drop(columns=columns_to_drop, errors='ignore')
df_x_test = df_x_test.drop(columns=columns_to_drop, errors='ignore')

# Imputation & Scaling
imputer = SimpleImputer(strategy='mean')
df_x_train_imputed = imputer.fit_transform(df_x_train)
df_x_test_imputed = imputer.transform(df_x_test)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(df_x_train_imputed)
x_test_scaled = scaler.transform(df_x_test_imputed)

y_train = df_y_train.values.flatten()
y_test = df_y_test.values.flatten()

# Results dictionary to compile metrics
all_results = []""")

add_markdown("# 2. Machine Learning Models (Standard Train/Test)\nModèles demandés : Regression linéaire (Multiple), Ridge, Decision Tree, SVM (RBF)")
add_code("""ml_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'SVM (RBF)': SVR(kernel='rbf', C=100, gamma='auto')
}

for name, model in ml_models.items():
    print(f"Training {name}...")
    model.fit(x_train_scaled, y_train)
    preds = model.predict(x_test_scaled)
    
    # Metrics
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    
    all_results.append({
        'Model': name,
        'Type': 'Machine Learning',
        'Validation': 'Standard',
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    })
    
    # Plot & Save Figure
    plt.figure(figsize=(10,5))
    plt.plot(y_test, label='Real', color='black')
    plt.plot(preds, label=f'{name} Prediction', color='blue', linestyle='--')
    plt.title(f'{name} - Real vs Prediction')
    plt.legend()
    plt.grid()
    fig_name = f"figures/{name.replace(' ', '_').replace('(', '').replace(')', '')}_Prediction.png"
    plt.savefig(fig_name)
    plt.close()
    print(f"{name} done. Metrics: R2={r2:.3f}, RMSE={rmse:.3f}, MAPE={mape:.3f}")""")

add_markdown("# 3. Walk-Forward Training Preparation\nPréparation des données pour l'évaluation Walk-Forward (XGBoost et Deep Learning).")
add_code("""# Différenciation pour stationnarité
y_train_diff = df_y_train.diff().dropna()
y_test_diff = df_y_test.diff().dropna()

X_train_diff = x_train_scaled[1:]
X_test_diff = x_test_scaled[1:]

X_all_diff = np.vstack((X_train_diff, X_test_diff))
y_all_diff = pd.concat([y_train_diff, y_test_diff]).values.flatten()

# Reshape pour les modèles Deep Learning (samples, timesteps, features)
X_all_diff_dl = X_all_diff.reshape((X_all_diff.shape[0], 1, X_all_diff.shape[1]))""")

add_markdown("# 4. Walk-Forward Training (XGBoost & Deep Learning)\nModèles : XGBoost, LSTM, GRU, LSTM 2-layer, LSTM + CNN.")
add_code("""def walk_forward_evaluate(model_name, model_fn, is_dl=False, epochs=50):
    print(f"\\n--- Début de l'entraînement Walk-Forward pour {model_name} ---")
    tscv = TimeSeriesSplit(n_splits=len(X_test_diff), test_size=1)
    y_pred_diff_wf = []
    test_start_idx = len(X_train_diff)
    
    for train_index, test_index in tscv.split(X_all_diff):
        if test_index[0] < test_start_idx:
            continue
            
        if is_dl:
            X_train_wf, X_test_wf = X_all_diff_dl[train_index], X_all_diff_dl[test_index]
            model = model_fn()
            model.fit(X_train_wf, y_all_diff[train_index], epochs=epochs, batch_size=16, verbose=0)
            pred = model.predict(X_test_wf, verbose=0)
            y_pred_diff_wf.append(pred[0][0])
        else:
            X_train_wf, X_test_wf = X_all_diff[train_index], X_all_diff[test_index]
            model = model_fn()
            model.fit(X_train_wf, y_all_diff[train_index])
            pred = model.predict(X_test_wf)
            y_pred_diff_wf.append(pred[0])
            
    # Reconstruire les prédictions (inverser diff)
    valeurs_ref = df_y_test.iloc[:-1].values.flatten()
    y_pred_finales = valeurs_ref + np.array(y_pred_diff_wf)
    y_true_aligned = df_y_test.iloc[1:].values.flatten()
    
    r2 = r2_score(y_true_aligned, y_pred_finales)
    rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_finales))
    mae = mean_absolute_error(y_true_aligned, y_pred_finales)
    mape = mean_absolute_percentage_error(y_true_aligned, y_pred_finales)
    
    all_results.append({
        'Model': model_name,
        'Type': 'Deep Learning' if is_dl else 'Machine Learning',
        'Validation': 'Walk-Forward',
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    })
    
    plt.figure(figsize=(10,5))
    plt.plot(y_true_aligned, label='Réel', color='black')
    plt.plot(y_pred_finales, label=f'{model_name} Walk-Forward', color='red', linestyle='--')
    plt.title(f'{model_name} - Walk-Forward Prediction vs Real')
    plt.legend()
    plt.grid()
    fig_name = f"figures/{model_name.replace(' ', '_').replace('+', 'plus')}_WF_Prediction.png"
    plt.savefig(fig_name)
    plt.close()
    print(f"{model_name} terminé. R2={r2:.3f}, RMSE={rmse:.3f}, MAPE={mape:.3f}")

# 1. XGBoost
def get_xgb():
    return xgb.XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=50, reg_alpha=0, reg_lambda=10, subsample=0.7, random_state=42)
walk_forward_evaluate("XGBoost", get_xgb, is_dl=False)

# 2. LSTM
def get_lstm():
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(1, X_all_diff.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
walk_forward_evaluate("LSTM", get_lstm, is_dl=True, epochs=20)

# 3. GRU
def get_gru():
    model = Sequential([
        GRU(32, activation='relu', input_shape=(1, X_all_diff.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
walk_forward_evaluate("GRU", get_gru, is_dl=True, epochs=20)

# 4. LSTM 2-Layer
def get_lstm_2layer():
    model = Sequential([
        LSTM(32, activation='relu', return_sequences=True, input_shape=(1, X_all_diff.shape[1])),
        LSTM(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
walk_forward_evaluate("LSTM 2 Layers", get_lstm_2layer, is_dl=True, epochs=20)

# 5. LSTM + CNN
def get_lstm_cnn():
    model = Sequential([
        Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(1, X_all_diff.shape[1])),
        LSTM(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
walk_forward_evaluate("LSTM + CNN", get_lstm_cnn, is_dl=True, epochs=20)""")

add_markdown("# 5. Compilation des résultats\nEnregistrement des performances de tous les modèles dans un fichier CSV.")
add_code("""results_df = pd.DataFrame(all_results)
results_df.to_csv('results/model_comparison_results.csv', index=False)
print("\\nTous les modèles ont été évalués. Résultats enregistrés dans 'results/model_comparison_results.csv'.")
display(results_df)""")

with open('notebooks/04_machine_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
