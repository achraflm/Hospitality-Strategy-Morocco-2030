import json
import os

def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

def add_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]]}

def add_cd(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]]}

# ================= 03_feature_engineering =================
cells_03 = []
cells_03.append(add_md("# Feature Engineering pour Arrivals et Nights"))
cells_03.append(add_cd("""import pandas as pd
import numpy as np
import os

file_path = '../data/merged_tourism_data_final.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])

targets = ['Arrivals', 'Nights']
folder_path = '../data/separted'
os.makedirs(folder_path, exist_ok=True)

for target in targets:
    print(f"--- Feature Engineering pour {target} ---")
    df_t = df.copy()
    
    # Lags
    lags = [1 , 2 ,6 ,12]
    for k in lags:
        df_t[f'lags_{k}'] = df_t[target].shift(k)
        
    # Rolling
    windows = [3 , 6, 12]
    for w in windows :
        df_t[f'roll_mean_{w}'] = df_t[target].shift(1).rolling(window=w).mean()
        df_t[f'roll_std_{w}'] = df_t[target].shift(1).rolling(window=w).std()

    # Growth
    df_t['growth_yoy'] = (df_t[target] - df_t[target].shift(12)) / df_t[target].shift(12) * 100
    df_t.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Time
    df_t['month'] = df_t['Date'].dt.month
    df_t['year'] = df_t['Date'].dt.year
    df_t['quarter'] = df_t['Date'].dt.quarter
    df_t['month_sin'] = np.sin(2 * np.pi * df_t['month'] / 12)
    df_t['month_cos'] = np.cos(2 * np.pi * df_t['month'] / 12)
    
    def get_season(m):
        if m in [12, 1, 2]: return 0
        elif m in [3, 4, 5]: return 1
        elif m in [6, 7, 8]: return 2
        else: return 3
    df_t['season'] = df_t['month'].apply(get_season)
    df_t = pd.get_dummies(df_t, columns=['season'], prefix='saison', drop_first=True)
    
    # Eco
    eco_vars = ['Oil_price', 'FDI', 'Poverty_rate', 'GDP_Construction']
    for var in eco_vars:
        if var in df_t.columns:
            df_t[f'{var}_lag1'] = df_t[var].shift(1)
            df_t[f'{var}_lag3'] = df_t[var].shift(3)
    if 'REER' in df_t.columns:
        df_t['REER_lag1'] = df_t['REER'].shift(1)
        
    # Events
    df_t['cdm_event'] = 0
    df_t.loc[(df_t['Date'] >= '2025-12-01') & (df_t['Date'] <= '2026-01-31'), 'cdm_event'] = 1
    df_t.loc[(df_t['Date'] >= '2030-06-01') & (df_t['Date'] <= '2030-07-31'), 'cdm_event'] = 1
    
    FEATURES_ML = [
        'lags_1', 'lags_2', 'lags_12',
        'roll_mean_3', 'roll_mean_6', 'roll_mean_12',
        'roll_std_3', 'roll_std_6', 'roll_std_12', 'growth_yoy',
        'month_sin', 'month_cos', 'quarter', 'year',
        'saison_1', 'saison_2', 'saison_3',
        'Oil_price_lag1', 'FDI_lag1', 'Poverty_rate_lag1', 'REER_lag1',
        'is_covid', 'cdm_event'
    ]
    
    df_ml = df_t.dropna(subset=[target]).copy()
    valid_features = [c for c in FEATURES_ML if c in df_ml.columns]
    
    train_end = '2022-12-31'
    test_start = '2023-01-01'
    
    X_train = df_ml[df_ml['Date'] <= train_end][valid_features]
    y_train = df_ml[df_ml['Date'] <= train_end][target]
    
    X_test = df_ml[df_ml['Date'] >= test_start][valid_features]
    y_test = df_ml[df_ml['Date'] >= test_start][target]
    
    X_train.to_csv(f'{folder_path}/X_train_{target}.csv', index=False)
    y_train.to_csv(f'{folder_path}/y_train_{target}.csv', index=False)
    X_test.to_csv(f'{folder_path}/X_test_{target}.csv', index=False)
    y_test.to_csv(f'{folder_path}/y_test_{target}.csv', index=False)
    
    print(f"Features extraites pour {target} : Train size = {len(X_train)}, Test size = {len(X_test)}")
"""))

with open('notebooks/03_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
    json.dump(create_nb(cells_03), f, indent=1, ensure_ascii=False)

# ================= 02_modele_statistique =================
cells_02 = []
cells_02.append(add_md("# Modèles Statistiques pour Arrivals et Nights"))
cells_02.append(add_cd("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

targets = ['Arrivals', 'Nights']
all_results = []

for target in targets:
    print(f"\\n--- Modèle Statistique (ARIMA) pour {target} ---")
    y_train = pd.read_csv(f'../data/separted/y_train_{target}.csv').values.flatten()
    y_test = pd.read_csv(f'../data/separted/y_test_{target}.csv').values.flatten()
    
    # Modèle ARIMA basique
    model = ARIMA(y_train, order=(5,1,0))
    model_fit = model.fit()
    preds = model_fit.forecast(steps=len(y_test))
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    
    all_results.append({
        'Model': 'ARIMA',
        'Target': target,
        'Type': 'Statistical',
        'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape
    })
    
    plt.figure(figsize=(10,5))
    plt.plot(y_test, label='Real', color='black')
    plt.plot(preds, label='ARIMA Prediction', color='blue', linestyle='--')
    plt.title(f'ARIMA - {target} - Real vs Prediction')
    plt.legend()
    plt.grid()
    plt.savefig(f"figures/ARIMA_{target}_Prediction.png")
    plt.close()
    
pd.DataFrame(all_results).to_csv('results/statistical_results.csv', index=False)
print("Modèles statistiques évalués.")
"""))

with open('notebooks/02_modele_statistique.ipynb', 'w', encoding='utf-8') as f:
    json.dump(create_nb(cells_02), f, indent=1, ensure_ascii=False)

# ================= 04_machine_learning =================
cells_04 = []
cells_04.append(add_md("# Machine Learning Models (Arrivals & Nights)\nModèles : Regression linéaire, Ridge, Decision Tree, SVM (RBF)"))
cells_04.append(add_cd("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import warnings
import sys
sys.path.append('../src')
from autoresearch import AutoResearchEvaluator
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
auto_eval = AutoResearchEvaluator(output_dir='results/autoresearch_output')

targets = ['Arrivals', 'Nights']
all_results = []

ml_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'SVM (RBF)': SVR(kernel='rbf', C=100, gamma='auto')
}

for target in targets:
    print(f"\\n--- Cible : {target} ---")
    df_x_train = pd.read_csv(f'../data/separted/X_train_{target}.csv')
    df_y_train = pd.read_csv(f'../data/separted/y_train_{target}.csv')
    df_x_test = pd.read_csv(f'../data/separted/X_test_{target}.csv')
    df_y_test = pd.read_csv(f'../data/separted/y_test_{target}.csv')

    # Nettoyage et Imputation & Scaling
    cols = ['GDP_Construction_lag1']
    df_x_train = df_x_train.drop(columns=cols, errors='ignore')
    df_x_test = df_x_test.drop(columns=cols, errors='ignore')

    imputer = SimpleImputer(strategy='mean')
    df_x_train_imputed = imputer.fit_transform(df_x_train)
    df_x_test_imputed = imputer.transform(df_x_test)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(df_x_train_imputed)
    x_test_scaled = scaler.transform(df_x_test_imputed)

    y_train = df_y_train.values.flatten()
    y_test = df_y_test.values.flatten()

    for name, model in ml_models.items():
        print(f"Training {name} for {target}...")
        model.fit(x_train_scaled, y_train)
        preds = model.predict(x_test_scaled)
        
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)
        
        # AutoResearch evaluation
        res_dict = auto_eval.evaluate_model(target_name=target, model_name=name, y_true=y_test, y_pred=preds, is_walk_forward=False)
        print(f"AutoResearch Insights: {res_dict['Insights']}")
        
        all_results.append({
            'Target': target,
            'Model': name,
            'Type': 'Machine Learning',
            'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape
        })
        
        plt.figure(figsize=(10,5))
        plt.plot(y_test, label='Real', color='black')
        plt.plot(preds, label=f'{name} Prediction', color='blue', linestyle='--')
        plt.title(f'{name} - {target} - Real vs Prediction')
        plt.legend()
        plt.grid()
        fig_name = f"figures/{name.replace(' ', '_').replace('(', '').replace(')', '')}_{target}_Prediction.png"
        plt.savefig(fig_name)
        plt.close()

auto_eval.generate_report()
print('AutoResearch report generated successfully.')
results_df = pd.DataFrame(all_results)
results_df.to_csv('results/ml_results.csv', index=False)
display(results_df)
"""))

with open('notebooks/04_machine_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(create_nb(cells_04), f, indent=1, ensure_ascii=False)

# ================= 05_deep_learning =================
cells_05 = []
cells_05.append(add_md("# Deep Learning & XGBoost Models (Walk-Forward)\nModèles : XGBoost, LSTM, GRU, LSTM 2-layer, LSTM + CNN\nCibles : Arrivals et Nights"))
cells_05.append(add_cd("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
import os
import warnings
import sys
sys.path.append('../src')
from autoresearch import AutoResearchEvaluator
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
auto_eval = AutoResearchEvaluator(output_dir='results/autoresearch_output')

targets = ['Arrivals', 'Nights']
all_results = []

def get_xgb(): return xgb.XGBRegressor(learning_rate=0.1, max_depth=4, n_estimators=50, reg_alpha=0, reg_lambda=10, subsample=0.7, random_state=42)
def get_lstm(shape):
    model = Sequential([LSTM(32, activation='relu', input_shape=shape), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model
def get_gru(shape):
    model = Sequential([GRU(32, activation='relu', input_shape=shape), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model
def get_lstm_2layer(shape):
    model = Sequential([LSTM(32, activation='relu', return_sequences=True, input_shape=shape), LSTM(16, activation='relu'), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model
def get_lstm_cnn(shape):
    model = Sequential([Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=shape), LSTM(16, activation='relu'), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    return model

for target in targets:
    print(f"\\n=== Cible : {target} ===")
    df_x_train = pd.read_csv(f'../data/separted/X_train_{target}.csv')
    df_y_train = pd.read_csv(f'../data/separted/y_train_{target}.csv')
    df_x_test = pd.read_csv(f'../data/separted/X_test_{target}.csv')
    df_y_test = pd.read_csv(f'../data/separted/y_test_{target}.csv')

    # Nettoyage
    cols = ['GDP_Construction_lag1']
    df_x_train = df_x_train.drop(columns=cols, errors='ignore')
    df_x_test = df_x_test.drop(columns=cols, errors='ignore')

    # Imputation & Scaling
    imputer = SimpleImputer(strategy='mean')
    df_x_train_imputed = imputer.fit_transform(df_x_train)
    df_x_test_imputed = imputer.transform(df_x_test)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(df_x_train_imputed)
    x_test_scaled = scaler.transform(df_x_test_imputed)

    # Différenciation
    y_train_diff = df_y_train.diff().dropna()
    y_test_diff = df_y_test.diff().dropna()
    X_train_diff = x_train_scaled[1:]
    X_test_diff = x_test_scaled[1:]
    X_all_diff = np.vstack((X_train_diff, X_test_diff))
    y_all_diff = pd.concat([y_train_diff, y_test_diff]).values.flatten()

    X_all_diff_dl = X_all_diff.reshape((X_all_diff.shape[0], 1, X_all_diff.shape[1]))
    dl_shape = (1, X_all_diff.shape[1])

    models_to_run = {
        'XGBoost': (get_xgb, False),
        'LSTM': (lambda: get_lstm(dl_shape), True),
        'GRU': (lambda: get_gru(dl_shape), True),
        'LSTM 2 Layers': (lambda: get_lstm_2layer(dl_shape), True),
        'LSTM + CNN': (lambda: get_lstm_cnn(dl_shape), True)
    }

    for model_name, (model_fn, is_dl) in models_to_run.items():
        print(f"Walk-Forward {model_name} pour {target}...")
        tscv = TimeSeriesSplit(n_splits=len(X_test_diff), test_size=1)
        y_pred_diff_wf = []
        test_start_idx = len(X_train_diff)
        
        for train_index, test_index in tscv.split(X_all_diff):
            if test_index[0] < test_start_idx: continue
            if is_dl:
                X_train_wf, X_test_wf = X_all_diff_dl[train_index], X_all_diff_dl[test_index]
                model = model_fn()
                model.fit(X_train_wf, y_all_diff[train_index], epochs=10, batch_size=16, verbose=0)
                pred = model.predict(X_test_wf, verbose=0)
                y_pred_diff_wf.append(pred[0][0])
            else:
                X_train_wf, X_test_wf = X_all_diff[train_index], X_all_diff[test_index]
                model = model_fn()
                model.fit(X_train_wf, y_all_diff[train_index])
                pred = model.predict(X_test_wf)
                y_pred_diff_wf.append(pred[0])
                
        # Reconstruire
        valeurs_ref = df_y_test.iloc[:-1].values.flatten()
        y_pred_finales = valeurs_ref + np.array(y_pred_diff_wf)
        y_true_aligned = df_y_test.iloc[1:].values.flatten()
        
        r2 = r2_score(y_true_aligned, y_pred_finales)
        rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_finales))
        mae = mean_absolute_error(y_true_aligned, y_pred_finales)
        mape = mean_absolute_percentage_error(y_true_aligned, y_pred_finales)
        
        # AutoResearch evaluation
        res_dict = auto_eval.evaluate_model(target_name=target, model_name=model_name, y_true=y_true_aligned, y_pred=y_pred_finales, is_walk_forward=True)
        print(f"AutoResearch Insights: {res_dict['Insights']}")
        
        all_results.append({'Target': target, 'Model': model_name, 'Type': 'Deep Learning' if is_dl else 'XGBoost', 'Validation': 'Walk-Forward', 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape})
        
        plt.figure(figsize=(10,5))
        plt.plot(y_true_aligned, label='Réel', color='black')
        plt.plot(y_pred_finales, label=f'{model_name} Walk-Forward', color='red', linestyle='--')
        plt.title(f'{model_name} - {target} - WF Prediction vs Real')
        plt.legend()
        plt.grid()
        plt.savefig(f"figures/{model_name.replace(' ', '_').replace('+', 'plus')}_{target}_WF_Prediction.png")
        plt.close()

auto_eval.generate_report()
print('AutoResearch report generated successfully.')
results_df = pd.DataFrame(all_results)
results_df.to_csv('results/dl_wf_results.csv', index=False)
display(results_df)
"""))

with open('notebooks/05_deep_learning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(create_nb(cells_05), f, indent=1, ensure_ascii=False)
