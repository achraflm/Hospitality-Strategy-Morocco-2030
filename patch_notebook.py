import nbformat

nb = nbformat.read('notebooks/05_deep_learning.ipynb', as_version=4)

new_source = '''import pandas as pd
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

    cols = ['GDP_Construction_lag1']
    df_x_train = df_x_train.drop(columns=cols, errors='ignore')
    df_x_test = df_x_test.drop(columns=cols, errors='ignore')

    imputer = SimpleImputer(strategy='mean')
    df_x_train_imputed = imputer.fit_transform(df_x_train)
    df_x_test_imputed = imputer.transform(df_x_test)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(df_x_train_imputed)
    x_test_scaled = scaler.transform(df_x_test_imputed)

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
        y_pred_diff_wf = []
        test_start_idx = len(X_train_diff)
        n_samples = len(X_all_diff)
        
        shared_model = None
        if is_dl:
            shared_model = model_fn()
            
        for i in range(test_start_idx, n_samples):
            if is_dl:
                X_tr, y_tr = X_all_diff_dl[:i], y_all_diff[:i]
                X_val, y_val = X_all_diff_dl[i:i+1], y_all_diff[i:i+1]
                shared_model.fit(X_tr, y_tr, epochs=5, batch_size=16, verbose=0)
                pred = shared_model.predict(X_val, verbose=0)
                y_pred_diff_wf.append(pred[0][0])
            else:
                X_tr, y_tr = X_all_diff[:i], y_all_diff[:i]
                X_val, y_val = X_all_diff[i:i+1], y_all_diff[i:i+1]
                model = model_fn()
                model.fit(X_tr, y_tr)
                pred = model.predict(X_val)
                y_pred_diff_wf.append(pred[0])
                
        valeurs_ref = df_y_test.iloc[:-1].values.flatten()
        y_pred_finales = valeurs_ref + np.array(y_pred_diff_wf)
        y_true_aligned = df_y_test.iloc[1:].values.flatten()
        
        r2 = r2_score(y_true_aligned, y_pred_finales)
        rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pred_finales))
        mae = mean_absolute_error(y_true_aligned, y_pred_finales)
        mape = mean_absolute_percentage_error(y_true_aligned, y_pred_finales) * 100
        
        res_dict = auto_eval.evaluate_model(target_name=target, model_name=model_name, y_true=y_true_aligned, y_pred=y_pred_finales, is_walk_forward=True)
        print(f"AutoResearch Insights: {res_dict['Insights']}")
        
        all_results.append({'Target': target, 'Model': model_name, 'Validation': 'Walk-Forward', 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape})
        
        plt.figure(figsize=(10,5))
        plt.plot(y_true_aligned, label='Reel', color='black')
        plt.plot(y_pred_finales, label=f'{model_name} Walk-Forward', color='red', linestyle='--')
        plt.title(f'{model_name} - {target} - WF Prediction vs Real')
        plt.legend()
        plt.grid()
        plt.savefig(f"figures/{model_name.replace(' ', '_').replace('+', 'plus')}_{target}_WF_Prediction.png")
        plt.close()

auto_eval.generate_report()
print('AutoResearch report generated successfully.')
results_df = pd.DataFrame(all_results)
results_df.to_csv('results/autoresearch_output/autoresearch_comparison.csv', index=False)
display(results_df)
'''

for cell in nb.cells:
    if 'TimeSeriesSplit' in cell.source or 'forecast_model' in cell.source:
        cell.source = new_source

nbformat.write(nb, 'notebooks/05_deep_learning.ipynb')
print('Notebook updated successfully.')
