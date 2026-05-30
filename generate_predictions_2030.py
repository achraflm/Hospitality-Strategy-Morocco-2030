import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from src.features import build_features, get_nights_feature_list, get_feature_list
from src.models.xgboost import XgboostModel

df = pd.read_csv('data/merged_tourism_data_final.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_clean = df.copy()

df_featured = build_features(df_clean)
target_year = 2030
future_dates = pd.date_range(start='2026-05-01', end=f'{target_year}-12-01', freq='MS')

# --- Arrivals ---
df_ml_arr = df_featured.dropna(subset=['Arrivals']).copy()
selected_features_arr = get_feature_list()
valid_features_arr = [f for f in selected_features_arr if f in df_ml_arr.columns]
X_arr = df_ml_arr[valid_features_arr].fillna(df_ml_arr[valid_features_arr].median())
y_arr = df_ml_arr['Arrivals']
model_arr = XgboostModel()
model_arr.fit(X_arr, y_arr)

# Recursive projection for arrivals
history_arr = df_ml_arr.copy()
proj_arr = []
for date in future_dates:
    new_row = pd.DataFrame({'Date': [date], 'Arrivals': [np.nan]})
    for var in ['Oil_price', 'FDI', 'Poverty_rate', 'REER']:
        new_row[var] = history_arr[var].iloc[-1]
    history_arr = pd.concat([history_arr, new_row], ignore_index=True)
    history_arr = build_features(history_arr)
    feat_vec = history_arr[valid_features_arr].iloc[[-1]].fillna(0)
    pred = max(0, model_arr.predict(feat_vec)[0])
    history_arr.loc[history_arr.index[-1], 'Arrivals'] = pred
    proj_arr.append(pred)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_ml_arr['Date'], df_ml_arr['Arrivals'], label='Historique', color='black')
ax.plot(future_dates, proj_arr, label='Prediction XGBoost 2030', color='blue', linestyle='--')
ax.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde')
plt.title("Prediction des Arrivees jusqu'a 2030")
plt.legend()
os.makedirs('figures', exist_ok=True)
os.makedirs('presentation/figures', exist_ok=True)
os.makedirs('backend/figures', exist_ok=True)
plt.savefig('figures/prediction_2030_arrivees.png', dpi=150)
plt.savefig('presentation/figures/prediction_2030_arrivees.png', dpi=150)
plt.savefig('backend/figures/prediction_2030_arrivees.png', dpi=150)
plt.close()

# --- Nights ---
df_ml_nights = df_featured.dropna(subset=['Nights']).copy()
selected_features_nights = get_nights_feature_list()
valid_features_nights = [f for f in selected_features_nights if f in df_ml_nights.columns]
X_nights = df_ml_nights[valid_features_nights].fillna(df_ml_nights[valid_features_nights].median())
y_nights = df_ml_nights['Nights']
model_nights = XgboostModel()
model_nights.fit(X_nights, y_nights)

# Recursive projection for nights
history_nights = df_ml_nights.copy()
proj_nights = []
for date in future_dates:
    new_row = pd.DataFrame({'Date': [date], 'Nights': [np.nan], 'Arrivals': [history_arr[history_arr['Date']==date]['Arrivals'].values[0]]})
    for var in ['Oil_price', 'FDI', 'Poverty_rate', 'REER']:
        new_row[var] = history_nights[var].iloc[-1]
    history_nights = pd.concat([history_nights, new_row], ignore_index=True)
    history_nights = build_features(history_nights)
    feat_vec = history_nights[valid_features_nights].iloc[[-1]].fillna(0)
    pred = max(0, model_nights.predict(feat_vec)[0])
    history_nights.loc[history_nights.index[-1], 'Nights'] = pred
    proj_nights.append(pred)

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_ml_nights['Date'], df_ml_nights['Nights'], label='Historique', color='black')
ax.plot(future_dates, proj_nights, label='Prediction XGBoost 2030', color='red', linestyle='--')
ax.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde')
plt.title("Prediction des Nuitees jusqu'a 2030")
plt.legend()
plt.savefig('figures/prediction_2030_nuites.png', dpi=150)
plt.savefig('presentation/figures/prediction_2030_nuites.png', dpi=150)
plt.savefig('backend/figures/prediction_2030_nuites.png', dpi=150)
plt.close()

print("Images generated successfully")
