import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import xgboost as xgb
from src.features import build_features, get_nights_feature_list
from main import forecast_recursive_ml
from src.models.xgboost import XgboostModel

# Prepare data
df = pd.read_csv('data/merged_tourism_data_final.csv')
df['Date'] = pd.to_datetime(df['Date'])
df_clean = df.copy()

df_featured = build_features(df_clean)
target_year = 2030
future_dates = pd.date_range(start='2026-05-01', end=f'{target_year}-12-01', freq='MS')

# --- Arrivals ---
df_ml_arr = df_featured.dropna(subset=['Arrivals']).copy()
selected_features_arr = ['Arrivals_lag_1', 'Arrivals_lag_2', 'Arrivals_lag_3', 'Arrivals_lag_12',
                     'Arrivals_rolling_mean_3', 'Arrivals_rolling_std_3',
                     'Month', 'Quarter', 'Year',
                     'is_summer', 'is_ramadan', 'covid_shock', 'cdm_event',
                     'world_gdp_growth', 'jet_fuel_price']
X_arr = df_ml_arr[selected_features_arr].fillna(df_ml_arr[selected_features_arr].median())
y_arr = df_ml_arr['Arrivals']
model_arr = XgboostModel()
model_arr.fit(X_arr, y_arr)

proj_arr = forecast_recursive_ml(model_arr, df_ml_arr, future_dates, selected_features_arr, target_col='Arrivals')
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_ml_arr['Date'], df_ml_arr['Arrivals'], label='Historique', color='black')
ax.plot(future_dates, np.clip(proj_arr, 0, None), label='Prediction XGBoost 2030', color='blue', linestyle='--')
ax.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde')
plt.title("Prediction des Arrivees jusqu'a 2030")
plt.legend()
plt.savefig('figures/prediction_2030_arrivees.png', dpi=150)
os.makedirs('presentation/figures', exist_ok=True)
plt.savefig('presentation/figures/prediction_2030_arrivees.png', dpi=150)
os.makedirs('backend/figures', exist_ok=True)
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

proj_nights = forecast_recursive_ml(model_nights, df_ml_nights, future_dates, valid_features_nights, target_col='Nights')
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df_ml_nights['Date'], df_ml_nights['Nights'], label='Historique', color='black')
ax.plot(future_dates, np.clip(proj_nights, 0, None), label='Prediction XGBoost 2030', color='red', linestyle='--')
ax.axvspan('2030-06-01', '2030-07-31', color='teal', alpha=0.15, label='Coupe du Monde')
plt.title("Prediction des Nuitees jusqu'a 2030")
plt.legend()
plt.savefig('figures/prediction_2030_nuites.png', dpi=150)
plt.savefig('presentation/figures/prediction_2030_nuites.png', dpi=150)
plt.savefig('backend/figures/prediction_2030_nuites.png', dpi=150)
plt.close()

print("Images generated successfully")
