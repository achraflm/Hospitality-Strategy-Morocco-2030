import numpy as np
import pandas as pd

def forecast_recursive_ml(model, df_historical, future_dates, valid_features, target_col='Arrivals'):
    """Prédiction récursive pas-à-pas pour les modèles de Machine Learning."""
    
    # Assurer que les colonnes existent
    for col, default_val in zip(['Oil_price', 'FDI', 'Poverty_rate', 'REER'], [75.0, 2.0, 4.0, 100.0]):
        if col not in df_historical.columns:
            df_historical[col] = default_val

    history_df = df_historical[['Date', target_col, 'Oil_price', 'FDI', 'Poverty_rate', 'REER']].copy()
    history_df['Date'] = pd.to_datetime(history_df['Date'])
    
    last_row = df_historical.iloc[-1]
    last_oil = last_row.get('Oil_price', 75.0)
    last_fdi = last_row.get('FDI', 2.0)
    last_poverty = last_row.get('Poverty_rate', 4.0)
    last_reer = last_row.get('REER', 100.0)
    
    predictions = []
    
    for date in future_dates:
        new_row = {
            'Date': date,
            target_col: np.nan,
            'Oil_price': last_oil,
            'FDI': last_fdi,
            'Poverty_rate': last_poverty,
            'REER': last_reer,
            'is_covid': 1 if (date >= pd.to_datetime('2020-03-01') and date <= pd.to_datetime('2021-12-01')) else 0
        }
        
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        idx = history_df.index[-1]
        
        # Recalculer les lags
        for k in [1, 2, 6, 12]:
            history_df.loc[idx, f'lags_{k}'] = history_df[target_col].iloc[-1 - k]
            
        # Moyennes mobiles
        for w in [3, 6, 12]:
            history_df.loc[idx, f'roll_mean_{w}'] = history_df[target_col].iloc[-1-w:-1].mean()
            history_df.loc[idx, f'roll_std_{w}'] = history_df[target_col].iloc[-1-w:-1].std()
            
        # Croissance YoY
        val_prev = history_df[target_col].iloc[-2]
        val_yoy = history_df[target_col].iloc[-14]
        history_df.loc[idx, 'growth_yoy'] = ((val_prev - val_yoy) / (val_yoy if val_yoy != 0 else 1.0)) * 100
        
        # Temporel et saisonnier
        m = date.month
        y = date.year
        q = (m - 1) // 3 + 1
        history_df.loc[idx, 'month_sin'] = np.sin(2 * np.pi * m / 12)
        history_df.loc[idx, 'month_cos'] = np.cos(2 * np.pi * m / 12)
        history_df.loc[idx, 'quarter'] = q
        history_df.loc[idx, 'year'] = y
        
        # Saison One-Hot
        season = 0 if m in [12, 1, 2] else 1 if m in [3, 4, 5] else 2 if m in [6, 7, 8] else 3
        history_df.loc[idx, 'saison_1'] = 1 if season == 1 else 0
        history_df.loc[idx, 'saison_2'] = 1 if season == 2 else 0
        history_df.loc[idx, 'saison_3'] = 1 if season == 3 else 0
        
        history_df.loc[idx, 'is_summer'] = 1 if m in [6, 7, 8] else 0
        history_df.loc[idx, 'is_high_season'] = 1 if m in [4, 5, 7, 8, 10, 12] else 0
        history_df.loc[idx, 'is_vacances'] = 1 if m in [7, 8, 12] else 0
        
        holiday_map = {1: 2, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1, 8: 3, 9: 0, 10: 0, 11: 2, 12: 0}
        history_df.loc[idx, 'jours_feries_count'] = holiday_map.get(m, 0)
        
        ramadan_months = {
            (2026, 2), (2026, 3), (2027, 2), (2028, 1), (2028, 2), (2029, 1), (2030, 1), (2030, 12)
        }
        history_df.loc[idx, 'is_ramadan'] = 1 if (y, m) in ramadan_months else 0
        history_df.loc[idx, 'is_special_event'] = 0
        
        for var in ['Oil_price', 'FDI', 'Poverty_rate', 'REER']:
            history_df.loc[idx, f'{var}_lag1'] = history_df[var].iloc[-2]
            history_df.loc[idx, f'{var}_lag3'] = history_df[var].iloc[-4]
            
        history_df.loc[idx, 'cdm_event'] = 1 if y == 2030 and m in [6, 7] else 0
        history_df.loc[idx, 'anomaly_zscore'] = 0
        history_df.loc[idx, 'anomaly_iforest'] = 0
        history_df.loc[idx, 'anomaly_prophet'] = 0
        
        # Vecteur de caractéristiques
        feat_vec = history_df[valid_features].iloc[[-1]].fillna(0)
        
        # Prédiction
        pred = model.predict(feat_vec)[0]
        pred = max(0.0, pred)
        
        history_df.loc[idx, target_col] = pred
        predictions.append(pred)
        
    return np.array(predictions)

def forecast_recursive_dl(model, df_historical, future_dates, valid_features, target_col='Arrivals'):
    """Prédiction récursive pas-à-pas pour les modèles Deep Learning (LSTM, RNN)."""
    
    # Assurer que les colonnes existent
    for col, default_val in zip(['Oil_price', 'FDI', 'Poverty_rate', 'REER'], [75.0, 2.0, 4.0, 100.0]):
        if col not in df_historical.columns:
            df_historical[col] = default_val

    history_df = df_historical[['Date', target_col, 'Oil_price', 'FDI', 'Poverty_rate', 'REER']].copy()
    history_df['Date'] = pd.to_datetime(history_df['Date'])
    
    last_row = df_historical.iloc[-1]
    last_oil = last_row.get('Oil_price', 75.0)
    last_fdi = last_row.get('FDI', 2.0)
    last_poverty = last_row.get('Poverty_rate', 4.0)
    last_reer = last_row.get('REER', 100.0)
    
    predictions = []
    
    for date in future_dates:
        new_row = {
            'Date': date,
            target_col: np.nan,
            'Oil_price': last_oil,
            'FDI': last_fdi,
            'Poverty_rate': last_poverty,
            'REER': last_reer,
            'is_covid': 1 if (date >= pd.to_datetime('2020-03-01') and date <= pd.to_datetime('2021-12-01')) else 0
        }
        
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        idx = history_df.index[-1]
        
        # Calculer les lags et features (identique à ML)
        for k in [1, 2, 6, 12]:
            history_df.loc[idx, f'lags_{k}'] = history_df[target_col].iloc[-1 - k]
        for w in [3, 6, 12]:
            history_df.loc[idx, f'roll_mean_{w}'] = history_df[target_col].iloc[-1-w:-1].mean()
            history_df.loc[idx, f'roll_std_{w}'] = history_df[target_col].iloc[-1-w:-1].std()
            
        val_prev = history_df[target_col].iloc[-2]
        val_yoy = history_df[target_col].iloc[-14]
        history_df.loc[idx, 'growth_yoy'] = ((val_prev - val_yoy) / (val_yoy if val_yoy != 0 else 1.0)) * 100
        
        m = date.month
        y = date.year
        q = (m - 1) // 3 + 1
        history_df.loc[idx, 'month_sin'] = np.sin(2 * np.pi * m / 12)
        history_df.loc[idx, 'month_cos'] = np.cos(2 * np.pi * m / 12)
        history_df.loc[idx, 'quarter'] = q
        history_df.loc[idx, 'year'] = y
        
        season = 0 if m in [12, 1, 2] else 1 if m in [3, 4, 5] else 2 if m in [6, 7, 8] else 3
        history_df.loc[idx, 'saison_1'] = 1 if season == 1 else 0
        history_df.loc[idx, 'saison_2'] = 1 if season == 2 else 0
        history_df.loc[idx, 'saison_3'] = 1 if season == 3 else 0
        
        history_df.loc[idx, 'is_summer'] = 1 if m in [6, 7, 8] else 0
        history_df.loc[idx, 'is_high_season'] = 1 if m in [4, 5, 7, 8, 10, 12] else 0
        history_df.loc[idx, 'is_vacances'] = 1 if m in [7, 8, 12] else 0
        
        holiday_map = {1: 2, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1, 8: 3, 9: 0, 10: 0, 11: 2, 12: 0}
        history_df.loc[idx, 'jours_feries_count'] = holiday_map.get(m, 0)
        
        ramadan_months = {
            (2026, 2), (2026, 3), (2027, 2), (2028, 1), (2028, 2), (2029, 1), (2030, 1), (2030, 12)
        }
        history_df.loc[idx, 'is_ramadan'] = 1 if (y, m) in ramadan_months else 0
        history_df.loc[idx, 'is_special_event'] = 0
        
        for var in ['Oil_price', 'FDI', 'Poverty_rate', 'REER']:
            history_df.loc[idx, f'{var}_lag1'] = history_df[var].iloc[-2]
            history_df.loc[idx, f'{var}_lag3'] = history_df[var].iloc[-4]
            
        history_df.loc[idx, 'cdm_event'] = 1 if y == 2030 and m in [6, 7] else 0
        history_df.loc[idx, 'anomaly_zscore'] = 0
        history_df.loc[idx, 'anomaly_iforest'] = 0
        history_df.loc[idx, 'anomaly_prophet'] = 0
        
        # Construire la séquence et normaliser
        last_window_df = history_df[valid_features].iloc[-model.window_size:].fillna(0)
        scaled_seq = model.scaler_x.transform(last_window_df.values)
        
        # Prédiction scaled
        pred_scaled = model.model.predict(scaled_seq[np.newaxis, :, :], verbose=0)[0, 0]
        
        # Inversion d'échelle
        pred = model.scaler_y.inverse_transform(np.array([[pred_scaled]]))[0, 0]
        pred = max(0.0, pred)
        
        history_df.loc[idx, target_col] = pred
        predictions.append(pred)
        
    return np.array(predictions)
