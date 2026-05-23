import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def build_features(df_clean):
    """
    Builds lag features, rolling statistics, seasonal/cyclical variables,
    special temporal/holiday indicators, and unsupervised anomaly detection flags.
    """
    df = df_clean.copy()
    
    # 1. Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Lag features (Arrivals target)
    target_col = 'Arrivals'
    lags = [1, 2, 6, 12]
    for k in lags:
        df[f'lags_{k}'] = df[target_col].shift(k)
        
    # 3. Rolling Features (shifted by 1 to prevent data leakage)
    windows = [3, 6, 12]
    for w in windows:
        # Rolling Mean
        df[f'roll_mean_{w}'] = (
            df[target_col]
            .shift(1)
            .rolling(window=w)
            .mean()
        )
        # Rolling Std (Volatility)
        df[f'roll_std_{w}'] = (
            df[target_col]
            .shift(1)
            .rolling(window=w)
            .std()
        )
        
    # 4. Growth YoY (lagged by 1 to prevent data leakage)
    df['growth_yoy'] = (
        (df[target_col].shift(1) - df[target_col].shift(13)) /
        df[target_col].shift(13).replace(0, np.nan) * 100
    ).fillna(0)
    
    # 5. Temporal components
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['quarter'] = df['Date'].dt.quarter
    
    # Cyclic encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Season mapping
    def get_season(m):
        if m in [12, 1, 2]: return 0  # Hiver
        elif m in [3, 4, 5]: return 1 # Printemps
        elif m in [6, 7, 8]: return 2 # Été
        else: return 3                # Automne

    df['season'] = df['month'].apply(get_season)
    
    # One-Hot Encoding for season (saison_1, saison_2, saison_3)
    df = pd.get_dummies(df, columns=['season'], prefix='saison', drop_first=True)
    # Ensure one-hot encoded columns are 0/1 integers
    for col in [c for c in df.columns if 'saison_' in c]:
        df[col] = df[col].astype(int)
        
    # --- ADDED: Additional Temporal & Holiday Features ---
    
    # A. Été (Summer: June, July, August)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    
    # B. Haute Saison (High Season: April, May, July, August, October, December)
    df['is_high_season'] = df['month'].isin([4, 5, 7, 8, 10, 12]).astype(int)
    
    # C. Vacances (Tourist/School Holiday peaks: July, August, December)
    df['is_vacances'] = df['month'].isin([7, 8, 12]).astype(int)
    
    # D. Jours Fériés (Fixed national holidays count per month in Morocco)
    # Jan: New Year (1), Manifesto (11) = 2
    # May: Labor Day (1) = 1
    # Jul: Throne Day (30) = 1
    # Aug: Oued Ed-Dahab (14), King's Revolution (20), Youth Day (21) = 3
    # Nov: Green March (6), Independence Day (18) = 2
    holiday_map = {1: 2, 2: 0, 3: 0, 4: 0, 5: 1, 6: 0, 7: 1, 8: 3, 9: 0, 10: 0, 11: 2, 12: 0}
    df['jours_feries_count'] = df['month'].map(holiday_map)
    
    # E. Ramadan moving lunar calendar lookup (1995 to 2026)
    ramadan_months = {
        (1995, 2), (1996, 1), (1996, 2), (1997, 1), (1997, 12), (1998, 1), (1998, 12), 
        (1999, 12), (2000, 11), (2000, 12), (2001, 11), (2002, 11), (2003, 10), (2003, 11), 
        (2004, 10), (2005, 10), (2006, 9), (2006, 10), (2007, 9), (2008, 9), (2009, 8), 
        (2009, 9), (2010, 8), (2011, 8), (2012, 7), (2012, 8), (2013, 7), (2014, 6), 
        (2014, 7), (2015, 6), (2016, 6), (2017, 5), (2017, 6), (2018, 5), (2019, 5), 
        (2020, 4), (2020, 5), (2021, 4), (2022, 4), (2023, 3), (2023, 4), (2024, 3), 
        (2025, 3), (2026, 2), (2026, 3)
    }
    df['is_ramadan'] = df.apply(lambda r: 1 if (r['year'], r['month']) in ramadan_months else 0, axis=1)
    
    # F. Événements Spéciaux (Flag major shocks/milestones: e.g. COP22 in Nov 2016, Earthquake Sep 2023)
    df['is_special_event'] = 0
    df.loc[(df['year'] == 2016) & (df['month'] == 11), 'is_special_event'] = 1  # COP22 Marrakech
    df.loc[(df['year'] == 2023) & (df['month'] == 9), 'is_special_event'] = 1   # Séisme d'Al Haouz
    
    # 6. Economic lagged/delayed variables (t-1 and t-3)
    eco_vars = ['Oil_price', 'FDI', 'Poverty_rate', 'GDP_Construction']
    for var in eco_vars:
        if var in df.columns:
            df[f'{var}_lag1'] = df[var].shift(1)
            df[f'{var}_lag3'] = df[var].shift(3)
            
    if 'REER' in df.columns:
        df['REER_lag1'] = df['REER'].shift(1)
        df['REER_lag3'] = df['REER'].shift(3)
        
    # 7. Event dummies
    # is_covid dummy: Mar 2020 to Dec 2021
    df['is_covid'] = 0
    df.loc[(df['Date'] >= '2020-03-01') & (df['Date'] <= '2021-12-01'), 'is_covid'] = 1
    
    # World Cup Event (cdm_event)
    df['cdm_event'] = 0
    # WC 2026: Dec 2025 - Jan 2026
    df.loc[(df['Date'] >= '2025-12-01') & (df['Date'] <= '2026-01-31'), 'cdm_event'] = 1
    # WC 2030: Jun 2030 - Jul 2030 (if exists)
    df.loc[(df['Date'] >= '2030-06-01') & (df['Date'] <= '2030-07-31'), 'cdm_event'] = 1
    
    # --- ADDED: Anomaly Detection Features ---
    
    # A. Z-Score anomalies (calculated on target first differences to highlight sudden jumps/crises)
    y_diff = df[target_col].diff().fillna(0)
    mean_diff = y_diff.mean()
    std_diff = y_diff.std()
    z_scores = (y_diff - mean_diff) / (std_diff if std_diff != 0 else 1)
    df['anomaly_zscore'] = (np.abs(z_scores) > 2.2).astype(int)
    
    # B. Isolation Forest anomalies
    # Train Isolation Forest on Arrivals and key economic indicators
    features_for_if = [target_col]
    for col in ['Oil_price', 'FDI', 'REER']:
        if col in df.columns:
            features_for_if.append(col)
    
    # Impute temporary missing values to feed into Isolation Forest
    temp_data = df[features_for_if].interpolate().bfill().ffill()
    clf = IsolationForest(contamination=0.08, random_state=42)
    # Output: -1 for outliers, 1 for inliers
    iforest_preds = clf.fit_predict(temp_data)
    df['anomaly_iforest'] = (iforest_preds == -1).astype(int)
    
    # C. Prophet-like anomalies (Rolling Residuals from Trend + Seasonality)
    # We estimate trend using rolling average and season using month means,
    # then flag residuals exceeding 2.5 standard deviations.
    rolling_trend = df[target_col].rolling(window=12, min_periods=1, center=True).mean()
    detrended = df[target_col] - rolling_trend
    seasonal_pattern = detrended.groupby(df['month']).transform('mean')
    residuals = detrended - seasonal_pattern
    res_std = residuals.std()
    df['anomaly_prophet'] = (np.abs(residuals) > (2.5 * res_std)).astype(int)
    
    return df

def get_feature_list():
    """
    Returns the updated list of features to be used for machine learning.
    """
    return [
        'lags_1', 'lags_2', 'lags_12',
        'roll_mean_3', 'roll_mean_6', 'roll_mean_12',
        'roll_std_3', 'roll_std_6', 'roll_std_12', 'growth_yoy',
        'month_sin', 'month_cos', 'quarter', 'year',
        'saison_1', 'saison_2', 'saison_3',
        'is_summer', 'is_high_season', 'is_vacances', 'jours_feries_count',
        'is_ramadan', 'is_special_event',
        'Oil_price_lag1', 'FDI_lag1', 'Poverty_rate_lag1', 'REER_lag1',
        'Oil_price_lag3', 'FDI_lag3', 'Poverty_rate_lag3', 'REER_lag3',
        'is_covid', 'cdm_event',
        'anomaly_zscore', 'anomaly_iforest', 'anomaly_prophet'
    ]
