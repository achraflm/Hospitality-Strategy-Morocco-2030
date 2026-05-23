import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from src.data_loader import get_covid_data

def integrate_covid_data(merged_df):
    """
    Integrates manually collected COVID data (2020-2021) into the merged dataset.
    Overwrites Arrivals, Total_Receipts_MDH, and sets is_covid=1 for that period.
    """
    df_covid = get_covid_data()
    
    # Initialize is_covid column
    merged_df['is_covid'] = 0
    
    # Use update to override existing values
    merged_df_indexed = merged_df.set_index('Date')
    df_covid_indexed = df_covid.set_index('Date')
    
    merged_df_indexed.update(df_covid_indexed[['Arrivals', 'Total_Receipts_MDH', 'is_covid']])
    
    # Specifically ensure is_covid is set to 1 for the COVID range
    merged_df_indexed.loc['2020-03-01':'2021-12-01', 'is_covid'] = 1
    
    return merged_df_indexed.reset_index()

def reconstruct_historical_arrivals(merged_df, random_seed=42):
    """
    Reconstructs monthly Arrivals for 1996-2019 using seasonal patterns from 2022-2026
    and annual historical totals.
    """
    np.random.seed(random_seed)
    df_temp = merged_df.copy()
    
    # 1. Extract seasonal pattern and noise from recent data (2022-2026)
    df_recent = df_temp[(df_temp['Date'] >= '2022-01-01') & (df_temp['Date'] <= '2026-04-01')].copy()
    df_recent.set_index('Date', inplace=True)
    
    # Fill any NaNs for decomposition
    df_recent['Arrivals'] = df_recent['Arrivals'].interpolate(method='linear').bfill().ffill()
    
    decomp_recent = seasonal_decompose(df_recent['Arrivals'], model='multiplicative', period=12)
    seasonal_pattern = decomp_recent.seasonal.groupby(decomp_recent.seasonal.index.month).mean()
    seasonal_pattern = seasonal_pattern / seasonal_pattern.sum() * 12
    residual_std = decomp_recent.resid.dropna().std()
    
    # 2. Prepare historical reference (1996-2019)
    df_hist = df_temp[(df_temp['Date'] >= '1996-01-01') & (df_temp['Date'] <= '2019-12-31')].copy()
    
    # Calculate annual totals
    df_annual = df_hist.groupby(df_hist['Date'].dt.year)['Arrivals'].sum().reset_index()
    df_annual.columns = ['Year_Ref', 'Annual_Total']
    
    # Merge back
    df_hist['Year_Match'] = df_hist['Date'].dt.year
    df_hist = df_hist.merge(df_annual, left_on='Year_Match', right_on='Year_Ref', how='left')
    
    # Reconstruction function
    def reconstruct_monthly(row):
        month = row['Date'].month
        if pd.isna(row['Annual_Total']):
            return np.nan
        base_monthly_avg = row['Annual_Total'] / 12.0
        noise_factor = np.random.normal(1.0, residual_std)
        return max(0, base_monthly_avg * seasonal_pattern[month] * noise_factor)
    
    df_hist['Arrivals_Simulated'] = df_hist.apply(reconstruct_monthly, axis=1)
    
    # 3. Combine simulated historical and actual recent data
    df_hist_final = df_hist[['Date', 'Arrivals_Simulated']].rename(columns={'Arrivals_Simulated': 'Arrivals'})
    df_recent_final = df_temp[df_temp['Date'] >= '2020-01-01'][['Date', 'Arrivals']].copy()
    
    full_tourisme_data = pd.concat([df_hist_final, df_recent_final], axis=0).sort_values('Date').reset_index(drop=True)
    full_tourisme_data['Arrivals'] = full_tourisme_data['Arrivals'].interpolate(method='linear').bfill()
    
    # Integrate back into merged_df
    df_temp = df_temp.drop(columns=['Arrivals'], errors='ignore')
    df_temp = pd.merge(df_temp, full_tourisme_data, on='Date', how='left')
    
    # Final interpolation for the remaining 1995 dates
    df_temp['Arrivals'] = df_temp['Arrivals'].interpolate(method='linear').bfill().ffill()
    
    return df_temp

def reconstruct_historical_receipts(merged_df, random_seed=42):
    """
    Reconstructs monthly Total_Receipts_MDH for 1996-2019 using seasonal patterns from 2022-2026
    and annual historical totals.
    """
    np.random.seed(random_seed)
    df_temp = merged_df.copy()
    
    # Ensure receipts column is numeric and filled
    df_temp['Total_Receipts_MDH'] = pd.to_numeric(df_temp['Total_Receipts_MDH'], errors='coerce')
    df_temp['Total_Receipts_MDH'] = df_temp['Total_Receipts_MDH'].interpolate(method='linear').bfill().ffill()
    
    # 1. Extract seasonal pattern from recent receipts (2022-2026)
    df_recent_receipts = df_temp[(df_temp['Date'] >= '2022-01-01') & (df_temp['Date'] <= '2026-04-01')].copy()
    df_recent_receipts.set_index('Date', inplace=True)
    
    # Replace zeros with tiny positive to allow multiplicative decomposition
    df_recent_receipts['Total_Receipts_MDH_positive'] = df_recent_receipts['Total_Receipts_MDH'].replace(0, 0.1)
    
    if len(df_recent_receipts) >= 25:
        decomp_recent_receipts = seasonal_decompose(df_recent_receipts['Total_Receipts_MDH_positive'], model='multiplicative', period=12)
        seasonal_pattern_receipts = decomp_recent_receipts.seasonal.groupby(decomp_recent_receipts.seasonal.index.month).mean()
        seasonal_pattern_receipts = seasonal_pattern_receipts / seasonal_pattern_receipts.sum() * 12
        residual_std_receipts = decomp_recent_receipts.resid.dropna().std()
    else:
        seasonal_pattern_receipts = pd.Series([1.0]*12, index=range(1, 13))
        residual_std_receipts = 0.05
        
    # 2. Prepare historical reference (1996-2019)
    df_hist_receipts = df_temp[(df_temp['Date'] >= '1996-01-01') & (df_temp['Date'] <= '2019-12-31')].copy()
    df_annual_receipts = df_hist_receipts.groupby(df_hist_receipts['Date'].dt.year)['Total_Receipts_MDH'].sum().reset_index()
    df_annual_receipts.columns = ['Year_Ref', 'Annual_Total_Receipts']
    
    df_hist_receipts['Year_Match'] = df_hist_receipts['Date'].dt.year
    df_hist_receipts = df_hist_receipts.merge(df_annual_receipts, left_on='Year_Match', right_on='Year_Ref', how='left')
    
    # 3. Reconstruct
    def reconstruct_monthly_receipts(row):
        month = row['Date'].month
        if pd.isna(row['Annual_Total_Receipts']):
            return np.nan
        base_monthly_avg = row['Annual_Total_Receipts'] / 12.0
        noise_factor = np.random.normal(1.0, residual_std_receipts)
        return max(0, base_monthly_avg * seasonal_pattern_receipts[month] * noise_factor)
        
    df_hist_receipts['Total_Receipts_MDH_Simulated'] = df_hist_receipts.apply(reconstruct_monthly_receipts, axis=1)
    
    # 4. Combine
    df_hist_receipts_final = df_hist_receipts[['Date', 'Total_Receipts_MDH_Simulated']].rename(columns={'Total_Receipts_MDH_Simulated': 'Total_Receipts_MDH'})
    df_recent_receipts_final = df_temp[df_temp['Date'] >= '2020-01-01'][['Date', 'Total_Receipts_MDH']].copy()
    
    full_receipts_data = pd.concat([df_hist_receipts_final, df_recent_receipts_final], axis=0).sort_values('Date').reset_index(drop=True)
    
    # Update merged_df
    df_temp = df_temp.drop(columns=['Total_Receipts_MDH'], errors='ignore')
    df_temp = pd.merge(df_temp, full_receipts_data, on='Date', how='left')
    df_temp['Total_Receipts_MDH'] = df_temp['Total_Receipts_MDH'].interpolate(method='linear').bfill().ffill()
    
    return df_temp
