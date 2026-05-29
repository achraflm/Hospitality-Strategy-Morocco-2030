import os
import io
import pandas as pd
from src.config import DATA_DIR, START_DATE, END_DATE

def load_csv_file(filename, fallback_path=None):
    """
    Helper function to load a CSV file from the configured data directory,
    checking subdirectories (processed, raw, external) for re-organization.
    """
    # 1. Try directly in DATA_DIR
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
        
    # 2. Try in processed/
    processed_path = os.path.join(DATA_DIR, 'processed', filename)
    if os.path.exists(processed_path):
        return pd.read_csv(processed_path)
        
    # 3. Try in raw/
    raw_path = os.path.join(DATA_DIR, 'raw', filename)
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
        
    # 4. Try in external/
    external_path = os.path.join(DATA_DIR, 'external', filename)
    if os.path.exists(external_path):
        return pd.read_csv(external_path)

    # 5. Try fallback_path
    if fallback_path and os.path.exists(fallback_path):
        return pd.read_csv(fallback_path)
        
    raise FileNotFoundError(f"Required file '{filename}' not found in {DATA_DIR} (or subdirs: processed, raw, external) or fallback.")

def load_morocco_cleaned():
    """Loads Morocco economic indicators data."""
    return load_csv_file('Morocco_cleaned.csv')

def load_maroc_tourism():
    """Loads Morocco tourism arrivals data."""
    return load_csv_file('maroc_tourism_2030_all_arrival_sources.csv')

def get_covid_data():
    """
    Creates and returns the dataframe containing the manual COVID data for 2020-2021.
    """
    covid_csv = """Annee,Mois,Arrivees,Recettes_Mrd_MAD
2020,Janvier,850000,6.5
2020,Fevrier,780000,5.8
2020,Mars,320000,3.2
2020,Avril,0,0.4
2020,Mai,0,0.3
2020,Juin,0,0.3
2020,Juillet,20000,0.8
2020,Aout,60000,1.2
2020,Septembre,80000,2.1
2020,Octobre,120000,3.1
2020,Novembre,150000,3.5
2020,Decembre,200000,4.2
2021,Janvier,100000,2.5
2021,Fevrier,90000,2.1
2021,Mars,110000,2.4
2021,Avril,130000,2.8
2021,Mai,150000,3.1
2021,Juin,450000,4.5
2021,Juillet,800000,7.2
2021,Aout,950000,8.4
2021,Septembre,350000,4.8
2021,Octobre,380000,5.1
2021,Novembre,120000,2.9
2021,Decembre,80000,1.8"""
    df_covid = pd.read_csv(io.StringIO(covid_csv))
    
    mois_map = {
        'Janvier': 1, 'Fevrier': 2, 'Mars': 3, 'Avril': 4, 'Mai': 5, 'Juin': 6,
        'Juillet': 7, 'Aout': 8, 'Septembre': 9, 'Octobre': 10, 'Novembre': 11, 'Decembre': 12
    }
    df_covid['Month_Num'] = df_covid['Mois'].map(mois_map)
    df_covid['Date'] = pd.to_datetime(
        df_covid[['Annee', 'Month_Num']]
        .assign(Day=1)
        .rename(columns={'Annee': 'year', 'Month_Num': 'month'})
    )
    df_covid['is_covid'] = 1
    df_covid['Total_Receipts_MDH'] = df_covid['Recettes_Mrd_MAD'] * 1000
    df_covid = df_covid.rename(columns={'Arrivees': 'Arrivals'})
    
    return df_covid[['Date', 'Arrivals', 'Total_Receipts_MDH', 'is_covid']]

def load_and_merge_tourism_data():
    """
    Loads economic and tourism source files, cleans them, merges them,
    and returns the outer-joined dataframe filtered for 1995-2026.
    """
    df_eco = load_morocco_cleaned()
    df_tourisme = load_maroc_tourism()
    
    # Clean tourism receipts
    df_tourisme = df_tourisme.drop(columns=['Arrivals_EHTC', 'Arrivals_WB'], errors='ignore')
    
    df_tourisme['Total_Receipts_MDH'] = df_tourisme['receipts_MHD'].fillna(df_tourisme['Recettes_Mensuelles_MDH'])
    df_tourisme = df_tourisme.drop(columns=['receipts_MHD', 'Recettes_Mensuelles_MDH'], errors='ignore')
    
    # Setup dates
    df_eco['Date'] = pd.to_datetime(df_eco['Date'])
    df_tourisme['Date'] = pd.to_datetime(df_tourisme['Date'])
    
    # Filter by date range
    df_eco_filtered = df_eco[(df_eco['Date'] >= START_DATE) & (df_eco['Date'] <= END_DATE)]
    df_tour_filtered = df_tourisme[(df_tourisme['Date'] >= START_DATE) & (df_tourisme['Date'] <= END_DATE)]
    
    # Merge
    merged_df = pd.merge(df_eco_filtered, df_tour_filtered, on='Date', how='outer')
    
    # Drop GDP_Construction_lag1 as it's often empty, or any other completely redundant columns
    merged_df = merged_df.drop('InterTourismeReceipts', axis=1, errors='ignore')
    
    return merged_df

def get_separated_data(target_col='Arrivals'):
    """
    Loads pre-split data from backend/data/separted/ and returns X_train, X_test, y_train, y_test
    along with their respective Dates assigned based on 1995-01-01 to 2022-12-01 (train) and 2023-01-01 to 2024-12-01 (test).
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "data", "separted"))
    
    X_train = pd.read_csv(os.path.join(base_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(base_dir, 'X_test.csv'))
    
    try:
        y_train_df = pd.read_csv(os.path.join(base_dir, 'y_train.csv'))
        y_test_df = pd.read_csv(os.path.join(base_dir, 'y_test.csv'))
        
        # Fallback to the first column if target_col doesn't exist
        tgt = target_col if target_col in y_train_df.columns else y_train_df.columns[0]
        y_train = y_train_df[tgt]
        y_test = y_test_df[tgt]
    except Exception as e:
        y_train = None
        y_test = None
        
    dates_train = pd.date_range(start='1995-01-01', periods=len(X_train), freq='MS')
    dates_test = pd.date_range(start='2023-01-01', periods=len(X_test), freq='MS')
    
    X_train['Date'] = dates_train
    X_test['Date'] = dates_test
    
    return X_train, X_test, y_train, y_test
