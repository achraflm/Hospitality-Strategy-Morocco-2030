import pandas as pd
import numpy as np
import os

def clean_hotel_bookings():
    """
    Cleans the hotel bookings dataset based on the logic in project.ipynb
    """
    print("--- Cleaning hotel_bookings.csv ---")
    input_path = 'hotel/hotel_bookings.csv'
    output_path = 'Clean/hotel_bookings_clean.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")

    # 1. Drop unnecessary columns
    df = df.drop(columns=['agent', 'company'], errors='ignore')

    # 2. Fill missing values
    df['country'] = df['country'].fillna('Unknown')
    df['children'] = df['children'].fillna(0)

    # 3. Filter out bad ADR data (Keep only prices > 0)
    df = df[df['adr'] > 0]

    # 4. Calculate total nights and filter
    df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
    df = df[df['total_nights'] > 0] 

    # 5. Calculate revenue
    # revenue = adr * (1 - is_canceled) * total_nights
    df['revenue'] = df['adr'] * (1 - df['is_canceled']) * df['total_nights']

    # 6. Date transformation
    # Fixed month_map (removed trailing spaces from notebook version to ensure matches)
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    # Ensure no leading/trailing spaces in month names
    df['arrival_date_month'] = df['arrival_date_month'].str.strip()
    df["month_num"] = df['arrival_date_month'].map(month_map)
    
    # Check for unmapped months
    if df['month_num'].isna().any():
        unmapped = df[df['month_num'].isna()]['arrival_date_month'].unique()
        print(f"Warning: Unmapped months found: {unmapped}")

    # Create date column
    df['date'] = pd.to_datetime(dict(
        year=df['arrival_date_year'],
        month=df['month_num'],
        day=df['arrival_date_day_of_month']
    ))

    # 7. Aggregation
    df_monthly = df.groupby('date').agg(
        adr_mean=('adr', 'mean'),
        occupancy_rate=('is_canceled', lambda x: 1 - x.mean()),
        total_revenue=('revenue', 'sum'),
        avg_stay=('total_nights', 'mean'),
        cancel_rate=('is_canceled', 'mean')
    ).reset_index()

    # Save
    os.makedirs('Clean', exist_ok=True)
    df_monthly.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")
    print(f"Final shape: {df_monthly.shape}")


def clean_tourism_hospitality():
    """
    Cleans the Tourism Hospitality Industry Analysis dataset.
    Note: project.ipynb only contains the loading step for this dataset.
    This logic is based on the observed output file 'Clean/hospitality_benchmark_clean . csv'.
    """
    print("\n--- Cleaning Tourism_Hospitality_Industry_Analysis.csv ---")
    input_path = 'hotel/Tourism_Hospitality_Industry_Analysis.csv'
    output_path = 'Clean/hospitality_benchmark_clean.csv'

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Original shape: {df.shape}")

    # Month mapping
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    # Standardize Month column
    if 'Month' in df.columns:
        df['month_num'] = df['Month'].str.strip().map(month_map)
        
        # Create date column (first day of month)
        df['date'] = pd.to_datetime(dict(
            year=df['Year'],
            month=df['month_num'],
            day=1
        ))

        # Rename and select columns to match expected output schema
        rename_map = {
            'Hotel_Occupancy_Rate': 'occupancy',
            'Tourism_Revenue_USD': 'revenue_usd',
            'Number_of_Hotels': 'nb_hotels',
            'Average_Length_of_Stay': 'avg_stay',
            'Tourist_Satisfaction_Score': 'satisfaction'
        }
        
        df_clean = df.rename(columns=rename_map)
        cols_to_keep = ['Country', 'date', 'occupancy', 'revenue_usd', 'nb_hotels', 'avg_stay', 'satisfaction']
        
        # Keep only available columns from the set
        available_cols = [c for c in cols_to_keep if c in df_clean.columns]
        df_clean = df_clean[available_cols]

        os.makedirs('Clean', exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"Saved cleaned data to {output_path}")
        print(f"Final shape: {df_clean.shape}")
    else:
        print("Error: 'Month' column not found in dataset.")

if __name__ == "__main__":
    clean_hotel_bookings()
    clean_tourism_hospitality()
