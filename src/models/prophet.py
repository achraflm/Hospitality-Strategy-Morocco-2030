import pandas as pd
from prophet import Prophet

class ProphetModel:
    def __init__(self):
        self.model = None
        self.target_col = None
        self.exog_cols = []
        
    def fit(self, df, target_col='Arrivals', exog_cols=None):
        self.target_col = target_col
        self.exog_cols = exog_cols if exog_cols is not None else []
        
        # Prepare df for Prophet: ds (date) and y (target)
        prophet_df = pd.DataFrame({
            'ds': df['Date'],
            'y': df[target_col]
        })
        for col in self.exog_cols:
            prophet_df[col] = df[col]
            
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        for col in self.exog_cols:
            self.model.add_regressor(col)
            
        self.model.fit(prophet_df)
        return self
        
    def predict(self, future_df):
        prophet_future = pd.DataFrame({
            'ds': future_df['Date']
        })
        for col in self.exog_cols:
            prophet_future[col] = future_df[col]
            
        forecast = self.model.predict(prophet_future)
        return forecast['yhat'].values
