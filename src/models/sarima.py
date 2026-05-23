import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SarimaModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None
        
    def fit(self, y_train, exog_train=None):
        """
        Fits the SARIMAX model on historical target data and exogenous variables.
        """
        # Convert index to DatetimeIndex if not already
        if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
            y_data = y_train.values
        else:
            y_data = y_train
            
        exog_data = None
        if exog_train is not None:
            if isinstance(exog_train, pd.DataFrame) or isinstance(exog_train, pd.Series):
                exog_data = exog_train.values
            else:
                exog_data = exog_train
                
        model = SARIMAX(
            y_data,
            exog=exog_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.model_fit = model.fit(disp=False, maxiter=50)
        return self
        
    def predict(self, steps, exog_test=None):
        """
        Predicts future time steps.
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before making predictions.")
            
        exog_data = None
        if exog_test is not None:
            if isinstance(exog_test, pd.DataFrame) or isinstance(exog_test, pd.Series):
                exog_data = exog_test.values
            else:
                exog_data = exog_test
                
        start = len(self.model_fit.data.endog)
        end = start + steps - 1
        
        preds = self.model_fit.predict(start=start, end=end, exog=exog_data)
        return preds
