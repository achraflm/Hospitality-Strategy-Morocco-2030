import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Dynamic imports with auto-install fallback
def import_or_install(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        return __import__(import_name)
    except ImportError:
        print(f"Warning: {package_name} is not installed. Attempting to install...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])
            return __import__(import_name)
        except Exception as e:
            print(f"Error: Failed to install {package_name}: {e}")
            return None

# Import tree models
xgb = import_or_install("xgboost", "xgboost")
lgb = import_or_install("lightgbm", "lightgbm")
cat = import_or_install("catboost", "catboost")

def get_pipeline(model):
    """
    Returns a pipeline with mean imputation, scaling, and the estimator
    to prevent data leakage during CV.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

def train_ridge(X_train, y_train, cv=3):
    """Trains a Ridge Regression model using GridSearch and TimeSeriesSplit."""
    model = Ridge(random_state=42)
    pipeline = get_pipeline(model)
    
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'model__fit_intercept': [True, False]
    }
    
    tscv = TimeSeriesSplit(n_splits=cv)
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid

def train_svr(X_train, y_train, cv=3):
    """Trains a Support Vector Regression (SVR) model using GridSearch."""
    model = SVR(kernel='rbf')
    pipeline = get_pipeline(model)
    
    param_grid = {
        'model__C': [0.1, 1.0, 10.0, 50.0],
        'model__epsilon': [0.01, 0.1, 1.0],
        'model__gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    tscv = TimeSeriesSplit(n_splits=cv)
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid

def train_xgboost(X_train, y_train, cv=3):
    """Trains an XGBoost Regressor model using GridSearch."""
    if xgb is None:
        print("XGBoost is not available.")
        return None
        
    model = xgb.XGBRegressor(random_state=42, objective='reg:squarederror')
    pipeline = get_pipeline(model)
    
    param_grid = {
        'model__max_depth': [3, 4, 5],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__n_estimators': [100, 200, 500],
        'model__subsample': [0.8, 1.0],
        'model__reg_alpha': [0, 0.1, 1],
        'model__reg_lambda': [1, 5]
    }
    
    tscv = TimeSeriesSplit(n_splits=cv)
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid

def train_lightgbm(X_train, y_train, cv=3):
    """Trains a LightGBM Regressor model using GridSearch."""
    if lgb is None:
        print("LightGBM is not available.")
        return None
        
    model = lgb.LGBMRegressor(random_state=42, verbose=-1)
    pipeline = get_pipeline(model)
    
    param_grid = {
        'model__max_depth': [3, 5, -1],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__n_estimators': [100, 200, 500],
        'model__num_leaves': [15, 31, 63],
        'model__subsample': [0.8, 1.0]
    }
    
    tscv = TimeSeriesSplit(n_splits=cv)
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid

def train_catboost(X_train, y_train, cv=3):
    """Trains a CatBoost Regressor model using GridSearch."""
    if cat is None:
        print("CatBoost is not available.")
        return None
        
    model = cat.CatBoostRegressor(random_state=42, verbose=0)
    pipeline = get_pipeline(model)
    
    param_grid = {
        'model__depth': [4, 6, 8],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__iterations': [100, 200, 500]
    }
    
    tscv = TimeSeriesSplit(n_splits=cv)
    grid = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid

class DifferencedModelWrapper:
    """
    Wrapper to train models on the first-differenced target variable
    and automatically reconstruct the predictions back to the original scale.
    """
    def __init__(self, base_model_trainer):
        self.base_model_trainer = base_model_trainer
        self.grid_ = None
        self.best_estimator_ = None
        
    def fit(self, X, y, cv=3):
        y_diff = y.diff().dropna()
        X_diff = X.iloc[1:]
        
        self.grid_ = self.base_model_trainer(X_diff, y_diff, cv=cv)
        if self.grid_:
            self.best_estimator_ = self.grid_.best_estimator_
        return self
        
    def predict(self, X_test, y_test_prev):
        if not self.best_estimator_:
            raise ValueError("Model is not fitted yet.")
            
        y_pred_diff = self.best_estimator_.predict(X_test)
        y_pred_reconstructed = y_test_prev + y_pred_diff
        return y_pred_reconstructed

# --- Feature Importance ---

def get_feature_importance(pipeline_model, feature_names):
    """
    Extracts feature importances or coefficients from a trained scikit-learn pipeline model.
    """
    estimator = pipeline_model.named_steps['model']
    
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
    elif hasattr(estimator, 'coef_'):
        importances = np.abs(estimator.coef_)
        if len(importances.shape) > 1:
            importances = importances.mean(axis=0)
    else:
        # For SVR or other models without clear feature importance, return empty or dummy
        return pd.Series(index=feature_names, dtype=float)
        
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)

# --- ADDED: Hybrid Forecasting ---

class HybridForecaster:
    """
    Combines predictions from multiple distinct time series models
    (e.g., SARIMA, XGBoost, LSTM) using a weighted average.
    """
    def __init__(self, weights=None):
        """
        weights: dict of {model_name: weight} summing to 1.
                 e.g. {'SARIMAX': 0.3, 'XGBoost': 0.4, 'LSTM': 0.3}
        """
        self.weights = weights
        
    def predict(self, model_predictions_dict):
        """
        Calculates the weighted average of the predictions.
        model_predictions_dict: dict of {model_name: 1D predictions array}
        """
        if self.weights is None:
            # Equal weighting by default
            num_models = len(model_predictions_dict)
            self.weights = {k: 1.0 / num_models for k in model_predictions_dict.keys()}
            
        # Normalize weights just in case
        total_weight = sum(self.weights.values())
        norm_weights = {k: v / total_weight for k, v in self.weights.items()}
        
        hybrid_pred = None
        for name, pred in model_predictions_dict.items():
            if name not in norm_weights:
                continue
            weighted_pred = np.array(pred) * norm_weights[name]
            if hybrid_pred is None:
                hybrid_pred = weighted_pred
            else:
                hybrid_pred += weighted_pred
                
        return hybrid_pred
