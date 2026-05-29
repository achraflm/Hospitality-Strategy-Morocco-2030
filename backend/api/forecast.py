import os
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Imports internes
import src.data_loader as loader
import src.cleaning as cleaner
import src.features as feat
import src.metrics as metrics_mod
from src.config import TARGET_COL
from src.forecasting import forecast_recursive_ml, forecast_recursive_dl

# Import des modèles
from src.models.sarima import SarimaModel
from src.models.ridge import RidgeModel
from src.models.lstm import LstmModel
from src.models.xgboost import XgboostModel
from src.models.lstm_cnn import LstmCnnModel

router = APIRouter()

# Schémas Pydantic
class MetricsRequest(BaseModel):
    split_year: int = 2023
    selected_features: List[str]
    selected_models: List[str]
    dl_epochs: int = 5

class PredictRequest(BaseModel):
    target_year: int = 2030
    selected_features: List[str]
    selected_models: List[str]
    dl_epochs: int = 5
    inflation_rate: float = 0.015
    wc_boost: float = 0.20

# Endpoints
@router.get("/models")
def get_available_models():
    return ["LSTM + CNN", "LSTM", "XGBoost"]

@router.get("/features")
def get_available_features():
    return feat.get_feature_list()

@router.post("/metrics")
def calculate_model_metrics(req: MetricsRequest):
    try:
        # 1. Charger les données pré-séparées
        X_train_sep, X_test_sep, y_train_sep, y_test_sep = loader.get_separated_data(TARGET_COL)
        
        # Filtre sur les caractéristiques demandées et imputation
        actual_features = [f for f in req.selected_features if f in X_train_sep.columns]
        X_train = X_train_sep[actual_features].fillna(X_train_sep[actual_features].median())
        X_test = X_test_sep[actual_features].fillna(X_test_sep[actual_features].median())
        
        y_train = y_train_sep
        y_test = y_test_sep
        
        predictions = {}
        
        # Entraînement
        if "SARIMA" in req.selected_models:
            predictions['SARIMA'] = SarimaModel().fit(y_train).predict(steps=len(y_test))
            
        ml_class_map = {
            'XGBoost': XgboostModel,
            'Ridge': RidgeModel
        }
        
        for ml_name in ml_class_map.keys():
            if ml_name in req.selected_models:
                predictions[ml_name] = ml_class_map[ml_name]().fit(X_train, y_train).predict(X_test)
                
        if "LSTM" in req.selected_models:
            lstm = LstmModel(epochs=req.dl_epochs).fit(X_train, y_train)
            predictions['LSTM'] = lstm.predict(X_test, X_train_history=X_train)
            
        if "LSTM + CNN" in req.selected_models:
            lstm_cnn = LstmCnnModel(epochs=req.dl_epochs).fit(X_train, y_train)
            predictions['LSTM + CNN'] = lstm_cnn.predict(X_test, X_train_history=X_train)
            
        # Calculer métriques
        results_df = metrics_mod.compare_models(predictions, y_test)
        
        # Formater pour le frontend
        records = results_df.to_dict(orient='records')
        
        # Préparer les tracés de comparaison
        chart_data = []
        test_dates = X_test_sep['Date'].dt.strftime('%Y-%m-%d').values
        
        for idx, date in enumerate(test_dates):
            row = {'Date': date, 'Actual': float(y_test.iloc[idx])}
            for model_name, preds in predictions.items():
                row[model_name] = float(preds[idx])
            chart_data.append(row)
            
        return {
            "metrics": records,
            "chartData": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
def run_forecasting_projections(req: PredictRequest):
    try:
        # 1. Charger les données pré-séparées
        X_train_sep, X_test_sep, y_train_sep, y_test_sep = loader.get_separated_data(TARGET_COL)
        
        # Concaténer pour obtenir l'historique complet
        actual_features = [f for f in req.selected_features if f in X_train_sep.columns]
        
        X_train_full = pd.concat([X_train_sep[actual_features], X_test_sep[actual_features]]).reset_index(drop=True)
        y_train_full = pd.concat([y_train_sep, y_test_sep]).reset_index(drop=True)
        
        X_train = X_train_full.fillna(X_train_full.median())
        y_train = y_train_full
        
        df_history = pd.concat([X_train_sep, X_test_sep]).reset_index(drop=True)
        df_history[TARGET_COL] = y_train_full
        # Date range futur
        future_dates = pd.date_range(start='2026-05-01', end=f'{req.target_year}-12-01', freq='MS')
        
        # Calcul de recettes théorique basé sur le ratio
        # Since df_history doesn't have Total_Receipts_MDH, we load the raw df quickly
        df = loader.load_and_merge_tourism_data()
        df = cleaner.reconstruct_historical_receipts(cleaner.integrate_covid_data(df))
        df_ml = df.dropna(subset=[TARGET_COL])
        mean_ratio = (df_ml['Total_Receipts_MDH'] / df_ml['Arrivals']).mean() if 'Total_Receipts_MDH' in df_ml.columns else 4.0
        
        def calculate_receipts(arrivals):
            receipts = []
            for arr, date in zip(arrivals, future_dates):
                years_since_2026 = date.year - 2026
                ratio = mean_ratio * ((1 + req.inflation_rate) ** years_since_2026)
                if date.year == 2030:
                    ratio = ratio * (1 + req.wc_boost)
                receipts.append(float(arr * ratio))
            return receipts
            
        projections = {}
        
        # Projections ML
        ml_class_map = {
            'Ridge': RidgeModel,
            'XGBoost': XgboostModel
        }
        
        # Trouver le meilleur modèle ML sélectionné
        for ml_name in ml_class_map.keys():
            if ml_name in req.selected_models:
                best_ml_model = ml_class_map[ml_name]().fit(X_train, y_train)
                arr_ml = forecast_recursive_ml(best_ml_model, df_ml, future_dates, req.selected_features)
                arr_ml = np.clip(arr_ml, 0, None)
                projections[ml_name] = {
                    "arrivals": arr_ml.tolist(),
                    "receipts": calculate_receipts(arr_ml)
                }
                
        # Projections DL
        dl_class_map = {
            'LSTM': LstmModel,
            'LSTM + CNN': LstmCnnModel
        }
        for dl_name in dl_class_map.keys():
            if dl_name in req.selected_models:
                best_dl_model = dl_class_map[dl_name](epochs=req.dl_epochs).fit(X_train, y_train)
                arr_dl = forecast_recursive_dl(best_dl_model, df_ml, future_dates, req.selected_features)
                arr_dl = np.clip(arr_dl, 0, None)
                projections[dl_name] = {
                    "arrivals": arr_dl.tolist(),
                    "receipts": calculate_receipts(arr_dl)
                }
                
        # Formater pour Recharts
        formatted_projections = []
        historical_subset = df_ml.tail(36).copy()
        
        # Ajouter l'historique récent
        for idx, row in historical_subset.iterrows():
            formatted_projections.append({
                "Date": row['Date'].strftime('%Y-%m-%d'),
                "IsFuture": False,
                "Actual_Arrivals": float(row['Arrivals']),
                "Actual_Receipts": float(row['Total_Receipts_MDH'])
            })
            
        # Ajouter les points futurs
        future_str_dates = future_dates.strftime('%Y-%m-%d').values
        for idx, date in enumerate(future_str_dates):
            point = {
                "Date": date,
                "IsFuture": True
            }
            for model_name, data in projections.items():
                point[f"{model_name}_Arrivals"] = data["arrivals"][idx]
                point[f"{model_name}_Receipts"] = data["receipts"][idx]
            formatted_projections.append(point)
            
        return {
            "projections": formatted_projections,
            "models": list(projections.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
